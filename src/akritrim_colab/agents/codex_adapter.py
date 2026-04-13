from __future__ import annotations

import asyncio
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .stream_events import StreamEvent

STUCK_PATTERN = re.compile(r"\b(?:i|we)\s+got\s+stuck\b", re.IGNORECASE)
UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
PREFERRED_ID_KEYS = {
    "session_id",
    "sessionId",
    "conversation_id",
    "conversationId",
    "thread_id",
    "threadId",
    "run_id",
    "runId",
    "id",
}


@dataclass
class CodexReply:
    text: str
    exit_code: int
    failed: bool
    stuck_hint: bool
    resume_handle: str | None
    command: list[str]
    stderr: str = ""


class CodexAdapter:
    def __init__(self, cwd: Path, model: str | None = None, skip_git_check: bool = False) -> None:
        self.cwd = Path(cwd)
        self.model = model
        self.skip_git_check = skip_git_check

    def send(self, prompt: str, resume_handle: str | None = None) -> CodexReply:
        with tempfile.NamedTemporaryFile(prefix="codex-last-message-", suffix=".txt") as tmp:
            command = self._build_command(prompt=prompt, resume_handle=resume_handle, output_file=Path(tmp.name))
            result = subprocess.run(
                command,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                check=False,
            )

            text = Path(tmp.name).read_text(encoding="utf-8").strip()
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            failed = result.returncode != 0

            if not text:
                text = self._extract_text_from_jsonl(stdout)
            if failed and not text:
                text = stderr or "Codex adapter failed without stdout."

            detected_handle = self._extract_resume_handle(stdout) or resume_handle
            if not detected_handle and result.returncode == 0:
                detected_handle = "__last__"

            stuck_hint = bool(
                STUCK_PATTERN.search(text)
                or STUCK_PATTERN.search(stderr)
                or (not text and result.returncode == 0)
            )

            return CodexReply(
                text=text,
                exit_code=result.returncode,
                failed=failed,
                stuck_hint=stuck_hint,
                resume_handle=detected_handle,
                command=command,
                stderr=stderr,
            )

    def _build_command(self, prompt: str, resume_handle: str | None, output_file: Path) -> list[str]:
        base = ["codex"]
        if self.model:
            base.extend(["--model", self.model])
        if self.skip_git_check:
            base.append("--skip-git-repo-check")
        if resume_handle:
            base.extend(["exec", "resume"])
            if resume_handle == "__last__":
                base.append("--last")
            else:
                base.append(resume_handle)
        else:
            base.append("exec")
        base.extend(["--json", "-o", str(output_file), prompt])
        return base

    def _extract_text_from_jsonl(self, stdout: str) -> str:
        lines: list[str] = []
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = self._find_message_text(payload)
            if text:
                lines.append(text)
        return "\n".join(lines).strip()

    def _extract_resume_handle(self, stdout: str) -> str | None:
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            value = self._find_identifier(payload)
            if value:
                return value
        return None

    def _find_identifier(self, payload: Any) -> str | None:
        if isinstance(payload, dict):
            for key in PREFERRED_ID_KEYS:
                value = payload.get(key)
                if isinstance(value, str):
                    m = UUID_PATTERN.search(value)
                    if m:
                        return m.group(0)
            for value in payload.values():
                found = self._find_identifier(value)
                if found:
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = self._find_identifier(item)
                if found:
                    return found
        elif isinstance(payload, str):
            match = UUID_PATTERN.search(payload)
            if match:
                return match.group(0)
        return None

    def _find_message_text(self, payload: Any) -> str | None:
        if isinstance(payload, dict):
            for key in ("last_message", "lastMessage", "message", "content", "text"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for value in payload.values():
                found = self._find_message_text(value)
                if found:
                    return found
        elif isinstance(payload, list):
            parts: list[str] = []
            for item in payload:
                found = self._find_message_text(item)
                if found:
                    parts.append(found)
            if parts:
                return "\n".join(parts).strip()
        return None

    async def stream_send(self, prompt: str, resume_handle: str | None = None):
        command = ["codex"]
        if self.model:
            command.extend(["--model", self.model])
        if self.skip_git_check:
            command.append("--skip-git-repo-check")
        if resume_handle:
            command.extend(["exec", "resume", resume_handle])
        else:
            command.append("exec")
        command.extend(["--json", prompt])

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        try:
            async for line in self._iter_json_lines(process.stdout):
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = payload.get("type") or ""
                item = payload.get("item") or {}
                item_type = item.get("type") or ""

                # ── Session identity ──────────────────────────────────────
                if event_type == "thread.started":
                    thread_id = payload.get("thread_id") or item.get("thread_id")
                    if thread_id:
                        yield StreamEvent(
                            agent="codex", kind="session_started", session_handle=thread_id
                        )

                # ── Progressive streaming delta (if Codex CLI emits it) ───
                elif event_type == "item.delta":
                    delta = item.get("delta") or {}
                    delta_text = delta.get("text") or ""
                    if not delta_text and isinstance(delta.get("content"), list):
                        delta_text = "".join(
                            c.get("text", "") for c in delta["content"]
                            if isinstance(c, dict)
                        )
                    if delta_text:
                        yield StreamEvent(agent="codex", kind="text_delta", text=delta_text)

                # ── Completed item ────────────────────────────────────────
                elif event_type == "item.completed":
                    if item_type == "agent_message":
                        text = item.get("text") or item.get("content") or ""
                        if isinstance(text, list):
                            text = "".join(
                                c.get("text", "") for c in text if isinstance(c, dict)
                            )
                        text = str(text).strip()
                        if text:
                            # Emit text_delta so active_text accumulates and pane updates
                            # immediately, then message_completed to mark relay text.
                            yield StreamEvent(agent="codex", kind="text_delta", text=text)
                            yield StreamEvent(agent="codex", kind="message_completed", text=text)
                    else:
                        label = self._format_tool_item(item_type, item)
                        if label:
                            yield StreamEvent(agent="codex", kind="tool_event", text=label)

            stderr_bytes = await process.stderr.read()
            exit_code = await process.wait()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            if stderr_text:
                yield StreamEvent(agent="codex", kind="stderr", text=stderr_text)
            if exit_code != 0:
                yield StreamEvent(agent="codex", kind="failed", text=stderr_text, exit_code=exit_code)
            yield StreamEvent(agent="codex", kind="completed", exit_code=exit_code)
        except asyncio.CancelledError:
            raise
        finally:
            # Terminate the subprocess on any exit path — including consumer-side break
            # (GeneratorExit via aclose()) and CancelledError.
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

    def _format_tool_item(self, item_type: str, item: dict) -> str:
        """Render a non-text item as a human-readable one-liner for the Codex pane."""
        if item_type in ("function_call", "tool_call"):
            name = item.get("name") or item_type
            args = item.get("arguments") or item.get("parameters") or ""
            if isinstance(args, (dict, list)):
                args = json.dumps(args, ensure_ascii=False)
            args = str(args)
            if len(args) > 120:
                args = args[:120] + "\u2026"
            return f"\u2699 {name}({args})"

        if item_type in ("function_call_output", "function_call_result", "tool_result"):
            output = item.get("output") or item.get("content") or item.get("result") or ""
            if isinstance(output, (dict, list)):
                output = json.dumps(output, ensure_ascii=False)
            output = str(output).strip()
            if len(output) > 300:
                output = output[:300] + "\u2026"
            return f"\u2192 {output}" if output else "\u2192 (no output)"

        if item_type == "local_shell_call":
            action = item.get("action") or {}
            if isinstance(action, dict):
                cmd_list = action.get("command") or []
                cmd = " ".join(str(c) for c in cmd_list) if cmd_list else str(action)
            else:
                cmd = item.get("command") or item.get("cmd") or str(action) or ""
            return f"$ {cmd}" if cmd else "$ (shell)"

        if item_type == "local_shell_call_result":
            output = item.get("output") or ""
            exit_code = item.get("exit_code")
            output = str(output).strip()
            if len(output) > 300:
                output = output[:300] + "\u2026"
            if output:
                return f"\u2192 {output}"
            return f"\u2192 exit={exit_code}" if exit_code is not None else "\u2192 (done)"

        if item_type == "reasoning":
            text = item.get("text") or item.get("content") or ""
            if not text and isinstance(item.get("summary"), list):
                text = " ".join(
                    s.get("text", "") for s in item["summary"] if isinstance(s, dict)
                )
            text = str(text).strip()
            if len(text) > 200:
                text = text[:200] + "\u2026"
            return f"\U0001f4ad {text}" if text else ""

        # Generic fallback: extract any embedded text
        text = self._find_message_text(item)
        if text:
            if len(text) > 200:
                text = text[:200] + "\u2026"
            return f"[{item_type}] {text}"
        return ""

    async def _iter_json_lines(self, stream: asyncio.StreamReader):
        buffer = ""
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            while True:
                newline = buffer.find("\n")
                if newline == -1:
                    break
                line = buffer[:newline].strip()
                buffer = buffer[newline + 1:]
                yield line
        trailing = buffer.strip()
        if trailing:
            yield trailing
