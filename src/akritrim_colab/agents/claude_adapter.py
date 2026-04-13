from __future__ import annotations

import asyncio
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .stream_events import StreamEvent


STUCK_PATTERN = re.compile(r"\b(?:i|we)\s+got\s+stuck\b", re.IGNORECASE)


def _write_stream_debug(
    debug_log: Path,
    session_id: str | None,
    payload_type_counts: dict[str, int],
    text_delta_chars: int,
    text_source: str,
    exit_code: int,
    stderr_snippet: str,
) -> None:
    """Append one JSONL entry per turn recording which payload types the CLI emitted.

    Answers post-hoc: did text_delta / assistant events fire, or was 'result'
    the only source of text for this turn?

    Usage: ``jq . .collab/session/stream_debug.jsonl``
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "payload_types": payload_type_counts,
        "text_delta_chars": text_delta_chars,
        "text_source": text_source,
        "exit_code": exit_code,
        "stderr_snippet": stderr_snippet,
    }
    try:
        debug_log.parent.mkdir(parents=True, exist_ok=True)
        with debug_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # never let debug writes break the turn


_SHELL_TOOL_NAMES = frozenset({"bash", "computer", "execute_bash", "run_bash", "shell"})


def _format_tool_event(tool_name: str, input_json: str) -> str:
    """Render a tool-use block as a human-readable one-liner for the Claude pane.

    Shell-like tools (bash, computer) that carry a ``command`` key are formatted
    as ``$ command`` so dangerous-op patterns in the app match them with the same
    regex used for Codex ``local_shell_call`` events.
    """
    try:
        args = json.loads(input_json) if input_json.strip() else {}
    except (json.JSONDecodeError, ValueError):
        args = {}
    if tool_name.lower() in _SHELL_TOOL_NAMES and isinstance(args, dict):
        cmd = args.get("command") or args.get("cmd") or args.get("input") or ""
        if cmd:
            return f"$ {cmd}"
    truncated = input_json[:120] + "\u2026" if len(input_json) > 120 else input_json
    return f"\u2699 {tool_name}({truncated})"


@dataclass
class ClaudeReply:
    text: str
    exit_code: int
    failed: bool
    stuck_hint: bool
    session_id: str | None
    command: list[str]
    stderr: str = ""


class ClaudeAdapter:
    def __init__(self, cwd: Path, model: str | None = None, debug_log: Path | None = None) -> None:
        self.cwd = Path(cwd)
        self.model = model
        self.debug_log = debug_log

    def _base_command(self) -> list[str]:
        cmd = ["claude", "-p", "--dangerously-skip-permissions"]
        if self.model:
            cmd.extend(["--model", self.model])
        return cmd

    def send(self, prompt: str, session_id: str | None = None) -> ClaudeReply:
        command = self._base_command()
        if session_id:
            command.extend(["--resume", session_id])
        command.append(prompt)

        result = subprocess.run(
            command,
            cwd=self.cwd,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        text = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        failed = result.returncode != 0
        stuck_hint = bool(STUCK_PATTERN.search(text) or STUCK_PATTERN.search(stderr))

        if failed and not text:
            text = stderr or "Claude adapter failed without stdout."

        return ClaudeReply(
            text=text,
            exit_code=result.returncode,
            failed=failed,
            stuck_hint=stuck_hint,
            session_id=session_id,
            command=command,
            stderr=stderr,
        )

    async def stream_send(self, prompt: str, session_id: str | None = None):
        command = self._base_command() + [
            "--verbose",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
        ]
        if session_id:
            command.extend(["--resume", session_id])
        command.append(prompt)

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self.cwd),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        emitted_session = False
        # Cache module-level helper to work around Pyright false positive inside async generators.
        _fmt_tool = _format_tool_event
        # Debug tracking — counts every payload type seen this turn.
        payload_type_counts: dict[str, int] = {}
        text_delta_chars = 0
        text_source = "none"   # which path ultimately provided text
        # Tool-use block accumulator — tracks the tool call being streamed.
        # Claude CLI streams content_block_start(tool_use) → input_json_delta* → content_block_stop
        # BEFORE executing the tool.  We emit tool_event at content_block_stop so the app can
        # intercept and abort before any execution happens.
        current_tool_name: str | None = None
        current_tool_input_json: str = ""

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

                ptype = payload.get("type", "<unknown>") if isinstance(payload, dict) else "<non-dict>"
                payload_type_counts[ptype] = payload_type_counts.get(ptype, 0) + 1

                if not emitted_session:
                    detected_session = self._extract_session_id(payload)
                    if detected_session:
                        yield StreamEvent(agent="claude", kind="session_started", session_handle=detected_session)
                        emitted_session = True

                if payload.get("type") == "stream_event":
                    event = payload.get("event", {})
                    etype = event.get("type", "")

                    if etype == "content_block_start":
                        cb = event.get("content_block") or {}
                        if cb.get("type") == "tool_use":
                            current_tool_name = cb.get("name") or "tool"
                            current_tool_input_json = ""

                    elif etype == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            text_delta_chars += len(delta["text"])
                            text_source = "text_delta"
                            yield StreamEvent(agent="claude", kind="text_delta", text=delta["text"])
                        elif delta.get("type") == "input_json_delta" and current_tool_name:
                            current_tool_input_json += delta.get("partial_json", "")

                    elif etype == "content_block_stop" and current_tool_name:
                        # Tool call fully streamed — emit tool_event BEFORE Claude CLI executes it.
                        label = _fmt_tool(current_tool_name, current_tool_input_json)
                        yield StreamEvent(agent="claude", kind="tool_event", text=label)
                        current_tool_name = None
                        current_tool_input_json = ""

                elif payload.get("type") == "assistant":
                    text = self._extract_assistant_text(payload)
                    if text:
                        if text_source == "none":
                            text_source = "assistant"
                        yield StreamEvent(agent="claude", kind="message_completed", text=text)
                elif payload.get("type") == "result":
                    # Claude CLI stream-json emits a final {"type":"result","result":"..."} payload.
                    # This is the authoritative text when no text_delta / assistant events fired.
                    result_text = payload.get("result", "")
                    if isinstance(result_text, str) and result_text.strip():
                        if text_source == "none":
                            text_source = "result"
                        yield StreamEvent(agent="claude", kind="message_completed", text=result_text.strip())
                    if not emitted_session:
                        detected_session = self._extract_session_id(payload)
                        if detected_session:
                            yield StreamEvent(agent="claude", kind="session_started", session_handle=detected_session)
                            emitted_session = True

            stderr = await process.stderr.read()
            exit_code = await process.wait()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            if self.debug_log:
                _write_stream_debug(
                    debug_log=self.debug_log,
                    session_id=session_id,
                    payload_type_counts=payload_type_counts,
                    text_delta_chars=text_delta_chars,
                    text_source=text_source,
                    exit_code=exit_code,
                    stderr_snippet=stderr_text[:200] if stderr_text else "",
                )
            if stderr_text:
                yield StreamEvent(agent="claude", kind="stderr", text=stderr_text)
            if exit_code != 0:
                yield StreamEvent(agent="claude", kind="failed", text=stderr_text, exit_code=exit_code)
            yield StreamEvent(agent="claude", kind="completed", exit_code=exit_code)
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
                buffer = buffer[newline + 1 :]
                yield line
        trailing = buffer.strip()
        if trailing:
            yield trailing

    def _extract_session_id(self, payload: object) -> str | None:
        if isinstance(payload, dict):
            for key in ("session_id", "sessionId"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for value in payload.values():
                found = self._extract_session_id(value)
                if found:
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = self._extract_session_id(item)
                if found:
                    return found
        return None

    def _extract_assistant_text(self, payload: dict) -> str:
        message = payload.get("message") or {}
        content = message.get("content") or []
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                texts.append(item["text"])
        return "".join(texts).strip()
