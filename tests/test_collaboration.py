"""
Tests for collaboration handoff and Codex incomplete reply handling.

Scenario 1 — Handoff collaboration:
  When @both dispatches Codex (implementer) then Claude (reviewer),
  Claude's prompt must include Codex's completed reply as sequential
  conditioning. The first agent must receive no prior-reply block.

Scenario 2 — Codex incomplete reply:
  When Codex's stream yields no text content, the fallback message is
  recorded. When it yields a `failed` event, stuck is set. tool_event
  entries must never leak into the relay text.

Scenario 4 — Claude adapter `result` payload fallback (D-049):
  Claude CLI stream-json may emit only a final {"type":"result","result":"..."}
  payload with no prior text_delta or assistant events. The adapter must surface
  this as a message_completed event so _consume_stream captures it instead of
  falling back to "Claude returned no text."
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from akritrim_colab.agents.claude_adapter import ClaudeAdapter
from akritrim_colab.agents.stream_events import StreamEvent
from akritrim_colab.app import (
    CollabApp, RouteCommand, SessionState, THEMES,
    _expand_file_mentions, _file_completions, DANGEROUS_OP_PATTERNS,
)
from akritrim_colab.config import ProjectConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state(**overrides) -> SessionState:
    defaults = dict(
        active_identity="claude",
        role="collaborator",
        codex_role="implementer",
        claude_role="reviewer",
        mode=None,
        turn_index=1,
        next_speaker="both",
        waiting_for_master=False,
        collaboration_active=False,  # keep dispatch tests isolated
        stuck=False,
        session_log_path="",
    )
    defaults.update(overrides)
    return SessionState(**defaults)


def _config(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "session").mkdir(parents=True)
    (tmp_path / "session" / "current.md").write_text("# Transcript\n\n")
    return ProjectConfig(
        project_root=tmp_path,
        memory_path=tmp_path / "memory.md",
        scratchpad_path=tmp_path / "scratchpad.md",
        rules_path=tmp_path / "rules.md",
        session_dir=tmp_path / "session",
        settings_path=tmp_path / "settings.toml",
    )


def _fake_app(tmp_path: Path, **state_overrides) -> types.SimpleNamespace:
    """Minimal namespace with attributes needed by _consume_stream and _dispatch_route."""
    config = _config(tmp_path)
    state = _state(**state_overrides)
    app = types.SimpleNamespace()
    app.state = state
    app.config = config
    app.transcript_path = config.session_dir / "current.md"
    app.state_path = config.session_dir / "state.json"
    # Agent runtime state
    app.latest_completed_reply = {"claude": "", "codex": ""}
    app.active_text = {"claude": "", "codex": ""}
    app.active_message = {"claude": "", "codex": ""}
    app.active_speaker = {"claude": "Claude", "codex": "Codex"}
    app.pane_history = {"claude": [], "codex": []}
    app.agent_busy = {"claude": False, "codex": False}
    # Silence all Textual / UI calls
    app._write_pane = MagicMock()
    app._refresh_status = MagicMock()
    app._refresh_activity_indicator = MagicMock()
    app._write_chat = MagicMock()
    app._write_transcript = MagicMock()
    app._surface_master_request = AsyncMock()  # now async — must be AsyncMock
    app._debug = MagicMock()
    app.persist_state = MagicMock()
    # Bind pure helpers that only touch state / pane_history, not Textual widgets
    app._apply_stream_event = lambda e, p: CollabApp._apply_stream_event(app, e, p)
    app._write_system = lambda p, m: CollabApp._write_system(app, p, m)
    app._role_constraint = lambda role: CollabApp._role_constraint(app, role)
    # _dangerous_ops_policy is called inside _role_preamble; wire it with an empty list so
    # tests don't depend on the default ops set (avoids brittle text matching).
    app.dangerous_ops = []
    app._active_approvals = set()
    app._seen_post_exec = set()
    app._last_agent_message = {"claude": "", "codex": ""}
    app._dangerous_ops_policy = lambda: CollabApp._dangerous_ops_policy(app)
    app._check_dangerous_op = lambda text: CollabApp._check_dangerous_op(app, text)
    app._handle_dangerous_op_abort = AsyncMock()
    app._role_preamble = lambda n, r: CollabApp._role_preamble(app, n, r)
    app._last_turn_changed_files = {"claude": [], "codex": []}
    app._master_request_handled = False
    app._pending_wfm_pane = None
    app._format_prior_replies = lambda rs: CollabApp._format_prior_replies(app, rs)
    app._format_prior_changed_files = lambda ts: CollabApp._format_prior_changed_files(app, ts)
    app._consume_collab_dirty_notice = MagicMock(return_value="")
    app._collab_rules_reminder = MagicMock(return_value="")
    return app


async def _stream(*events: StreamEvent):
    """Async generator yielding a fixed sequence of StreamEvents. Pass no args for empty."""
    for event in events:
        yield event


# ── Scenario 1: Handoff collaboration ─────────────────────────────────────────

class TestHandoff:
    """Sequential dispatch — second agent receives first agent's completed reply."""

    @pytest.mark.asyncio
    async def test_codex_runs_first_when_codex_is_implementer(self, tmp_path):
        """Dispatch order: codex_role=implementer → Codex before Claude."""
        app = _fake_app(tmp_path, codex_role="implementer", claude_role="reviewer")
        call_order: list[str] = []

        async def mock_codex(msg, final, next_speaker):
            call_order.append("codex")
            app.latest_completed_reply["codex"] = "codex done"

        async def mock_claude(msg, final, next_speaker):
            call_order.append("claude")

        app._invoke_codex = mock_codex
        app._invoke_claude = mock_claude
        app._run_collaboration_loop = AsyncMock()

        await CollabApp._dispatch_route(
            app, RouteCommand(kind="route", target="both", message="Implement X.")
        )

        assert call_order == ["codex", "claude"], (
            f"Expected codex first, got {call_order}"
        )

    @pytest.mark.asyncio
    async def test_second_agent_receives_first_agent_reply(self, tmp_path):
        """Claude's prompt includes Codex's completed reply as conditioning."""
        app = _fake_app(tmp_path, codex_role="implementer", claude_role="reviewer")
        codex_reply = "I've applied the patch to auth.py."
        captured_claude_msg: list[str] = []

        async def mock_codex(msg, final, next_speaker):
            app.latest_completed_reply["codex"] = codex_reply

        async def mock_claude(msg, final, next_speaker):
            captured_claude_msg.append(msg)

        app._invoke_codex = mock_codex
        app._invoke_claude = mock_claude
        app._run_collaboration_loop = AsyncMock()

        await CollabApp._dispatch_route(
            app, RouteCommand(kind="route", target="both", message="Fix the login bug.")
        )

        assert captured_claude_msg, "Claude was never invoked"
        assert codex_reply in captured_claude_msg[0], (
            "Claude's prompt must contain Codex's completed reply"
        )
        assert "Codex replied:" in captured_claude_msg[0], (
            "Prior-reply block must have the 'Codex replied:' header"
        )

    @pytest.mark.asyncio
    async def test_first_agent_has_no_prior_conditioning(self, tmp_path):
        """The first dispatched agent gets no prior-reply block."""
        app = _fake_app(tmp_path, codex_role="implementer", claude_role="reviewer")
        captured_codex_msg: list[str] = []

        async def mock_codex(msg, final, next_speaker):
            captured_codex_msg.append(msg)
            app.latest_completed_reply["codex"] = "codex reply"

        async def mock_claude(msg, final, next_speaker):
            pass

        app._invoke_codex = mock_codex
        app._invoke_claude = mock_claude
        app._run_collaboration_loop = AsyncMock()

        await CollabApp._dispatch_route(
            app, RouteCommand(kind="route", target="both", message="Task.")
        )

        assert " replied:" not in captured_codex_msg[0], (
            "First agent must not receive any prior-reply conditioning"
        )

    @pytest.mark.asyncio
    async def test_single_agent_target_skips_conditioning(self, tmp_path):
        """@claude target — Claude receives no prior-reply block."""
        app = _fake_app(tmp_path)
        captured: list[str] = []

        async def mock_claude(msg, final, next_speaker):
            captured.append(msg)

        app._invoke_claude = mock_claude
        app._run_collaboration_loop = AsyncMock()

        await CollabApp._dispatch_route(
            app, RouteCommand(kind="route", target="claude", message="Explain auth.py.")
        )

        assert captured, "Claude was never invoked"
        assert " replied:" not in captured[0]


# ── Scenario 2: Codex incomplete / failed reply ────────────────────────────────

class TestCodexIncompleteReply:
    """Stream-level handling of empty or failed Codex output."""

    @pytest.mark.asyncio
    async def test_empty_stream_sets_fallback_message(self, tmp_path):
        """No text events → fallback 'Codex returned no text.' is recorded."""
        app = _fake_app(tmp_path)

        await CollabApp._consume_stream(
            app,
            pane="codex",
            speaker="Codex",
            stream=_stream(),
            final=True,
            next_speaker="master",
        )

        assert app.latest_completed_reply["codex"] == "Codex returned no text."

    @pytest.mark.asyncio
    async def test_empty_stream_clears_busy_flag(self, tmp_path):
        """agent_busy must be False after stream ends, even on empty stream."""
        app = _fake_app(tmp_path)
        app.agent_busy["codex"] = True

        await CollabApp._consume_stream(
            app,
            pane="codex",
            speaker="Codex",
            stream=_stream(),
            final=True,
            next_speaker="master",
        )

        assert app.agent_busy["codex"] is False

    @pytest.mark.asyncio
    async def test_failed_event_sets_stuck_flag(self, tmp_path):
        """`failed` stream event must set state.stuck = True."""
        app = _fake_app(tmp_path)

        await CollabApp._consume_stream(
            app,
            pane="codex",
            speaker="Codex",
            stream=_stream(
                StreamEvent(agent="codex", kind="failed", text="CLI exited 1", exit_code=1)
            ),
            final=True,
            next_speaker="master",
        )

        assert app.state.stuck is True

    @pytest.mark.asyncio
    async def test_failed_event_captures_error_text(self, tmp_path):
        """Error text from `failed` event becomes the completed reply."""
        app = _fake_app(tmp_path)
        error_text = "codex: command not found"

        await CollabApp._consume_stream(
            app,
            pane="codex",
            speaker="Codex",
            stream=_stream(
                StreamEvent(agent="codex", kind="failed", text=error_text, exit_code=127)
            ),
            final=True,
            next_speaker="master",
        )

        assert app.latest_completed_reply["codex"] == error_text

    @pytest.mark.asyncio
    async def test_text_delta_accumulates_into_reply(self, tmp_path):
        """Multiple text_delta events are concatenated into the final reply."""
        app = _fake_app(tmp_path)

        await CollabApp._consume_stream(
            app,
            pane="codex",
            speaker="Codex",
            stream=_stream(
                StreamEvent(agent="codex", kind="text_delta", text="Part 1. "),
                StreamEvent(agent="codex", kind="text_delta", text="Part 2."),
                StreamEvent(agent="codex", kind="message_completed", text=""),
                StreamEvent(agent="codex", kind="completed", exit_code=0),
            ),
            final=True,
            next_speaker="master",
        )

        assert app.latest_completed_reply["codex"] == "Part 1. Part 2."

    @pytest.mark.asyncio
    async def test_tool_events_do_not_pollute_relay_text(self, tmp_path):
        """tool_event entries appear in pane_history but must NOT enter latest_completed_reply."""
        app = _fake_app(tmp_path)

        await CollabApp._consume_stream(
            app,
            pane="codex",
            speaker="Codex",
            stream=_stream(
                StreamEvent(agent="codex", kind="tool_event", text="⚙ edit(auth.py)"),
                StreamEvent(agent="codex", kind="text_delta", text="Done."),
                StreamEvent(agent="codex", kind="message_completed", text=""),
            ),
            final=True,
            next_speaker="master",
        )

        assert app.latest_completed_reply["codex"] == "Done."
        tool_entries = [e for e in app.pane_history["codex"] if e[0] == "tool"]
        assert len(tool_entries) == 1, "tool_event must appear in pane_history"

    @pytest.mark.asyncio
    async def test_empty_reply_recorded_in_pane_history(self, tmp_path):
        """Fallback message is appended to pane_history with the correct speaker."""
        app = _fake_app(tmp_path)

        await CollabApp._consume_stream(
            app,
            pane="codex",
            speaker="Codex",
            stream=_stream(),
            final=True,
            next_speaker="master",
        )

        assert ("Codex", "Codex returned no text.") in app.pane_history["codex"]


# ── Helpers for _write_pane tests ─────────────────────────────────────────────

def _fake_app_for_write_pane(tmp_path: Path) -> tuple[types.SimpleNamespace, MagicMock]:
    """Fake app that runs the real _write_pane with a mock RichLog widget.

    Returns (app, mock_widget) so tests can inspect widget.write() calls.
    """
    mock_widget = MagicMock()
    mock_widget.write = MagicMock()
    mock_widget.clear = MagicMock()
    mock_widget.scroll_end = MagicMock()

    app = types.SimpleNamespace()
    app.state = _state()
    app.pane_history = {"claude": [], "codex": []}
    # Indexed list of (path, diff_text) — each occurrence has a stable index.
    app.turn_diff_entries = {"claude": [], "codex": []}
    app.agent_busy = {"claude": False, "codex": False}
    app.active_text = {"claude": "", "codex": ""}
    app.active_message = {"claude": "", "codex": ""}
    app.active_speaker = {"claude": "Claude", "codex": "Codex"}
    app._message_store = {}
    app._active_palette = THEMES["akritrim"]
    app.query_one = lambda selector, widget_type=None: mock_widget
    # Bind all rendering helpers that _write_pane calls indirectly
    app._system_message_style = lambda m: CollabApp._system_message_style(app, m)
    app._speaker_label = lambda s: CollabApp._speaker_label(app, s)
    app._render_message_text = lambda s, m: CollabApp._render_message_text(app, s, m)
    app._copy_link = lambda mid: CollabApp._copy_link(app, mid)
    app._render_active_message = lambda p: None  # no active stream in these tests
    return app, mock_widget


def _written_texts(mock_widget: MagicMock) -> list[str]:
    """Collect plain string representations of all Text objects written to the widget."""
    results = []
    for call in mock_widget.write.call_args_list:
        arg = call.args[0] if call.args else None
        if arg is not None:
            results.append(str(arg))
    return results


# ── Scenario 3: Diff link repaint ─────────────────────────────────────────────

class TestDiffLinkRepaint:
    """
    Diff links must survive any full repaint of the pane widget AND each entry
    must always open the exact patch from its own turn.

    turn_diff_entries is a stable indexed list: index 0 is the first diff ever
    recorded, index 1 the second, etc.  pane_history stores ("diff", str(idx))
    so each link carries a permanent pointer that is unaffected by later edits
    to the same file.
    """

    @pytest.mark.asyncio
    async def test_update_diff_strip_stores_entry_and_calls_write_pane(self, tmp_path):
        """`_update_pane_diff_strip` must append to turn_diff_entries, add idx to pane_history, call _write_pane."""
        app = _fake_app(tmp_path)
        app.turn_diff_entries = {"claude": [], "codex": []}

        await CollabApp._update_pane_diff_strip(
            app, "codex", {"src/auth.py": "diff content"}
        )

        assert app.turn_diff_entries["codex"][0][:2] == ("src/auth.py", "diff content")
        assert ("diff", "0") in app.pane_history["codex"]
        app._write_pane.assert_called_with("codex")

    @pytest.mark.asyncio
    async def test_update_diff_strip_accumulates_across_turns(self, tmp_path):
        """Successive calls accumulate entries; earlier entries are not lost."""
        app = _fake_app(tmp_path)
        app.turn_diff_entries = {"claude": [], "codex": []}

        await CollabApp._update_pane_diff_strip(app, "codex", {"file_a.py": "diff1"})
        await CollabApp._update_pane_diff_strip(app, "codex", {"file_b.py": "diff2"})

        paths = [p for p, *_ in app.turn_diff_entries["codex"]]
        assert "file_a.py" in paths
        assert "file_b.py" in paths
        indices = [idx for spk, idx in app.pane_history["codex"] if spk == "diff"]
        assert "0" in indices
        assert "1" in indices

    @pytest.mark.asyncio
    async def test_same_file_edited_twice_has_distinct_indices(self, tmp_path):
        """Editing the same file in two turns creates two entries with distinct stable indices."""
        app = _fake_app(tmp_path)
        app.turn_diff_entries = {"claude": [], "codex": []}

        await CollabApp._update_pane_diff_strip(app, "codex", {"app.py": "first diff"})
        await CollabApp._update_pane_diff_strip(app, "codex", {"app.py": "second diff"})

        assert len(app.turn_diff_entries["codex"]) == 2
        assert app.turn_diff_entries["codex"][0][:2] == ("app.py", "first diff")
        assert app.turn_diff_entries["codex"][1][:2] == ("app.py", "second diff")
        diff_indices = [idx for spk, idx in app.pane_history["codex"] if spk == "diff"]
        assert len(diff_indices) == 2
        assert diff_indices[0] != diff_indices[1], "each occurrence must have a distinct index"

    def test_write_pane_emits_diff_links_from_pane_history(self, tmp_path):
        """`_write_pane` resolves [DIFF] path via index and writes a clickable link."""
        app, mock_widget = _fake_app_for_write_pane(tmp_path)
        app.turn_diff_entries["codex"].append(("src/auth.py", "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new", 1))
        app.pane_history["codex"].append(("diff", "0"))

        CollabApp._write_pane(app, "codex")

        written = _written_texts(mock_widget)
        assert any("[DIFF]" in t and "src/auth.py" in t for t in written), (
            f"Expected [DIFF] src/auth.py in written output, got: {written}"
        )

    def test_write_pane_emits_all_accumulated_diff_links(self, tmp_path):
        """All accumulated paths must be rendered, not just the most recent."""
        app, mock_widget = _fake_app_for_write_pane(tmp_path)
        app.turn_diff_entries["codex"] = [
            ("src/auth.py", "diff1", 1),
            ("src/models.py", "diff2", 2),
        ]
        app.pane_history["codex"].extend([
            ("diff", "0"),
            ("diff", "1"),
        ])

        CollabApp._write_pane(app, "codex")

        written = _written_texts(mock_widget)
        assert any("auth.py" in t for t in written), "auth.py link missing"
        assert any("models.py" in t for t in written), "models.py link missing"

    def test_write_pane_clears_then_repaints_so_no_duplication(self, tmp_path):
        """Calling _write_pane twice must not double the diff links."""
        app, mock_widget = _fake_app_for_write_pane(tmp_path)
        app.turn_diff_entries["codex"].append(("src/auth.py", "diff", 1))
        app.pane_history["codex"].append(("diff", "0"))

        CollabApp._write_pane(app, "codex")
        mock_widget.write.reset_mock()
        mock_widget.clear.reset_mock()
        CollabApp._write_pane(app, "codex")

        written = _written_texts(mock_widget)
        diff_writes = [t for t in written if "[DIFF]" in t and "auth.py" in t]
        assert len(diff_writes) == 1, (
            f"Expected exactly 1 diff link on second repaint, got {len(diff_writes)}: {diff_writes}"
        )


# ── Scenario 4: Claude adapter result payload fallback (D-049) ─────────────────

def _make_mock_process(stdout_bytes: bytes, stderr_bytes: bytes = b"", returncode: int = 0):
    """Build a minimal mock subprocess for ClaudeAdapter.stream_send."""
    mock_stdout = MagicMock()
    mock_stdout.read = AsyncMock(side_effect=[stdout_bytes, b""])
    mock_stderr = MagicMock()
    mock_stderr.read = AsyncMock(return_value=stderr_bytes)
    proc = MagicMock()
    proc.stdout = mock_stdout
    proc.stderr = mock_stderr
    proc.returncode = None
    proc.wait = AsyncMock(return_value=returncode)
    return proc


class TestClaudeAdapterResultPayload:
    """D-049: result-payload fallback in ClaudeAdapter.stream_send."""

    @pytest.mark.asyncio
    async def test_result_payload_yields_message_completed(self, tmp_path):
        """A JSON line with type='result' must yield a message_completed event."""
        adapter = ClaudeAdapter(cwd=tmp_path)
        line = json.dumps({"type": "result", "result": "Hello from result payload."}) + "\n"
        proc = _make_mock_process(line.encode())

        with patch("akritrim_colab.agents.claude_adapter.asyncio.create_subprocess_exec",
                   new_callable=AsyncMock, return_value=proc):
            events = [e async for e in adapter.stream_send("test")]

        completed = [e for e in events if e.kind == "message_completed"]
        assert completed, f"No message_completed event; got {events}"
        assert completed[0].text == "Hello from result payload."

    @pytest.mark.asyncio
    async def test_result_payload_fixes_no_text_fallback(self, tmp_path):
        """_consume_stream records result text instead of 'Claude returned no text.'"""
        app = _fake_app(tmp_path)

        await CollabApp._consume_stream(
            app,
            pane="claude",
            speaker="Claude",
            stream=_stream(
                StreamEvent(agent="claude", kind="message_completed", text="Result text from CLI."),
                StreamEvent(agent="claude", kind="completed", exit_code=0),
            ),
            final=True,
            next_speaker="master",
        )

        assert app.latest_completed_reply["claude"] == "Result text from CLI."

    @pytest.mark.asyncio
    async def test_text_delta_takes_precedence_over_result_payload(self, tmp_path):
        """When text_delta fires first, message_completed (from result) does not overwrite it."""
        app = _fake_app(tmp_path)

        await CollabApp._consume_stream(
            app,
            pane="claude",
            speaker="Claude",
            stream=_stream(
                StreamEvent(agent="claude", kind="text_delta", text="Streamed reply."),
                StreamEvent(agent="claude", kind="message_completed", text="Result payload text."),
                StreamEvent(agent="claude", kind="completed", exit_code=0),
            ),
            final=True,
            next_speaker="master",
        )

        assert app.latest_completed_reply["claude"] == "Streamed reply."

    @pytest.mark.asyncio
    async def test_result_payload_session_id_extracted(self, tmp_path):
        """session_id in the result payload is emitted as session_started when none seen earlier."""
        adapter = ClaudeAdapter(cwd=tmp_path)
        line = json.dumps({
            "type": "result",
            "result": "text",
            "session_id": "sess-xyz-123",
        }) + "\n"
        proc = _make_mock_process(line.encode())

        with patch("akritrim_colab.agents.claude_adapter.asyncio.create_subprocess_exec",
                   new_callable=AsyncMock, return_value=proc):
            events = [e async for e in adapter.stream_send("test")]

        session_events = [e for e in events if e.kind == "session_started"]
        assert session_events, f"No session_started event; got {events}"
        assert session_events[0].session_handle == "sess-xyz-123"

    @pytest.mark.asyncio
    async def test_empty_result_payload_does_not_yield_message_completed(self, tmp_path):
        """result payload with empty string must not emit message_completed (avoids blank reply)."""
        adapter = ClaudeAdapter(cwd=tmp_path)
        line = json.dumps({"type": "result", "result": ""}) + "\n"
        proc = _make_mock_process(line.encode())

        with patch("akritrim_colab.agents.claude_adapter.asyncio.create_subprocess_exec",
                   new_callable=AsyncMock, return_value=proc):
            events = [e async for e in adapter.stream_send("test")]

        completed = [e for e in events if e.kind == "message_completed"]
        assert not completed, f"Empty result must not produce message_completed; got {completed}"


# ── Scenario 5: @file mention expansion security and correctness (D-053) ──────

class TestFileMentionExpansion:
    """_expand_file_mentions: path containment, size limit, binary guard, routing names."""

    def test_normal_file_is_injected(self, tmp_path):
        """A file inside project_root is read and injected with fenced markers."""
        (tmp_path / "notes.txt").write_text("hello world", encoding="utf-8")
        expanded, injected = _expand_file_mentions("see @notes.txt for details", tmp_path)
        assert "notes.txt" in injected
        assert "hello world" in expanded
        assert "--- @notes.txt ---" in expanded

    def test_path_traversal_is_blocked(self, tmp_path):
        """@../../etc/passwd must not escape the repository root."""
        expanded, injected = _expand_file_mentions("@../../etc/passwd", tmp_path)
        assert not injected
        assert "path escapes repository root" in expanded

    def test_double_dot_in_subdir_blocked(self, tmp_path):
        """Even when combined with a subdir, traversal is blocked."""
        (tmp_path / "src").mkdir()
        expanded, injected = _expand_file_mentions("@src/../../etc/passwd", tmp_path)
        assert not injected
        assert "path escapes repository root" in expanded

    def test_file_too_large_is_skipped(self, tmp_path):
        """Files over 50 KB produce an inline error note, not content."""
        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * (51 * 1024))
        expanded, injected = _expand_file_mentions("@big.bin", tmp_path)
        assert not injected
        assert "file too large" in expanded

    def test_binary_file_is_skipped(self, tmp_path):
        """Files containing null bytes are treated as binary and skipped."""
        (tmp_path / "binary.dat").write_bytes(b"data\x00more")
        expanded, injected = _expand_file_mentions("@binary.dat", tmp_path)
        assert not injected
        assert "binary file" in expanded

    def test_routing_names_are_not_expanded(self, tmp_path):
        """@claude, @codex, @both, @master are never treated as file paths."""
        for name in ("claude", "codex", "both", "master"):
            # Even if a file with that name exists, routing names win
            (tmp_path / name).write_text("secret", encoding="utf-8")
            expanded, injected = _expand_file_mentions(f"@{name}", tmp_path)
            assert not injected, f"@{name} should not be injected"
            assert "secret" not in expanded

    def test_missing_file_token_preserved(self, tmp_path):
        """An @token with no matching file is left verbatim in the message."""
        expanded, injected = _expand_file_mentions("see @does_not_exist.py", tmp_path)
        assert not injected
        assert "@does_not_exist.py" in expanded

    def test_multiple_mentions_expanded(self, tmp_path):
        """Multiple @file tokens in one message are all expanded."""
        (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
        (tmp_path / "b.txt").write_text("BBB", encoding="utf-8")
        expanded, injected = _expand_file_mentions("@a.txt and @b.txt", tmp_path)
        assert sorted(injected) == ["a.txt", "b.txt"]
        assert "AAA" in expanded
        assert "BBB" in expanded

    def test_subdirectory_file_is_injected(self, tmp_path):
        """@src/foo.py resolves correctly inside project_root."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "foo.py").write_text("def foo(): pass", encoding="utf-8")
        expanded, injected = _expand_file_mentions("@src/foo.py", tmp_path)
        assert "src/foo.py" in injected
        assert "def foo(): pass" in expanded

    # ── Truncation (B) ────────────────────────────────────────────────────────

    def test_long_file_is_truncated_not_skipped(self, tmp_path):
        """Files over _MAX_FILE_INJECT_LINES are truncated; early lines appear."""
        from akritrim_colab.app import _MAX_FILE_INJECT_LINES
        lines = [f"line {i}\n" for i in range(1, _MAX_FILE_INJECT_LINES + 50)]
        (tmp_path / "big.py").write_text("".join(lines), encoding="utf-8")
        expanded, injected = _expand_file_mentions("@big.py", tmp_path)
        assert "big.py" in injected
        assert "line 1" in expanded
        assert f"line {_MAX_FILE_INJECT_LINES + 1}" not in expanded

    def test_truncation_notice_included(self, tmp_path):
        """The truncation notice states how many lines were omitted."""
        from akritrim_colab.app import _MAX_FILE_INJECT_LINES
        total = _MAX_FILE_INJECT_LINES + 10
        lines = [f"line {i}\n" for i in range(1, total + 1)]
        (tmp_path / "big.py").write_text("".join(lines), encoding="utf-8")
        expanded, _ = _expand_file_mentions("@big.py", tmp_path)
        assert "lines omitted" in expanded
        assert "START-END" in expanded  # range hint present

    def test_short_file_not_truncated(self, tmp_path):
        """Files within the line limit are injected in full without a notice."""
        (tmp_path / "small.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        expanded, _ = _expand_file_mentions("@small.py", tmp_path)
        assert "lines omitted" not in expanded
        assert "x = 1" in expanded

    # ── Line-range syntax (C) ─────────────────────────────────────────────────

    def test_line_range_injects_correct_lines(self, tmp_path):
        """@file.py:2-4 injects lines 2, 3, 4 only."""
        content = "line1\nline2\nline3\nline4\nline5\n"
        (tmp_path / "f.py").write_text(content, encoding="utf-8")
        expanded, injected = _expand_file_mentions("@f.py:2-4", tmp_path)
        assert "f.py:2-4" in injected
        assert "line2" in expanded
        assert "line4" in expanded
        assert "line1" not in expanded
        assert "line5" not in expanded

    def test_line_range_label_in_header(self, tmp_path):
        """The fenced header includes the range label."""
        content = "\n".join(f"L{i}" for i in range(1, 11)) + "\n"
        (tmp_path / "f.py").write_text(content, encoding="utf-8")
        expanded, _ = _expand_file_mentions("@f.py:3-5", tmp_path)
        assert "lines 3" in expanded

    def test_line_range_clamped_to_file_length(self, tmp_path):
        """End of range beyond file length is clamped, not an error."""
        (tmp_path / "f.py").write_text("a\nb\nc\n", encoding="utf-8")
        expanded, injected = _expand_file_mentions("@f.py:2-999", tmp_path)
        assert "f.py:2-999" in injected
        assert "b" in expanded
        assert "c" in expanded

    def test_invalid_range_returns_error_note(self, tmp_path):
        """start > end produces an inline error, not content."""
        (tmp_path / "f.py").write_text("a\nb\nc\n", encoding="utf-8")
        expanded, injected = _expand_file_mentions("@f.py:5-2", tmp_path)
        assert not injected
        assert "invalid range" in expanded

    def test_range_zero_start_returns_error_note(self, tmp_path):
        """start=0 (line 0 does not exist) produces an inline error."""
        (tmp_path / "f.py").write_text("a\nb\nc\n", encoding="utf-8")
        expanded, injected = _expand_file_mentions("@f.py:0-2", tmp_path)
        assert not injected
        assert "invalid range" in expanded


class TestFileCompletions:
    """_file_completions: path containment, basic results, hidden-file filtering."""

    def test_completions_returned_for_files_in_repo(self, tmp_path):
        """Files inside project_root appear in completions."""
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")
        results = _file_completions("main", tmp_path)
        tokens = [token for token, _desc in results]
        assert "@main.py" in tokens

    def test_traversal_with_dotdot_returns_empty(self, tmp_path):
        """@../  must not enumerate directories outside the repo root."""
        results = _file_completions("../", tmp_path)
        assert results == []

    def test_traversal_via_subdir_returns_empty(self, tmp_path):
        """Even a valid subdir combined with ../ cannot escape the root."""
        (tmp_path / "src").mkdir()
        results = _file_completions("src/../../", tmp_path)
        assert results == []

    def test_directory_completions_have_trailing_slash(self, tmp_path):
        """Subdirectory completions end with '/' to allow drilling."""
        (tmp_path / "pkg").mkdir()
        results = _file_completions("pkg", tmp_path)
        tokens = [token for token, _desc in results]
        assert "@pkg/" in tokens

    def test_hidden_items_excluded_by_default(self, tmp_path):
        """Items starting with '.' are hidden unless the partial starts with '.'."""
        (tmp_path / ".hidden.txt").write_text("x", encoding="utf-8")
        results = _file_completions("", tmp_path)
        tokens = [token for token, _desc in results]
        assert not any(".hidden" in t for t in tokens)

    def test_hidden_items_included_when_partial_starts_with_dot(self, tmp_path):
        """Typing '.' explicitly should reveal hidden items."""
        (tmp_path / ".env").write_text("SECRET=x", encoding="utf-8")
        results = _file_completions(".", tmp_path)
        tokens = [token for token, _desc in results]
        assert "@.env" in tokens


class TestDangerousOpDetection:
    """_check_dangerous_op and _consume_stream dangerous-op gate."""

    def _app_with_ops(self, tmp_path: Path) -> types.SimpleNamespace:
        """_fake_app variant with DANGEROUS_OPS_DEFAULT wired in."""
        from akritrim_colab.app import DANGEROUS_OPS_DEFAULT
        app = _fake_app(tmp_path)
        app.dangerous_ops = list(DANGEROUS_OPS_DEFAULT)
        return app

    # ── _check_dangerous_op unit tests ──────────────────────────────────────

    def test_git_push_detected(self, tmp_path):
        """'$ git push origin main' matches the git push pattern."""
        app = self._app_with_ops(tmp_path)
        result = CollabApp._check_dangerous_op(app, "$ git push origin main")
        assert result is not None
        op_name, event_text = result
        assert "git push" in op_name.lower()

    def test_npm_publish_detected(self, tmp_path):
        """'$ npm publish' is caught."""
        app = self._app_with_ops(tmp_path)
        result = CollabApp._check_dangerous_op(app, "$ npm publish")
        assert result is not None
        assert "npm" in result[0].lower()

    def test_rm_rf_detected(self, tmp_path):
        """'$ rm -rf ./dist' is caught."""
        app = self._app_with_ops(tmp_path)
        result = CollabApp._check_dangerous_op(app, "$ rm -rf ./dist")
        assert result is not None

    def test_safe_command_not_detected(self, tmp_path):
        """'$ git status' is not a dangerous op."""
        app = self._app_with_ops(tmp_path)
        assert CollabApp._check_dangerous_op(app, "$ git status") is None

    def test_session_approved_op_passes(self, tmp_path):
        """Exact command text in _active_approvals suppresses the gate for that command."""
        app = self._app_with_ops(tmp_path)
        # Approval key is the specific command text, not the op-class name.
        app._active_approvals.add("$ git push origin main")
        assert CollabApp._check_dangerous_op(app, "$ git push origin main") is None

    def test_different_git_push_still_prompts(self, tmp_path):
        """Approving 'git push origin main' does NOT approve 'git push --force'."""
        app = self._app_with_ops(tmp_path)
        app._active_approvals.add("$ git push origin main")
        # Different invocation — must still be detected.
        result = CollabApp._check_dangerous_op(app, "$ git push --force origin main")
        assert result is not None

    def test_seen_post_exec_suppresses_repeat_detection(self, tmp_path):
        """Command in _seen_post_exec is not re-detected (avoids notification spam)."""
        app = self._app_with_ops(tmp_path)
        app._seen_post_exec.add("$ git push origin main")
        assert CollabApp._check_dangerous_op(app, "$ git push origin main") is None

    def test_seen_post_exec_does_not_imply_approval(self, tmp_path):
        """A command suppressed by _seen_post_exec is NOT in _active_approvals."""
        app = self._app_with_ops(tmp_path)
        app._seen_post_exec.add("$ git push origin main")
        assert "$ git push origin main" not in app._active_approvals

    @pytest.mark.asyncio
    async def test_post_exec_path_writes_seen_set_not_approvals(self, tmp_path):
        """_handle_dangerous_op_abort with pre_execution=False adds to _seen_post_exec,
        not _active_approvals."""
        app = self._app_with_ops(tmp_path)

        await CollabApp._handle_dangerous_op_abort(
            app,
            pane="codex",
            speaker="Codex",
            op_name="git push",
            event_text="$ git push origin main",
            final=True,
            next_speaker="master",
            pre_execution=False,
        )

        assert "$ git push origin main" in app._seen_post_exec
        assert "$ git push origin main" not in app._active_approvals

    def test_empty_dangerous_ops_list_disables_gate(self, tmp_path):
        """dangerous_ops = [] disables detection entirely."""
        app = _fake_app(tmp_path)  # dangerous_ops=[] by default in _fake_app
        assert CollabApp._check_dangerous_op(app, "$ git push origin main") is None

    def test_function_call_format_matched(self, tmp_path):
        """⚙ shell({'command': 'git push'}) matches via the same regex."""
        app = self._app_with_ops(tmp_path)
        result = CollabApp._check_dangerous_op(app, "⚙ shell({\"command\": \"git push origin main\"})")
        assert result is not None

    # ── DANGEROUS_OP_PATTERNS coverage ──────────────────────────────────────

    def test_all_default_patterns_match_canonical_command(self, tmp_path):
        """Every pattern in DANGEROUS_OP_PATTERNS matches at least one canonical command."""
        canonical = [
            ("git push", "$ git push origin main"),
            ("git tag", "$ git tag v1.0.0"),
            ("npm publish/pack", "$ npm publish"),
            ("twine upload", "$ twine upload dist/*"),
            ("pip install --user/--break-system", "$ pip install --user requests"),
            ("rm -rf", "$ rm -rf ./dist"),
            ("docker push", "$ docker push myrepo/myimage:latest"),
            ("kubectl apply/delete", "$ kubectl apply -f deployment.yaml"),
        ]
        for op_name, cmd in canonical:
            matched = any(pattern.search(cmd) for name, pattern in DANGEROUS_OP_PATTERNS if name == op_name)
            assert matched, f"Pattern for '{op_name}' did not match '{cmd}'"

    # ── _consume_stream integration ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_dangerous_tool_event_triggers_abort_handler(self, tmp_path):
        """When a dangerous tool_event fires, _handle_dangerous_op_abort is called."""
        app = self._app_with_ops(tmp_path)

        await CollabApp._consume_stream(
            app,
            pane="codex",
            speaker="Codex",
            stream=_stream(
                StreamEvent(agent="codex", kind="tool_event", text="$ git push origin main"),
            ),
            final=True,
            next_speaker="master",
        )

        app._handle_dangerous_op_abort.assert_awaited_once()
        call_kwargs = app._handle_dangerous_op_abort.await_args
        assert call_kwargs.args[2] == "git push"          # op_name
        # Codex pane → post-execution, so pre_execution must be False.
        assert call_kwargs.kwargs.get("pre_execution") is False

    @pytest.mark.asyncio
    async def test_claude_pane_passes_pre_execution_true(self, tmp_path):
        """Claude pane passes pre_execution=True to the abort handler."""
        app = self._app_with_ops(tmp_path)

        await CollabApp._consume_stream(
            app,
            pane="claude",
            speaker="Claude",
            stream=_stream(
                StreamEvent(agent="claude", kind="tool_event", text="$ git push origin main"),
            ),
            final=True,
            next_speaker="master",
        )

        call_kwargs = app._handle_dangerous_op_abort.await_args
        assert call_kwargs.kwargs.get("pre_execution") is True

    @pytest.mark.asyncio
    async def test_approved_tool_event_does_not_abort(self, tmp_path):
        """Pre-approved ops (exact command text) pass through without calling the abort handler."""
        app = self._app_with_ops(tmp_path)
        # Approval key is event_text.strip(), not op class name.
        app._active_approvals.add("$ git push origin main")

        await CollabApp._consume_stream(
            app,
            pane="codex",
            speaker="Codex",
            stream=_stream(
                StreamEvent(agent="codex", kind="tool_event", text="$ git push origin main"),
                StreamEvent(agent="codex", kind="message_completed", text="pushed ok"),
            ),
            final=True,
            next_speaker="master",
        )

        app._handle_dangerous_op_abort.assert_not_awaited()
        assert app.latest_completed_reply["codex"] == "pushed ok"


# ── Scenario: Two-phase WAITING_FOR_MASTER relay (D-066) ─────────────────────

class TestWaitingForMasterRelay:
    """_run_collaboration_loop D-066: first WFM relays; second WFM stops."""

    def _relay_app(self, tmp_path: Path) -> types.SimpleNamespace:
        """Fake app wired for relay-loop unit tests."""
        app = _fake_app(tmp_path)
        app.state.collaboration_active = True
        app.state.stuck = False
        app.state.auto_turn_count = 1   # >0 so first_relay=False → WFM is checked
        app.state.last_speaker = "claude"
        app.max_auto_turns = 12
        app.session_approvals_granted = 0
        app.session_approvals_denied = 0
        # Real pause-check so WAITING_FOR_MASTER in latest_completed_reply triggers
        app._should_pause_for_master = (
            lambda skip_waiting_sentinel=False:
            CollabApp._should_pause_for_master(app, skip_waiting_sentinel)
        )
        app._extract_master_request = MagicMock(return_value=None)
        app._build_collaboration_prompt = MagicMock(return_value="relay-prompt")
        app._invoke_claude = AsyncMock()
        app._invoke_codex = AsyncMock()
        return app

    @pytest.mark.asyncio
    async def test_first_wfm_relays_to_other_agent(self, tmp_path):
        """First WAITING_FOR_MASTER from Claude sends one confirmation relay to Codex."""
        app = self._relay_app(tmp_path)
        app.latest_completed_reply["claude"] = "WAITING_FOR_MASTER"
        app.latest_completed_reply["codex"] = "Some prior reply."

        # After Codex receives the confirmation prompt, stop the loop.
        async def stop(*_a, **_kw):
            app.state.collaboration_active = False
        app._invoke_codex.side_effect = stop

        await CollabApp._run_collaboration_loop(app)

        # Codex (other agent) received the confirmation prompt.
        app._invoke_codex.assert_awaited_once()
        # Claude was NOT invoked (confirmation went to Codex only).
        app._invoke_claude.assert_not_awaited()
        # Claude's WFM reply was cleared so the next loop check doesn't re-trigger.
        assert app.latest_completed_reply["claude"] == ""
        # Chat was not yet told "agents completed" — that happens on the second stop.
        app._write_chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_second_wfm_stops_session(self, tmp_path):
        """Second WAITING_FOR_MASTER (after confirmation relay) pauses the session."""
        app = self._relay_app(tmp_path)
        # Simulate state after confirmation relay: Claude's WFM was cleared, pending set.
        app.latest_completed_reply["claude"] = ""
        app.latest_completed_reply["codex"] = "WAITING_FOR_MASTER"
        app._pending_wfm_pane = "claude"   # set during first WFM relay
        app.state.auto_turn_count = 2

        await CollabApp._run_collaboration_loop(app)

        # Neither agent is invoked — loop stopped immediately.
        app._invoke_codex.assert_not_awaited()
        app._invoke_claude.assert_not_awaited()
        # Session paused correctly.
        assert app.state.collaboration_active is False
        assert app.state.waiting_for_master is True
        assert app.state.next_speaker == "master"
        # Flag cleared on stop.
        assert app._pending_wfm_pane is None
        # Chat notification fired.
        app._write_chat.assert_called()

    @pytest.mark.asyncio
    async def test_normal_relay_clears_pending_wfm_pane(self, tmp_path):
        """A normal relay turn resets _pending_wfm_pane so future WFMs get a fresh relay."""
        app = self._relay_app(tmp_path)
        # _pending_wfm_pane is set from a prior WFM, but this turn has no WFM.
        app._pending_wfm_pane = "claude"
        app.latest_completed_reply["claude"] = "Here is my implementation."
        app.latest_completed_reply["codex"] = "Prior review text."

        async def stop(*_a, **_kw):
            app.state.collaboration_active = False
        app._invoke_codex.side_effect = stop

        await CollabApp._run_collaboration_loop(app)

        # Normal relay fired (Codex is next after Claude).
        app._invoke_codex.assert_awaited_once()
        # Flag was cleared during the normal relay turn.
        assert app._pending_wfm_pane is None
