from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class StreamEvent:
    agent: Literal["claude", "codex"]
    kind: Literal[
        "session_started",
        "text_delta",
        "message_completed",
        "tool_event",   # visual-only: tool calls, shell runs, file edits
        "stderr",
        "failed",
        "completed",
    ]
    text: str = ""
    session_handle: str | None = None
    exit_code: int | None = None
