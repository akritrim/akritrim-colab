from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass
class AppSettings:
    identity: str | None = None
    role: str | None = None
    codex_role: str | None = None
    claude_role: str | None = None
    max_auto_turns: int | None = None
    debug_log: str | None = None
    claude_model: str | None = None
    codex_model: str | None = None
    theme: str | None = None
    require_git: bool = True
    dangerous_ops: list[str] | None = None  # None = use DANGEROUS_OPS_DEFAULT
    claude_dangerously_skip_permissions: bool = True

    @classmethod
    def load(cls, path: Path) -> "AppSettings":
        if not path.exists():
            return cls()
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        app = payload.get("app", {})
        if not isinstance(app, dict):
            return cls()
        max_auto_turns = app.get("max_auto_turns", app.get("auto_turn_threshold"))
        if not isinstance(max_auto_turns, int):
            max_auto_turns = None
        require_git = app.get("require_git", True)
        if not isinstance(require_git, bool):
            require_git = True
        claude_dangerously_skip_permissions = app.get("claude_dangerously_skip_permissions", True)
        if not isinstance(claude_dangerously_skip_permissions, bool):
            claude_dangerously_skip_permissions = True
        raw_ops = app.get("dangerous_ops")
        dangerous_ops: list[str] | None = None
        if isinstance(raw_ops, list):
            dangerous_ops = [s for s in raw_ops if isinstance(s, str) and s.strip()]

        def _str(key: str) -> str | None:
            v = app.get(key)
            return v if isinstance(v, str) and v.strip() else None

        return cls(
            identity=_str("identity"),
            role=_str("role"),
            codex_role=_str("codex_role"),
            claude_role=_str("claude_role"),
            max_auto_turns=max_auto_turns,
            debug_log=_str("debug_log"),
            claude_model=_str("claude_model"),
            codex_model=_str("codex_model"),
            theme=_str("theme"),
            require_git=require_git,
            dangerous_ops=dangerous_ops,
            claude_dangerously_skip_permissions=claude_dangerously_skip_permissions,
        )

    @staticmethod
    def save_theme(path: Path, theme: str) -> None:
        """Write or update the theme key in settings.toml under [app]."""
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"[app]\ntheme = {theme!r}\n", encoding="utf-8")
            return
        text = path.read_text(encoding="utf-8")
        # If a theme line already exists anywhere in [app], replace it in-place.
        if re.search(r"^\s*theme\s*=", text, re.MULTILINE):
            text = re.sub(
                r"^\s*theme\s*=.*$",
                f"theme = {theme!r}",
                text,
                flags=re.MULTILINE,
            )
        elif "[app]" in text:
            # Insert right after the [app] header line.
            text = re.sub(
                r"(\[app\]\n)",
                rf"\g<1>theme = {theme!r}\n",
                text,
                count=1,
            )
        else:
            text += f"\n[app]\ntheme = {theme!r}\n"
        path.write_text(text, encoding="utf-8")


@dataclass
class ProjectConfig:
    project_root: Path
    memory_path: Path
    scratchpad_path: Path
    rules_path: Path
    session_dir: Path
    settings_path: Path

    @classmethod
    def detect(cls, root: Path, collab_dir: str = ".collab") -> "ProjectConfig":
        """Prefer legacy docs/collab if present, otherwise default to .collab."""
        legacy = root / "docs" / "collab"
        if legacy.exists():
            return cls.from_dir(root, legacy)
        return cls.from_dir(root, root / collab_dir)

    @classmethod
    def from_dir(cls, root: Path, base: Path) -> "ProjectConfig":
        return cls(
            project_root=root,
            memory_path=base / "memory.md",
            scratchpad_path=base / "scratchpad.md",
            rules_path=base / "rules.md",
            session_dir=base / "session",
            settings_path=base / "settings.toml",
        )
