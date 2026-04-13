from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import ProjectConfig


def event_log_path(config: ProjectConfig) -> Path:
    return config.session_dir / "events.jsonl"


def append_event(config: ProjectConfig, event: dict[str, object]) -> None:
    config.session_dir.mkdir(parents=True, exist_ok=True)
    path = event_log_path(config)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def build_file_diff(repo_root: Path, rel_path: str) -> str:
    file_path = repo_root / rel_path
    if not file_path.exists():
        return ""
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel_path],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode == 0:
        result = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", "HEAD", "--", rel_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            return result.stdout.strip()
        fallback = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", "--", rel_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return fallback.stdout.strip()

    result = subprocess.run(
        ["git", "diff", "--no-index", "--", "/dev/null", str(file_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()
