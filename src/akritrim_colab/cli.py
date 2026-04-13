from __future__ import annotations

import argparse
from pathlib import Path

from .app import CollabApp, load_or_create_state
from .config import AppSettings, ProjectConfig


INIT_FILES = {
    "memory.md": (
        "# Collaboration Memory\n"
        "> Permanent record of agreed decisions.\n\n"
        "---\n\n"
        "## Archive\n"
    ),
    "scratchpad.md": (
        "# Collaboration Scratchpad\n"
        "> Working space only.\n\n"
        "---\n\n"
        "## Active Topics\n\n"
        "*(none)*\n\n"
        "---\n\n"
        "## Closed Topics\n"
        "*(none)*\n\n"
        "---\n"
    ),
    "rules.md": (
        "# Collaboration Rules\n"
        "> Rules exist to resolve impasses, not suppress debate.\n\n"
        "---\n"
    ),
    "settings.toml": (
        "[app]\n"
        'identity = "codex"\n'
        'role = "collaborator"\n'
        'codex_role = "reviewer"\n'
        'claude_role = "implementer"\n'
        "max_auto_turns = 12\n"
        "# UI theme (akritrim | nord | dracula | solarized)\n"
        'theme = "akritrim"\n'
        "# Model overrides\n"
        'claude_model = "claude-sonnet-4-6"\n'
        'codex_model = "gpt-5.4"\n'
        '# debug_log = ".collab/session/tui_debug.log"\n'
        "# Set to false to allow running outside a git repository.\n"
        "# Codex will use --skip-git-repo-check; git-backed project context will be unavailable.\n"
        "# Turn-local diff display still works. Can also be overridden per-run: --no-require-git\n"
        "require_git = true\n"
        "# Operations that require explicit Master approval before an agent may execute them.\n"
        "# Agents must emit 'Master: requesting permission to [op]' and WAIT for your response.\n"
        "# Comment out or set to [] to disable the permission gate entirely.\n"
        "# dangerous_ops = [\n"
        "#   \"git push\",\n"
        "#   \"npm publish\",\n"
        "#   \"twine upload\",\n"
        "#   \"rm -rf\",\n"
        "# ]\n"
    ),
}


def is_initialized(config: ProjectConfig) -> bool:
    return all(
        path.exists()
        for path in (config.memory_path, config.scratchpad_path, config.rules_path)
    )


def cmd_init(config: ProjectConfig) -> None:
    config.session_dir.mkdir(parents=True, exist_ok=True)
    base = config.session_dir.parent
    for name, content in INIT_FILES.items():
        path = base / name
        if path.exists():
            print(f"skip {path} (exists)")
            continue
        path.write_text(content, encoding="utf-8")
        print(f"created {path}")
    print("Initialized. Run: akritrim-colab")


def cmd_run(config: ProjectConfig, args: argparse.Namespace) -> None:
    if not is_initialized(config):
        print(f"No collaboration scaffold found in {config.session_dir.parent}; initializing first.")
        cmd_init(config)
    settings = AppSettings.load(config.settings_path)
    debug_log_value = getattr(args, "debug_log", None) or settings.debug_log
    debug_log = Path(debug_log_value) if debug_log_value else config.session_dir / "tui_debug.log"
    state = load_or_create_state(
        config=config,
        identity=args.identity or settings.identity or "codex",
        role=args.role or settings.role or "collaborator",
        codex_role=args.codex_role or settings.codex_role or "reviewer",
        claude_role=args.claude_role or settings.claude_role or "implementer",
    )
    max_auto_turns = getattr(args, "max_auto_turns", None) or settings.max_auto_turns
    # CLI --require-git / --no-require-git overrides settings.toml for this run.
    # args.require_git is True/False when the flag was given, None when absent.
    cli_require_git = getattr(args, "require_git", None)
    require_git = settings.require_git if cli_require_git is None else cli_require_git
    app = CollabApp(
        state=state,
        config=config,
        debug_log_path=debug_log,
        max_auto_turns=max_auto_turns,
        claude_model=settings.claude_model or None,
        codex_model=settings.codex_model or None,
        require_git=require_git,
        dangerous_ops=settings.dangerous_ops,  # None → use DANGEROUS_OPS_DEFAULT
    )
    app.run()

def add_repo_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dir", default=".collab")
    parser.add_argument("--legacy", action="store_true", help="Use docs/collab/ layout")


def add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--identity", choices=["claude", "codex"])
    parser.add_argument("--role", default="collaborator")
    parser.add_argument("--codex-role", default="reviewer")
    parser.add_argument("--claude-role", default="implementer")
    parser.add_argument("--debug-log")
    parser.add_argument("--max-auto-turns", type=int, default=None,
                        help="Max auto-turns before pausing for Master (overrides settings.toml)")
    parser.add_argument(
        "--require-git",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="require_git",
        help=(
            "--require-git / --no-require-git: override require_git from settings.toml for this run. "
            "--no-require-git allows running outside a git repository "
            "(Codex uses --skip-git-repo-check; git-backed project context is unavailable). "
            "--require-git forces git requirement even when settings.toml has require_git = false."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="akritrim-colab",
        description="Three-party AI collaboration TUI",
    )
    add_repo_options(parser)
    add_run_options(parser)
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Scaffold collaboration files in this repo")
    add_repo_options(p_init)

    p_run = sub.add_parser("run", help="Launch the collaboration TUI")
    add_repo_options(p_run)
    add_run_options(p_run)

    return parser


def resolve_config(args: argparse.Namespace) -> ProjectConfig:
    root = Path.cwd()
    if getattr(args, "legacy", False):
        return ProjectConfig.from_dir(root, root / "docs" / "collab")
    return ProjectConfig.detect(root, getattr(args, "dir", ".collab"))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = resolve_config(args)
    if args.command == "init":
        cmd_init(config)
        return
    cmd_run(config, args)
