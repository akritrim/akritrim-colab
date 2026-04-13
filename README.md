# akritrim-colab

A three-pane collaboration TUI that puts you between Claude and Codex inside any repository. You direct, they implement and review.

Produly made by Claude code, Codex and me :)
> Requires [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) and [Codex CLI](https://github.com/openai/codex) installed and authenticated.

## Install

```bash
pip install akritrim-colab
```

Or, for an isolated CLI install:

```bash
pipx install akritrim-colab
```

## Quick Start

```bash
cd /path/to/your-repo
akritrim-colab
```

On first run, collaboration files are scaffolded into `.collab/`. Then the TUI launches.

## Layout

| Pane | Contents |
|------|----------|
| Left | Shared session chat |
| Upper right | Claude output |
| Lower right | Codex output |
| Bottom | Master input |

**Resizing panes:** Two draggable dividers let you adjust the layout:
- **Vertical divider** (between chat and agent panels) — drag left/right to give more space to the chat or to the agent output columns
- **Horizontal divider** (between Claude and Codex panels) — drag up/down to give more space to Claude or Codex output

Click and hold on a divider, then drag to resize. The layout adjusts in real time.

## Input

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Ctrl+Enter` | Insert newline |
| `Tab` | Accept autocomplete suggestion |
| `↑` / `↓` | Browse command history or navigate autocomplete |
| `Escape` | Cancel active agent work |
| `Ctrl+C` | Quit |

Type `/` to open the command autocomplete palette.

## Routing

Direct a message to a specific agent:

```
@claude propose a UI improvement
@codex review the current changes
@both compare these two approaches
```

Or use `/ask` for a one-off direct message without changing the default routing:

```
/ask claude explain this function
/ask codex run the tests
/ask both what are the risks here
```

## File Mentions

Inject file content directly into your message using `@path/to/file`:

```
@claude refactor this — @src/auth.py
@both compare the two implementations: @src/v1.py and @src/v2.py
```

The TUI expands the token at submit time. The agent receives the file content inline — no need to ask the agent to read it separately.

**Line limit:** Files over 150 lines are automatically truncated. A notice is appended showing how many lines were omitted and how to use a range:

```
[first 150 of 892 lines — 742 lines omitted. Use @src/auth.py:START-END for a specific range.]
```

**Line ranges:** Inject a specific section of a file:

```
@src/auth.py:40-80        lines 40 to 80 inclusive
```

The header shows `(lines 40–80 of 892)` so the agent knows the context position. The end of the range is clamped to the actual file length.

**Safety guards:**
- Path must resolve inside the repository root — `@../../etc/passwd` is blocked
- Files over 50 KB are skipped entirely
- Binary files (null bytes) are skipped
- `@claude`, `@codex`, `@both`, `@master` are never treated as file paths

## Work Flows

Structured multi-agent flows that coordinate Claude and Codex automatically:

```
/fix <bug>           isolate → patch → review
/audit <target>      independent dual review + comparison
/feature <desc>      design → implement → review
/refactor <target>   map call sites → refactor → verify
/analyze <question>  dual analysis + position comparison
/decide <text>       log a decision to collaboration memory
```

## Session Commands

```
/fresh                   start a new session (clears history)
/resume                  resume from last saved state
/status                  show current session state
/save                    persist session state to disk
/exit                    exit the TUI
```

## Dangerous Operation Enforcement

Certain operations require your explicit approval before an agent executes them:

```
git push (any form)        git tag
npm publish / npm pack     twine upload
pip install --user         rm -rf
docker push                kubectl apply / kubectl delete
CI/CD pipeline file edits  (.github/workflows/, .gitlab-ci.yml, Jenkinsfile)
```

**Claude (pre-execution blocking):** When Claude's tool call matches a dangerous operation, the subprocess is aborted and a full-screen approval modal appears. The turn pauses until you respond:
- `y` or the Approve button — allows that specific command and continues
- `n`, `Escape`, or the Deny button — blocks it; Claude stops and waits for your next instruction

Approval is command-specific: approving `git push origin main` does not approve `git push --force origin main`.

**Codex (post-execution notice):** Codex's event stream only surfaces tool calls after they have already run. The TUI cannot block them pre-execution. Instead, a `⚠ POST-EXEC` warning appears in the Codex pane after the fact. The same command will not produce a repeated warning in the same session.

The permission policy is also injected into every agent prompt, asking both agents to request `Master: requesting permission to ...` before executing dangerous operations.

To configure which operations are gated, see the `dangerous_ops` key in [Configuration](#configuration) below.

## Agent Control

```
/approve                 approve a pending agent request
/deny                    deny a pending agent request
/compare                 run a manual comparison round between agents
/mode <value>            broadcast a mode change to both agents
/role <target> <role>    reassign agent roles at runtime
```

Role targets: `claude`, `codex`, `both`. Role values: `implementer`, `reviewer`, `collaborator`.

Example:
```
/role codex implementer
/role both reviewer
```

## Collaboration Memory

```
/decision D-XXX | Title | Decision | Rationale    log a structured decision
/topic T-XXX | Title | Codex pos | Claude pos      open a scratchpad topic
/resolve-topic T-XXX | D-XXX                       mark a topic resolved
```

Decisions are appended to `.collab/memory.md`. Topics go to `.collab/scratchpad.md`. Both files are read by agents at session start and on update.

## Export

```
/export claude session_claude.txt     export Claude pane to file
/export codex session_codex.txt       export Codex pane to file
/export master session_chat.txt       export the shared chat to file
```

## Diff Viewer

After each agent turn, `▶ [DIFF] <file>` links appear inline for every file the agent changed. Each link is permanently bound to the exact patch from that turn — later edits to the same file do not overwrite earlier diff history.

- Click a link to open a full-screen diff overlay (colour-coded: `+` green, `-` red, `@@` cyan)
- Click `⎘ Copy` to copy the raw diff to the clipboard
- Click `X` to close

## Themes

```
/theme akritrim    (default)
/theme nord
/theme dracula
/theme solarized
```

The active theme is saved to `.collab/settings.toml` and restored on next launch.

## Configuration

Settings live in `.collab/settings.toml` inside your repository. The file is created with defaults on first run. Edit it directly — changes take effect on the next launch.

```toml
[app]
# Session identity and default agent roles
identity = "codex"
role = "collaborator"
codex_role = "reviewer"
claude_role = "implementer"

# Maximum number of automatic agent relay turns before pausing for Master input
max_auto_turns = 12

# UI theme: akritrim | nord | dracula | solarized
theme = "akritrim"

# Model overrides — change to use a different model for either agent
claude_model = "claude-sonnet-4-6"
codex_model = "gpt-5.4"

# Set to false to allow running outside a git repository.
# Codex will receive --skip-git-repo-check; diff display still works.
# Can also be overridden per run: --no-require-git
require_git = true

# Dangerous-operation gate.
# Agents must request Master approval before executing the listed operations.
# Stream-level interception aborts Claude's subprocess before the tool runs.
# Codex receives a post-execution warning (its event stream fires after execution).
#
# Set to an empty list to disable the gate entirely:
# dangerous_ops = []
#
# Or restrict to specific operations:
# dangerous_ops = ["git push", "npm publish", "rm -rf"]
#
# Omit the key to use the full default set (all operations listed above).

# TUI debug log — records lifecycle events, routing decisions, and stream activity.
# Useful for diagnosing unexpected routing or stuck sessions.
# The Claude stream JSONL log (one entry per turn) is always written to
# .collab/session/stream_debug.jsonl separately.
# debug_log = ".collab/session/tui_debug.log"
```

## Run Options

CLI flags override the corresponding `settings.toml` values for that run only:

```bash
akritrim-colab run --codex-role reviewer --claude-role implementer
akritrim-colab run --claude-model claude-opus-4-6
akritrim-colab run --no-require-git
akritrim-colab run --debug-log .collab/session/tui_debug.log
```

## Development

**Clone and install in editable mode:**

```bash
git clone https://github.com/akritrim/akritrim-colab
cd akritrim-colab
pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv pip install -e ".[dev]"
```

**Run from source** (after editable install):

```bash
akritrim-colab
# or
python -m akritrim_colab
```

**Run the test suite:**

```bash
pytest
```

Tests use `pytest-asyncio` (auto mode). No live agent connections are required — the test suite mocks all CLI subprocess calls.

**Project layout:**

```
src/akritrim_colab/
  app.py                     Main TUI — CollabApp, all UI, routing, and relay logic
  cli.py                     Entry point — init / run subcommands, settings wiring
  config.py                  ProjectConfig, AppSettings, path resolution
  agents/
    claude_adapter.py        Claude CLI wrapper — streaming, session resume
    codex_adapter.py         Codex CLI wrapper — streaming, thread resume
    stream_events.py         StreamEvent dataclass shared by both adapters
tests/                       pytest test suite
.collab/                     Per-repo collaboration files (created on first run)
  memory.md                  Permanent decisions log
  scratchpad.md              Open topics and working notes
  rules.md                   Collaboration protocol rules
  settings.toml              User configuration
  session/                   Runtime state (gitignored)
```
