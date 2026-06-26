#!/usr/bin/env sh
# PostToolUse hook: auto-format + lint-fix a Python file right after Claude edits it.
#
# Mirrors CI's first two gates (`ruff format --check` / `ruff check`) locally so
# formatting + autofixable lint never drift far enough to fail a push. Non-blocking
# by design: it fixes what it can and always exits 0 (CI still enforces the rest).
#
# Input: Claude Code pipes the tool call as JSON on stdin; the edited path is at
# .tool_input.file_path. The only env var the runtime sets is $CLAUDE_PROJECT_DIR.

file_path=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)

# Only act on Python source; skip everything else (and skip if parsing yielded nothing).
case "$file_path" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$file_path" ] || exit 0

# Run from the project root so uv resolves this project's pinned ruff (dev group).
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
# Both passes in one `uv run` so the env resolves once, not twice (format + check are
# distinct subcommands, so the two ruff passes themselves can't merge).
uv run sh -c 'ruff format "$1" >/dev/null 2>&1; ruff check --fix "$1" >/dev/null 2>&1' _ "$file_path"
exit 0
