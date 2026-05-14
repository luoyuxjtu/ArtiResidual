# CLAUDE.md

This file is auto-loaded by Claude Code at session start.

## Session start protocol
Always do these BEFORE any other action:
1. Read PROGRESS.md
2. Read DECISIONS.md (last 5 entries minimum)
3. Read CODE_INDEX.md
4. Summarize current state and wait for my instruction

## Session end protocol
Before I close the session:
1. Update PROGRESS.md
2. Append new decisions to DECISIONS.md
3. Update CODE_INDEX.md if any module status changed
4. Show me the diff

## Hard rules
- Never modify files in `docs/` (those are authoritative spec)
- Never refactor existing code unless explicitly asked
- Always use type hints, docstrings, tensor shape comments
- Always run tests after modifying a module
