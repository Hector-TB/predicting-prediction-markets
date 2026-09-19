#!/bin/bash
# PostToolUse — run ruff on any Python file Claude just edited.
# Runs after every Edit/Write; does nothing for non-.py files.

INPUT=$(cat)
FILE=$(echo "$INPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null)

if [[ "$FILE" == *.py ]]; then
    ruff check --fix "$FILE" 2>&1 || true
fi
exit 0
