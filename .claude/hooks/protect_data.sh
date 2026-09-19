#!/bin/bash
# PreToolUse — block direct writes to data/ parquet/csv files.
# These live on S3; manage them with `python data/sync.py`.

INPUT=$(cat)
FILE=$(echo "$INPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null)

if [[ "$FILE" == */data/*.parquet || "$FILE" == */data/*.csv ]]; then
    echo "Blocked: data files live on S3, not in git. Use 'python data/sync.py' to manage them."
    exit 2
fi
exit 0
