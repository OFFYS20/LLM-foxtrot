#!/usr/bin/env bash
# Run this to open Teacher. It sets everything up the first time.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
[ -x "ai_studio/.venv/bin/python" ] && PY="ai_studio/.venv/bin/python"

if ! "$PY" -c "import gradio, torch, transformers" >/dev/null 2>&1; then
  echo "First run — installing what Teacher needs. This takes a few minutes."
  "$PY" -m pip install -r ai_studio/requirements.txt
fi

echo "Opening Teacher in your browser..."
exec "$PY" -m teacher ui "$@"
