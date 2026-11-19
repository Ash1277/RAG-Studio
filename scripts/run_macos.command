#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11-3.13 is required."
  exit 1
fi
if [ ! -d .venv ]; then
  echo "[RAG Studio 1.0] Creating virtual environment (first run only)..."
  "$PYTHON_BIN" -m venv .venv
fi
REQ_HASH=$(shasum -a 256 requirements.txt | awk '{print $1}')
STAMP_FILE=".venv/.deps_stamp"
OLD_HASH=""
[ -f "$STAMP_FILE" ] && OLD_HASH=$(cat "$STAMP_FILE")
if [ "$REQ_HASH" != "$OLD_HASH" ]; then
  echo "[RAG Studio 1.0] Installing dependencies (cached for future launches)..."
  .venv/bin/python -m pip install --disable-pip-version-check --prefer-binary -r requirements.txt
  echo "$REQ_HASH" > "$STAMP_FILE"
fi
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8765 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
sleep 2
open http://127.0.0.1:8765 >/dev/null 2>&1 || true
wait "$SERVER_PID"
