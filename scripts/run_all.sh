#!/usr/bin/env bash
# Start the translator (optional), the FastAPI model service, and the Streamlit UI.
# Usage: bash scripts/run_all.sh          (Ctrl+C stops everything)
# Env vars (all optional): API_PORT (8001), UI_PORT (8501), TRANSLATOR_PORT (8080),
#   SKIP_TRANSLATOR=1, plus GEMINI_API_KEY / ANTHROPIC_API_KEY / MILVUS_* for the RAG tabs.
# Milvus is not started here: cd deploy/milvus && docker compose up -d
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API_PORT="${API_PORT:-8001}"
UI_PORT="${UI_PORT:-8501}"
TRANSLATOR_PORT="${TRANSLATOR_PORT:-8080}"

PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if [ "${SKIP_TRANSLATOR:-0}" != "1" ]; then
  (cd "$ROOT/services/translator" && exec uvicorn translator:app --host 127.0.0.1 --port "$TRANSLATOR_PORT") &
  PIDS+=($!)
  export TRANSLATOR_URL="http://127.0.0.1:$TRANSLATOR_PORT"
fi

(cd "$ROOT/app" && exec uvicorn main:app --host 127.0.0.1 --port "$API_PORT") &
PIDS+=($!)
export OBESITY_API_URL="http://127.0.0.1:$API_PORT"

for _ in $(seq 1 60); do
  curl -sf "$OBESITY_API_URL/health" >/dev/null && break
  sleep 1
done
echo "API ready at $OBESITY_API_URL (docs: $OBESITY_API_URL/docs)"

(cd "$ROOT/app" && exec streamlit run obesity_app_v2.py --server.port "$UI_PORT" --server.headless true) &
PIDS+=($!)
echo "UI at http://localhost:$UI_PORT"
wait
