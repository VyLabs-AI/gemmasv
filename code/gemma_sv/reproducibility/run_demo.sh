#!/usr/bin/env bash
# Start one demo API process. Serve the static site in a second terminal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$ROOT/.venv311/bin/python" ]]; then
  PY="$ROOT/.venv311/bin/python"
else
  PY=python3
fi
MODE="${1:-replay}"

export HERO_HOST="${HERO_HOST:-127.0.0.1}"
export HERO_PORT="${HERO_PORT:-8001}"
export HERO_ALLOWED_ORIGINS="${HERO_ALLOWED_ORIGINS:-http://127.0.0.1:8000,http://localhost:8000}"
export HERO_ACCESS_LOG="${HERO_ACCESS_LOG:-0}"

case "$MODE" in
  replay)
    export HERO_ENGINE=replay
    ;;
  live)
    export HERO_ENGINE=gemma
    ;;
  *)
    echo "usage: $0 {replay|live}" >&2
    exit 2
    ;;
esac

cat <<EOF
Starting the $MODE API at http://$HERO_HOST:$HERO_PORT

In a second terminal:
  $PY -m http.server 8000 --directory gemma_sv/demo_site

Then open:
  http://127.0.0.1:8000/?api=http://127.0.0.1:$HERO_PORT
EOF

exec "$PY" -m gemma_sv.demo_server

