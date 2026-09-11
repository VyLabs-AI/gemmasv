#!/usr/bin/env bash
# Data-free Kimi replay, transport, and checkpoint-contract checks.
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

"$PY" -m pytest \
  tests/test_kimi_checkpoint_tradeoff.py \
  tests/test_kimi_decay.py \
  tests/test_kimi_graft.py \
  tests/test_kimi_probes.py \
  tests/test_kimi_records.py \
  tests/test_kimi_state.py \
  tests/test_persistent_memory.py \
  tests/test_separability.py \
  -p no:cacheprovider \
  -q

echo "Kimi replay and transport verification passed."
