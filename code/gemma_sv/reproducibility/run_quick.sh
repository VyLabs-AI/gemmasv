#!/usr/bin/env bash
# Fast, data-free verification using requirements-core.txt.
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
  tests/test_binary.py \
  tests/test_exactness.py \
  tests/test_fast_solver.py \
  tests/test_forgetting_rigor.py \
  tests/test_gemma_multiseed_summary.py \
  tests/test_musique_rag_benchmark.py \
  tests/test_proxy_precision_sweep.py \
  tests/test_ruler_erasure_benchmark.py \
  tests/test_svr.py \
  tests/test_demo_certificate.py \
  tests/test_demo_contract.py \
  tests/test_demo_gate_context.py \
  tests/test_demo_span.py \
  tests/test_demo_state.py \
  tests/test_demo_api.py \
  tests/test_demo_static.py \
  tests/test_mimic_privacy.py \
  tests/test_robust_eval.py \
  -p no:cacheprovider \
  -q

echo "Gemma contract and replay verification passed."

