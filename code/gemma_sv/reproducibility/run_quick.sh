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
  tests/test_boundary_evidence_publication_v3.py \
  tests/test_publish_longmemeval_chat_result.py \
  tests/test_exactness.py \
  tests/test_build_longmemeval_chat_v3_paper_macros.py \
  tests/test_build_longmemeval_chat_suffix_disclosure_crosstab_v2.py \
  -p no:cacheprovider -q

"$PY" -m pytest \
  tests/test_longmemeval_chat_cohort_v3.py \
  tests/test_summarize_longmemeval_chat_v3.py \
  tests/test_longmemeval_chat_response_generation_audit_v2.py \
  -k 'not rebuild_is_value and not rehydration and not real_exact and not real_sample and not real_outputs' \
  -p no:cacheprovider -q

echo "Gemma offline result and contract verification passed."
