#!/usr/bin/env bash
# Model-level integration or paper-scale 1B forgetting evaluations.
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
MODE="${1:---quick}"
DEVICE="${DEVICE:-mps}"
MODEL="${MODEL:-google/gemma-3-1b-pt}"
LORA="${LORA:-outputs/gemma_sv_distill/lora_adapter}"

case "$MODE" in
  --quick)
    echo "=== random-model graft smoke test ==="
    "$PY" -m gemma_sv.smoke_test
    echo "=== complete demo tests ==="
    "$PY" -m pytest tests/test_demo_*.py -p no:cacheprovider -q
    ;;
  --paper)
    test -d "$LORA" || {
      echo "missing recovered adapter: $LORA" >&2
      exit 2
    }
    echo "=== output-level decrement/refit certificate ==="
    "$PY" -m gemma_sv.unlearn_output_demo \
      --model "$MODEL" --lora "$LORA" --trials 31
    "$PY" -m gemma_sv.unlearn_output_demo \
      --model "$MODEL" --lora "$LORA" --sequential 1,2,5,10,20,30
    echo "=== prefill-once decrement/proxy bridge ==="
    "$PY" -m gemma_sv.persistent_bridge_demo \
      --model "$MODEL" --lora "$LORA" \
      --out outputs/gemma_sv_demo/persistent_state_bridge_v2.json

    echo "=== efficacy, specificity, and adversarial robustness ==="
    "$PY" -m gemma_sv.unlearn_eval \
      --model "$MODEL" --lora "$LORA" --targets 40 --device "$DEVICE"
    "$PY" -m gemma_sv.unlearn_attack \
      --model "$MODEL" --lora "$LORA" --targets 50 --device "$DEVICE"
    "$PY" -m gemma_sv.unlearn_relearn \
      --model "$MODEL" --lora "$LORA" --targets 40 --device "$DEVICE"
    "$PY" -m gemma_sv.unlearn_mia_lira \
      --model "$MODEL" --lora "$LORA" --targets 40 \
      --shadows 32 --tests 8 --device "$DEVICE"
    "$PY" -m gemma_sv.unlearn_weightspace \
      --model "$MODEL" --lora "$LORA" --targets 20 --device "$DEVICE"
    ;;
  *)
    echo "usage: $0 [--quick|--paper]" >&2
    exit 2
    ;;
esac

echo "Gemma headline tier completed."

