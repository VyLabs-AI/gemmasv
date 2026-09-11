#!/usr/bin/env bash
# Resumable, one-target/one-condition robustness shards for Apple M3 systems.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$ROOT/.venv311/bin/python" ]]; then
  PY="$ROOT/.venv311/bin/python"
else
  PY=python3
fi

MODE="${1:---paper}"
case "$MODE" in
  --pilot)
    TARGETS="${TARGETS:-3}"
    PAIRED_TARGETS="${PAIRED_TARGETS:-3}"
    SAMPLES="${SAMPLES:-16}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
    K_VALUES="${K_VALUES:-1,2,4,8,16}"
    ;;
  --paper)
    # Admission gates (secret extractability) reject roughly 40% of
    # targets, so attempt 40 to admit roughly 20 per mode.
    TARGETS="${TARGETS:-40}"
    PAIRED_TARGETS="${PAIRED_TARGETS:-40}"
    SAMPLES="${SAMPLES:-200}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
    K_VALUES="${K_VALUES:-1,2,4,8,16,32,64,128}"
    ;;
  *)
    echo "usage: $0 [--pilot|--paper]" >&2
    exit 2
    ;;
esac

MODEL="${MODEL:-google/gemma-3-1b-pt}"
LORA="${LORA:-outputs/gemma_sv_distill/lora_adapter}"
DEVICE="${DEVICE:-mps}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-0}"
CONDITIONS_CSV="${CONDITIONS:-present,decrement,decay,icul,never}"
IFS=',' read -r -a CONDITION_LIST <<< "$CONDITIONS_CSV"

# "stemkv" marks the template-aware probe protocol with cached decoding;
# never mix shard roots across probe or sampler protocols.
RUN_TAG="stemkv_n${SAMPLES}_tok${MAX_NEW_TOKENS}_seed${SEED}"
SHARD_ROOT="${SHARD_ROOT:-outputs/gemma_sv_eval/robust_shards_${RUN_TAG}}"
mkdir -p "$SHARD_ROOT"

# Refuse to race a concurrent orchestrator on the same shard root.
LOCK_DIR="$SHARD_ROOT/.orchestrator.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another orchestrator holds $LOCK_DIR; remove it if stale" >&2
  exit 3
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT

valid_json() {
  "$PY" -c "import json; json.load(open('$1'))" >/dev/null 2>&1
}

run_shard() {
  local eval_mode="$1"
  local target_index="$2"
  local condition="$3"
  local output="$SHARD_ROOT/${eval_mode}_t$(printf '%03d' "$target_index")_${condition}.json"
  local temporary="${output}.tmp"

  if [[ -s "$output" ]] && valid_json "$output"; then
    echo "SHARD_SKIP mode=$eval_mode target=$target_index condition=$condition"
    return
  fi
  rm -f "$temporary"
  echo "SHARD_START mode=$eval_mode target=$target_index condition=$condition"

  local offsets=()
  if [[ "$eval_mode" == "leak" ]]; then
    offsets=(--targets 1 --target-start "$target_index")
  else
    offsets=(--paired-targets 1 --paired-start "$target_index")
  fi

  "$PY" -m gemma_sv.eval_robust_unlearning \
    --mode "$eval_mode" \
    "${offsets[@]}" \
    --conditions "$condition" \
    --samples "$SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --k "$K_VALUES" \
    --seed "$SEED" \
    --model "$MODEL" \
    --lora "$LORA" \
    --device "$DEVICE" \
    --save-generations \
    --out "$temporary"
  mv "$temporary" "$output"
  echo "SHARD_DONE mode=$eval_mode target=$target_index condition=$condition"
}

# LEAK_INDICES / PAIRED_INDICES (space-separated) restrict the grid to
# pre-screened admissible targets; admission is still re-checked per shard.
if [[ -n "${LEAK_INDICES:-}" ]]; then
  read -r -a LEAK_TARGET_LIST <<< "$LEAK_INDICES"
else
  LEAK_TARGET_LIST=()
  for ((target = 0; target < TARGETS; target++)); do
    LEAK_TARGET_LIST+=("$target")
  done
fi
if [[ -n "${PAIRED_INDICES:-}" ]]; then
  read -r -a PAIRED_TARGET_LIST <<< "$PAIRED_INDICES"
else
  PAIRED_TARGET_LIST=()
  for ((target = 0; target < PAIRED_TARGETS; target++)); do
    PAIRED_TARGET_LIST+=("$target")
  done
fi

# The ${arr[@]+...} form keeps empty arrays legal under bash 3.2's set -u.
for target in ${LEAK_TARGET_LIST[@]+"${LEAK_TARGET_LIST[@]}"}; do
  for condition in "${CONDITION_LIST[@]}"; do
    run_shard leak "$target" "$condition"
  done
done

for target in ${PAIRED_TARGET_LIST[@]+"${PAIRED_TARGET_LIST[@]}"}; do
  for condition in "${CONDITION_LIST[@]}"; do
    run_shard paired "$target" "$condition"
  done
done

shopt -s nullglob
shards=("$SHARD_ROOT"/*.json)
if (( ${#shards[@]} == 0 )); then
  echo "no completed shards found" >&2
  exit 1
fi

MERGED="${MERGED_OUT:-outputs/gemma_sv_eval/robust_unlearning.json}"
"$PY" -m gemma_sv.merge_robust_shards "${shards[@]}" --out "$MERGED"
echo "ROBUST_RUN_DONE shards=${#shards[@]} merged=$MERGED"

