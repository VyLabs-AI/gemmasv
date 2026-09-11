#!/usr/bin/env bash
# Recovery, utility, and scale-specific model evaluations.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$ROOT/.venv311/bin/python" ]]; then
  PY="$ROOT/.venv311/bin/python"
else
  PY=python3
fi
MODE="${1:-}"
DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-0 1 2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/gemma_sv_multiseed}"
OUTPUT_ROOT_4B="${OUTPUT_ROOT_4B:-outputs/gemma_sv_4b_multiseed}"
OUTPUT_ROOT_IT="${OUTPUT_ROOT_IT:-outputs/gemma_sv_it_multiseed}"
MODEL_1B="${MODEL_1B:-google/gemma-3-1b-pt}"
MODEL_REVISION_1B="${MODEL_REVISION_1B:-fcf18a2a879aab110ca39f8bffbccd5d49d8eb29}"
MODEL_4B="${MODEL_4B:-google/gemma-3-4b-pt}"
MODEL_REVISION_4B="${MODEL_REVISION_4B:-cc012e0a6d0787b4adcc0fa2c4da74402494554d}"
MODEL_IT="${MODEL_IT:-google/gemma-3-1b-it}"
MODEL_REVISION_IT="${MODEL_REVISION_IT:-dcc83ea841ab6100d6b47a070329e1ba4cf78752}"
BATCH_4B="${BATCH_4B:-8}"
MAX_PARALLEL_4B="${MAX_PARALLEL_4B:-2}"
STAGE1_STEPS="${STAGE1_STEPS:-2000}"
STAGE2_STEPS="${STAGE2_STEPS:-6000}"
EVAL_BLOCKS="${EVAL_BLOCKS:-400}"
BATCH="${BATCH:-8}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-100}"
CERT_TRIALS="${CERT_TRIALS:-31}"
TARGET_SEED="${TARGET_SEED:-0}"
AUDIT_WARMUP="${AUDIT_WARMUP:-1}"
AUDIT_REPEATS="${AUDIT_REPEATS:-3}"
ACTIVE_LOCK=""

require_adapter() {
  test -d "$1" || {
    echo "missing adapter: $1" >&2
    exit 2
  }
}

result_complete() {
  "$PY" - "$1" "$2" "${3:-}" "${4:-}" "${5:-}" "${6:-}" "${7:-}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
mode = sys.argv[3]
expected_model = sys.argv[4]
expected_revision = sys.argv[5]
expected_seed = sys.argv[6]
expected_offset = sys.argv[7]
if not path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
stream = payload.get("stage2_stream") or {}
if payload.get("schema") != "gemma-sv-recovery-v2":
    raise SystemExit(1)
if int(stream.get("batches", -1)) != expected:
    raise SystemExit(1)
if not stream.get("sha256"):
    raise SystemExit(1)
if mode == "control" and not (payload.get("pairing") or {}).get("valid"):
    raise SystemExit(1)
config = payload.get("config") or {}
if expected_model and payload.get("model") != expected_model:
    raise SystemExit(1)
if expected_revision and config.get("model_revision") != expected_revision:
    raise SystemExit(1)
if expected_seed:
    seed = int(expected_seed)
    if any(int(config.get(name, -1)) != seed for name in ("seed", "data_seed", "init_seed")):
        raise SystemExit(1)
if expected_offset and int(payload.get("stage2_offset_batches", -1)) != int(expected_offset):
    raise SystemExit(1)
adapter = path.parent / "lora_adapter" / "adapter_model.safetensors"
raise SystemExit(0 if adapter.exists() else 1)
PY
}

json_valid() {
  "$PY" - "$1" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if isinstance(payload, dict) else 1)
PY
}

audit_complete() {
  "$PY" - "$1" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "completed" else 1)
PY
}

run_logged() {
  local log="$1"
  shift
  mkdir -p "$(dirname "$log")"
  "$@" 2>&1 | tee "$log"
}

acquire_lock() {
  local lock="$1"
  mkdir -p "$(dirname "$lock")"
  if ! mkdir "$lock" 2>/dev/null; then
    echo "another model orchestrator holds $lock" >&2
    exit 2
  fi
  ACTIVE_LOCK="$lock"
  trap '[[ -z "$ACTIVE_LOCK" ]] || rmdir "$ACTIVE_LOCK" 2>/dev/null || true' EXIT
}

write_environment_lock() {
  local destination="$1/environment-lock.txt"
  local temporary="$destination.tmp"
  mkdir -p "$1"
  "$PY" -m pip freeze --all > "$temporary"
  mv "$temporary" "$destination"
}

write_source_state() {
  "$PY" -m gemma_sv.capture_source_state \
    --repo "$ROOT" \
    --out-dir "$1/source_state"
}

write_seed_cohort() {
  local destination="$1"
  local scale="$2"
  "$PY" - "$destination" "$scale" $SEEDS <<'PY'
import json
from pathlib import Path
import sys

destination = Path(sys.argv[1])
scale = sys.argv[2]
seeds = [int(value) for value in sys.argv[3:]]
if len(set(seeds)) != len(seeds):
    raise SystemExit("seed cohort contains duplicates")
payload = {
    "schema": "gemma-sv-predeclared-seed-cohort-v1",
    "scale": scale,
    "seeds": seeds,
    "retention_policy": (
        "retain every listed seed regardless of utility cost, admission, "
        "loss trajectory, or confidence-interval effect"
    ),
}
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    existing = json.loads(destination.read_text())
    if existing != payload:
        raise SystemExit(f"existing cohort declaration differs: {destination}")
else:
    destination.write_text(json.dumps(payload, indent=2) + "\n")
print(f"seed cohort -> {destination}")
PY
}

run_4b_seed() {
  local seed="$1"
  local seed_root="$OUTPUT_ROOT_4B/seed-$seed"
  local recovered_out="$seed_root/recovered"
  local control_out="$seed_root/control"
  local recovered_result="$recovered_out/results.json"
  local control_result="$control_out/control_results.json"
  local stage2_offset="$((STAGE1_STEPS + 1))"

  if result_complete \
      "$recovered_result" "$STAGE2_STEPS" "" \
      "$MODEL_4B" "$MODEL_REVISION_4B" "$seed" "$stage2_offset"; then
    echo "4B seed $seed recovery already complete; skipping"
  else
    run_logged "$recovered_out/run.log" \
      "$PY" -u -m gemma_sv.run_distill \
        --model "$MODEL_4B" \
        --model-revision "$MODEL_REVISION_4B" \
        --device "$DEVICE" \
        --batch "$BATCH_4B" \
        --stage1-steps "$STAGE1_STEPS" \
        --stage2-steps "$STAGE2_STEPS" \
        --eval-blocks "$EVAL_BLOCKS" \
        --ckpt-every "$CKPT_EVERY" \
        --log-every "$LOG_EVERY" \
        --seed "$seed" \
        --data-seed "$seed" \
        --init-seed "$seed" \
        --out "$recovered_out"
  fi

  if result_complete \
      "$control_result" "$STAGE2_STEPS" control \
      "$MODEL_4B" "$MODEL_REVISION_4B" "$seed" "$stage2_offset"; then
    echo "4B seed $seed matched control already complete; skipping"
  else
    run_logged "$control_out/run.log" \
      "$PY" -u -m gemma_sv.run_distill \
        --control-only \
        --model "$MODEL_4B" \
        --model-revision "$MODEL_REVISION_4B" \
        --device "$DEVICE" \
        --batch "$BATCH_4B" \
        --stage1-steps "$STAGE1_STEPS" \
        --stage2-steps "$STAGE2_STEPS" \
        --eval-blocks "$EVAL_BLOCKS" \
        --ckpt-every "$CKPT_EVERY" \
        --log-every "$LOG_EVERY" \
        --seed "$seed" \
        --data-seed "$seed" \
        --init-seed "$seed" \
        --data-skip "$stage2_offset" \
        --compare-to "$recovered_result" \
        --require-matched-stream \
        --out "$control_out"
  fi
}

run_it_seed() {
  local seed="$1"
  local seed_root="$OUTPUT_ROOT_IT/seed-$seed"
  local recovered_out="$seed_root/recovered"
  local control_out="$seed_root/control"
  local recovered_result="$recovered_out/results.json"
  local control_result="$control_out/control_results.json"
  local stage1_steps=2000
  local stage2_steps=4000
  local stage2_offset=2001
  local target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

  if result_complete \
      "$recovered_result" "$stage2_steps" "" \
      "$MODEL_IT" "$MODEL_REVISION_IT" "$seed" "$stage2_offset"; then
    echo "1B-IT seed $seed recovery already complete; skipping"
  else
    run_logged "$recovered_out/run.log" \
      "$PY" -u -m gemma_sv.run_distill \
        --model "$MODEL_IT" \
        --model-revision "$MODEL_REVISION_IT" \
        --device "$DEVICE" \
        --batch 4 \
        --stage1-steps "$stage1_steps" \
        --stage2-steps "$stage2_steps" \
        --lr2 0.0003 \
        --rank 64 \
        --target-modules "$target_modules" \
        --lora-global-only \
        --joint-kpar-stage2 \
        --eval-blocks 50 \
        --ckpt-every 1000 \
        --log-every 200 \
        --seed "$seed" \
        --data-seed "$seed" \
        --init-seed "$seed" \
        --out "$recovered_out"
  fi

  if result_complete \
      "$control_result" "$stage2_steps" control \
      "$MODEL_IT" "$MODEL_REVISION_IT" "$seed" "$stage2_offset"; then
    echo "1B-IT seed $seed matched control already complete; skipping"
  else
    run_logged "$control_out/run.log" \
      "$PY" -u -m gemma_sv.run_distill \
        --control-only \
        --model "$MODEL_IT" \
        --model-revision "$MODEL_REVISION_IT" \
        --device "$DEVICE" \
        --batch 4 \
        --stage1-steps "$stage1_steps" \
        --stage2-steps "$stage2_steps" \
        --lr2 0.0003 \
        --rank 64 \
        --target-modules "$target_modules" \
        --lora-global-only \
        --joint-kpar-stage2 \
        --eval-blocks 50 \
        --ckpt-every 1000 \
        --log-every 200 \
        --seed "$seed" \
        --data-seed "$seed" \
        --init-seed "$seed" \
        --data-skip "$stage2_offset" \
        --compare-to "$recovered_result" \
        --require-matched-stream \
        --out "$control_out"
  fi
}

run_admission_cohort() {
  local root="$1"
  local model="$2"
  local revision="$3"
  local window="$4"
  local prefix_fillers="$5"
  local count=0
  local reports=()
  local seed
  for seed in $SEEDS; do
    local adapter="$root/seed-$seed/recovered/lora_adapter"
    local out="$root/seed-$seed/behavioral/admission.json"
    require_adapter "$adapter"
    if json_valid "$out"; then
      echo "seed $seed admission report already complete; skipping"
    else
      run_logged "$root/seed-$seed/behavioral/admission.log" \
        "$PY" -u -m gemma_sv.eval_whole_record_unlearning \
          --admission-only \
          --model "$model" \
          --model-revision "$revision" \
          --lora "$adapter" \
          --device "$DEVICE" \
          --window "$window" \
          --n-fill 22 \
          --prefix-fillers "$prefix_fillers" \
          --seed "$seed" \
          --out "$out"
    fi
    reports+=(--report "$out")
    count=$((count + 1))
  done
  "$PY" -m gemma_sv.summarize_whole_record_multiseed \
    --minimum-reports "$count" \
    "${reports[@]}" \
    --out "$root/admission_summary.json"
}

wait_for_batch() {
  local status=0
  local pid
  for pid in "$@"; do
    wait "$pid" || status=$?
  done
  return "$status"
}

case "$MODE" in
  recover-1b)
    acquire_lock "$OUTPUT_ROOT/.recover-1b.lock"
    write_environment_lock "$OUTPUT_ROOT"
    write_source_state "$OUTPUT_ROOT"
    write_seed_cohort "$OUTPUT_ROOT/cohort-${SEEDS// /-}.json" "1B"
    for seed in $SEEDS; do
      SEED_ROOT="$OUTPUT_ROOT/seed-$seed"
      RECOVERED_OUT="$SEED_ROOT/recovered"
      CONTROL_OUT="$SEED_ROOT/control"
      RECOVERED_RESULT="$RECOVERED_OUT/results.json"
      CONTROL_RESULT="$CONTROL_OUT/control_results.json"

      if result_complete "$RECOVERED_RESULT" "$STAGE2_STEPS"; then
        echo "seed $seed recovery already complete; skipping"
      else
        run_logged "$RECOVERED_OUT/run.log" \
          "$PY" -u -m gemma_sv.run_distill \
            --model google/gemma-3-1b-pt \
            --device "$DEVICE" \
            --batch "$BATCH" \
            --stage1-steps "$STAGE1_STEPS" \
            --stage2-steps "$STAGE2_STEPS" \
            --eval-blocks "$EVAL_BLOCKS" \
            --ckpt-every "$CKPT_EVERY" \
            --log-every "$LOG_EVERY" \
            --seed "$seed" \
            --data-seed "$seed" \
            --init-seed "$seed" \
            --out "$RECOVERED_OUT"
      fi

      if result_complete "$CONTROL_RESULT" "$STAGE2_STEPS" control; then
        echo "seed $seed matched control already complete; skipping"
      else
        run_logged "$CONTROL_OUT/run.log" \
          "$PY" -u -m gemma_sv.run_distill \
            --control-only \
            --model google/gemma-3-1b-pt \
            --device "$DEVICE" \
            --batch "$BATCH" \
            --stage1-steps "$STAGE1_STEPS" \
            --stage2-steps "$STAGE2_STEPS" \
            --eval-blocks "$EVAL_BLOCKS" \
            --ckpt-every "$CKPT_EVERY" \
            --log-every "$LOG_EVERY" \
            --seed "$seed" \
            --data-seed "$seed" \
            --init-seed "$seed" \
            --data-skip "$((STAGE1_STEPS + 1))" \
            --compare-to "$RECOVERED_RESULT" \
            --require-matched-stream \
            --out "$CONTROL_OUT"
      fi
    done
    ;;
  utility-1b)
    acquire_lock "$OUTPUT_ROOT/.utility-1b.lock"
    for seed in $SEEDS; do
      SEED_ROOT="$OUTPUT_ROOT/seed-$seed"
      RECOVERED="$SEED_ROOT/recovered/lora_adapter"
      CONTROL="$SEED_ROOT/control/lora_adapter"
      UTILITY_OUT="$SEED_ROOT/utility"
      require_adapter "$RECOVERED"
      require_adapter "$CONTROL"
      mkdir -p "$UTILITY_OUT"
      if json_valid "$UTILITY_OUT/tasks.json"; then
        echo "seed $seed zero-shot utility already complete; skipping"
      else
        run_logged "$UTILITY_OUT/tasks.log" \
          "$PY" -u -m gemma_sv.eval_tasks \
            --model google/gemma-3-1b-pt \
            --recovered-lora "$RECOVERED" \
            --control-lora "$CONTROL" \
            --run-seed "$seed" \
            --limit 2000 \
            --device "$DEVICE" \
            --out "$UTILITY_OUT/tasks.json"
      fi
      if json_valid "$UTILITY_OUT/ppl2.json"; then
        echo "seed $seed cross-corpus utility already complete; skipping"
      else
        run_logged "$UTILITY_OUT/ppl2.log" \
          "$PY" -u -m gemma_sv.eval_ppl2 \
            --model google/gemma-3-1b-pt \
            --recovered-lora "$RECOVERED" \
            --control-lora "$CONTROL" \
            --run-seed "$seed" \
            --blocks 300 \
            --device "$DEVICE" \
            --out "$UTILITY_OUT/ppl2.json"
      fi
    done
    ;;
  certificates-1b)
    acquire_lock "$OUTPUT_ROOT/.certificates-1b.lock"
    for seed in $SEEDS; do
      SEED_ROOT="$OUTPUT_ROOT/seed-$seed"
      RECOVERED="$SEED_ROOT/recovered/lora_adapter"
      CERT_OUT="$SEED_ROOT/certificate"
      CERT_JSON="$CERT_OUT/output_kl.json"
      require_adapter "$RECOVERED"
      if json_valid "$CERT_JSON"; then
        echo "seed $seed output certificate already complete; skipping"
      else
        run_logged "$CERT_OUT/run.log" \
          "$PY" -u -m gemma_sv.unlearn_output_demo \
            --model google/gemma-3-1b-pt \
            --lora "$RECOVERED" \
            --trials "$CERT_TRIALS" \
            --target-seed "$TARGET_SEED" \
            --run-seed "$seed" \
            --json-out "$CERT_JSON"
      fi
    done
    ;;
  deletion-audit-1b)
    acquire_lock "$OUTPUT_ROOT/.deletion-audit-1b.lock"
    for seed in $SEEDS; do
      SEED_ROOT="$OUTPUT_ROOT/seed-$seed"
      RECOVERED="$SEED_ROOT/recovered/lora_adapter"
      AUDIT_OUT="$SEED_ROOT/deletion_audit"
      AUDIT_JSON="$AUDIT_OUT/persistent_baselines.json"
      require_adapter "$RECOVERED"
      if audit_complete "$AUDIT_JSON"; then
        echo "seed $seed persistent deletion audit already complete; skipping"
      else
        run_logged "$AUDIT_OUT/run.log" \
          "$PY" -u -m gemma_sv.eval_persistent_deletion_baselines \
            --model google/gemma-3-1b-pt \
            --adapters "$RECOVERED" \
            --devices "$DEVICE" \
            --seed "$seed" \
            --warmup "$AUDIT_WARMUP" \
            --repeats "$AUDIT_REPEATS" \
            --certificate-device cpu \
            --certificate-warmup 0 \
            --certificate-repeats 1 \
            --out "$AUDIT_JSON" \
            --overwrite
      fi
    done
    ;;
  admission-1b)
    run_admission_cohort \
      "$OUTPUT_ROOT" "$MODEL_1B" "$MODEL_REVISION_1B" 512 8
    ;;
  summarize-1b)
    SUMMARY_ARGS=()
    for seed in $SEEDS; do
      SEED_ROOT="$OUTPUT_ROOT/seed-$seed"
      SUMMARY_ARGS+=(
        --recovery "$SEED_ROOT/recovered/results.json"
        --control "$SEED_ROOT/control/control_results.json"
        --certificate "$SEED_ROOT/certificate/output_kl.json"
        --audit "$SEED_ROOT/deletion_audit/persistent_baselines.json"
      )
      if json_valid "$SEED_ROOT/utility/ppl2.json"; then
        SUMMARY_ARGS+=(--cross-corpus "$SEED_ROOT/utility/ppl2.json")
      fi
      if json_valid "$SEED_ROOT/utility/tasks.json"; then
        SUMMARY_ARGS+=(--zero-shot "$SEED_ROOT/utility/tasks.json")
      fi
    done
    "$PY" -m gemma_sv.summarize_multiseed \
      "${SUMMARY_ARGS[@]}" \
      --out "$OUTPUT_ROOT/summary.json"
    "$PY" -m gemma_sv.publish_iclr_summary \
      --summary "$OUTPUT_ROOT/summary.json" \
      --out gemma_sv/benchmarks/iclr_multiseed_v1.json
    ;;
  iclr-1b)
    "$0" recover-1b
    "$0" utility-1b
    "$0" certificates-1b
    "$0" deletion-audit-1b
    "$0" summarize-1b
    ;;
  recover-4b)
    acquire_lock "$OUTPUT_ROOT_4B/.recover-4b.lock"
    write_environment_lock "$OUTPUT_ROOT_4B"
    write_source_state "$OUTPUT_ROOT_4B"
    write_seed_cohort "$OUTPUT_ROOT_4B/cohort-${SEEDS// /-}.json" "4B"
    if (( MAX_PARALLEL_4B < 1 )); then
      echo "MAX_PARALLEL_4B must be at least 1" >&2
      exit 2
    fi
    echo "running 4B seeds with up to $MAX_PARALLEL_4B parallel pipelines"
    PIDS=()
    for seed in $SEEDS; do
      (trap - EXIT; run_4b_seed "$seed") &
      PIDS+=("$!")
      if (( ${#PIDS[@]} >= MAX_PARALLEL_4B )); then
        wait_for_batch "${PIDS[@]}"
        PIDS=()
      fi
    done
    if (( ${#PIDS[@]} )); then
      wait_for_batch "${PIDS[@]}"
    fi
    ;;
  admission-4b)
    run_admission_cohort \
      "$OUTPUT_ROOT_4B" "$MODEL_4B" "$MODEL_REVISION_4B" 1024 11
    ;;
  recover-1b-it)
    acquire_lock "$OUTPUT_ROOT_IT/.recover-1b-it.lock"
    write_environment_lock "$OUTPUT_ROOT_IT"
    write_source_state "$OUTPUT_ROOT_IT"
    write_seed_cohort "$OUTPUT_ROOT_IT/cohort-${SEEDS// /-}.json" "1B-IT"
    for seed in $SEEDS; do
      run_it_seed "$seed"
    done
    ;;
  summarize-1b-it)
    SUMMARY_ARGS=()
    for seed in $SEEDS; do
      SUMMARY_ARGS+=(
        --recovery "$OUTPUT_ROOT_IT/seed-$seed/recovered/results.json"
        --control "$OUTPUT_ROOT_IT/seed-$seed/control/control_results.json"
      )
    done
    "$PY" -m gemma_sv.summarize_multiseed \
      "${SUMMARY_ARGS[@]}" \
      --out "$OUTPUT_ROOT_IT/summary.json"
    ;;
  admission-1b-it)
    run_admission_cohort \
      "$OUTPUT_ROOT_IT" "$MODEL_IT" "$MODEL_REVISION_IT" 512 8
    ;;
  summarize-4b)
    "$PY" -m gemma_sv.summarize_scaling_4b \
      --root "$OUTPUT_ROOT_4B" \
      --seeds $SEEDS \
      --out "$OUTPUT_ROOT_4B/summary.json"
    "$PY" -m gemma_sv.summarize_scaling_4b \
      --root "$OUTPUT_ROOT_4B" \
      --seeds $SEEDS \
      --out gemma_sv/benchmarks/iclr_4b_multiseed_v1.json
    ;;
  scale-4b)
    RECOVERED=outputs/gemma_sv_distill_4b/lora_adapter
    CONTROL=outputs/gemma_sv_distill_4b_control/lora_adapter
    require_adapter "$RECOVERED"
    require_adapter "$CONTROL"
    "$PY" -m gemma_sv.unlearn_output_demo \
      --model google/gemma-3-4b-pt --lora "$RECOVERED" --trials 31
    "$PY" -m gemma_sv.eval_ppl2 \
      --model google/gemma-3-4b-pt \
      --recovered-lora "$RECOVERED" \
      --control-lora "$CONTROL" \
      --blocks 400 \
      --device "$DEVICE" \
      --out outputs/gemma_sv_eval/ppl2_4b.json
    ;;
  scale-12b)
    RECOVERED=outputs/gemma_sv_distill_12b/lora_adapter
    CONTROL=outputs/gemma_sv_distill_12b_control/lora_adapter
    require_adapter "$RECOVERED"
    require_adapter "$CONTROL"
    "$PY" -m gemma_sv.unlearn_output_demo \
      --model google/gemma-3-12b-pt --lora "$RECOVERED" --trials 31
    "$PY" -m gemma_sv.eval_ppl2 \
      --model google/gemma-3-12b-pt \
      --recovered-lora "$RECOVERED" \
      --control-lora "$CONTROL" \
      --blocks 400 \
      --device "$DEVICE" \
      --out outputs/gemma_sv_eval/ppl2_12b.json
    ;;
  *)
    echo "usage: $0 {recover-1b|utility-1b|certificates-1b|deletion-audit-1b|admission-1b|summarize-1b|iclr-1b|recover-4b|admission-4b|summarize-4b|recover-1b-it|admission-1b-it|summarize-1b-it|scale-4b|scale-12b}" >&2
    exit 2
    ;;
esac

