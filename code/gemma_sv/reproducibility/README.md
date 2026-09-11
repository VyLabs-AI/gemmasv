# Reproducibility guide

Run commands from the repository root. Generated checkpoints, logs, and figures
go under `outputs/`; they are not sources of authority and are not part of the
anonymous release unless explicitly listed.

## Tier 1: data-free contract checks

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -r gemma_sv/requirements-core.txt
bash gemma_sv/reproducibility/run_quick.sh
```

This checks:

- exact-deletion numerical contracts;
- selected-span/token alignment;
- request-scoped gate-state restoration;
- session isolation and expiry;
- recorded API flow and static replay integrity.

It does not download Gemma or claim to reproduce paper-scale model results.

### Optional Apple/Kimi contracts

Install `requirements-apple.txt`, then run:

```bash
bash gemma_sv/reproducibility/run_kimi_quick.sh
```

This data-free suite checks native Kimi cache replay, record boundaries,
full-transition receipt transport, changed-suffix forcing decomposition, decay
controls, privacy-safe probes, and the sparse-checkpoint trade-off calculator.
It uses a tiny random Kimi-compatible model and does not download the 48B
weights.

Before a 48B run, use a local hash-verified snapshot and require the lazy-load
smoke:

```bash
HF_HUB_OFFLINE=1 python -m kimi_sv.smoke_model \
  --model /local/Kimi-Linear-48B-A3B-Instruct-8bit/snapshot
```

Then run `kimi_sv.eval_separability` with `--lazy-load` and a portable
`--report-model-label`; eager parameter materialization is not supported for
this reference run.

### Qwen3.5 cross-family replay audit

The staged three-scenario Qwen validation protocol is frozen before its
live-kernel conformance rerun in
`gemma_sv/benchmarks/qwen35_replay_v3.json`. The first scenario had already run
before the other two were added, so this is not an independent replication.
First verify complete cache
snapshot/restore, schedule-matched replay, and DeltaNet recurrence algebra on a
tiny random model:

```bash
python -m pytest tests/test_qwen_replay_audit.py -q
python -m gemma_sv.qwen_replay_audit \
  --manifest gemma_sv/benchmarks/qwen35_replay_v3.json \
  --tiny --device cpu --overwrite \
  --out outputs/qwen35_replay/tiny_audit_v3.json
```

Then run the pinned 4B checkpoint:

```bash
caffeinate -dims python -m gemma_sv.qwen_replay_audit \
  --manifest gemma_sv/benchmarks/qwen35_replay_v3.json \
  --device mps --overwrite \
  --out outputs/qwen35_replay/audit_v3.json
python -m gemma_sv.publish_qwen_replay \
  --manifest gemma_sv/benchmarks/qwen35_replay_v3.json \
  --report outputs/qwen35_replay/audit_v3.json \
  --out gemma_sv/benchmarks/qwen35_replay_result_v3.json
```

The audit uses direct text forwards. Fresh omission and checkpoint replay use
the same suffix segmentation; present additionally ingests the victim, and
repeat independently reruns omission. Replay covers all 24 convolution arrays,
all 24 recurrent arrays, all 16 full-attention K/V arrays, 82 state flags,
cache length, wrapper RoPE state when active, and final logits. Receipt
transport is claimed only for the selected DeltaNet recurrence, whose
sequential implementation is checked against captured live-kernel final states.
Different tokenizations or chunk schedules remain separate controls. All three
victims and all six suffix paths pass the v3 validation checks.

The exact direct-package versions from the reference host and model identifiers
are recorded in `gemma_sv/reproducibility/ENVIRONMENT.md`. A reproducer should
also retain a full `pip freeze --all`, resolved model revisions, and adapter
hashes with each paper-scale report.

## Tier 2: random-model integration

Install `requirements-full.txt`, then run:

```bash
bash gemma_sv/reproducibility/run_headline.sh --quick
```

This runs the Gemma graft smoke test and the complete demo test suite. The smoke
model is randomly initialized; it verifies wiring, not language quality.

## Tier 3: training-free Gemma headline results

The primary Gemma configuration is frozen in
`gemma_sv/benchmarks/mass_preserving_boundary_v1.json`: `nu=0.7`, chunk
`128`, mass-preserving prefix readout, one `C=1/(nu*n_prefix)` per boundary,
deterministic bandwidth sampling, and solver seed `0`. It uses no adapter or
training data. The untouched confirmation cohort is
`whole_record_confirm_v2.json`. The historical source-free result is
`iclr_mass_preserving_boundary_v2.json`; the denominator-corrected,
hash-validated aggregate is `iclr_mass_preserving_boundary_v3.json`, rebuilt
with `python -m gemma_sv.publish_boundary_evidence_v3`.

Run admission with `eval_whole_record_unlearning --admission-only --lora none
--nu 0.7 --solver-seed 0 --preserve-prefix-mass --per-boundary-box`, using
window/prefix-filler pairs `(512,8)` at 1B and `(1024,11)` at 4B. Run paired
quality with `gemma_sv.eval_training_free_quality --blocks 400 --group-size
20`. The final 4B graft admits `6/8`, exactly matching base;
1B admits `3/8` versus base `7/8`. All records are fixed-C_b feasible.
WikiText costs are `+1.85%` at 4B and `+3.70%` at 1B. All-record certificate
sweeps contain 32 probes per scale: maximum decrement/refit KL is `6.48e-11`
at 4B and `1.25e-10` at 1B. Fallback counts are `2,020/4,320` and
`396/1,024` over all enumerated gates; restricting the denominator to affected
decrement attempts gives `2,020/3,040` (`66.4%`) and `396/640` (`61.9%`),
respectively.
The attempted-only rates are cost-path diagnostics. Stale all-enumerated-gate
fractions and the earlier `66.5%` rounding are rejected by:

```bash
python -m pytest tests/test_boundary_evidence_publication_v3.py -q
```

`corrected_audit_cost_v1.json` records already-constructed probe-state scoring
cost (`561.5 s` at 1B; `2,354.6 s` at 4B, 32 probes each). It excludes state
construction and full repack; no corrected Gemma speedup is claimed.

The predeclared 12B-PT extension is frozen in
`training_free_second_checkpoint_v1.json` and published in
`training_free_second_checkpoint_result_v1.json`. On the same eight records,
base/graft admission is `6/8 → 5/8`, all eight remain fixed-C_b-feasible, and
the identical 400 token blocks give `22.20 → 24.80` perplexity (`+11.699%`).
On this fixed packing, ungrafted 12B is `3.0%` worse than ungrafted 4B
(`22.20` versus `21.56`). A second loss implementation reproduces 12B on the
same token tensor; alternative packings/backends/corpora were not evaluated,
and the paired `+11.699%` cost remains within-checkpoint.
This run covers admission and quality only; it establishes no monotone scaling
trend and no 12B deletion certificate.
`training_free_second_checkpoint_provenance_v1.json` binds the pre-run
source-state capture and clarifies the fallback-policy wording without
rewriting the hashed predeclaration; no fallback executed.

The corrected 4B behavioral suite is frozen in
`boundary_attack_suite_v1.json`. `run_boundary_attack_suite` produces
200-sample whole-record Leak@k shards; `eval_boundary_attacks` runs
manifest-aware elicitation, LiRA, and relearning. Masked-refit and never-stored
each leak on `3/18` field queries at 200 samples (`16.7%`, the cohort rate); a
separate broad cohort admits `16/20` and gives masked-refit/full-repack policy
LiRA AUC `0.517` (TPR `1.37%` at `1%` FPR). Elicitation and relearning stay
near the never-stored floor. The qualitative panel binds one field's frozen
counts (`9/200` present, `10/200` prompt-only, `0/200` edited and never stored)
to a separate 24-sample replay; it does not treat that query as the cohort
rate. No six-record attack invokes full repack. Broad LiRA routes `16/512`
masked-refit test deletions (`3.1%`) to full repack under the fixed-C_b
precheck; the reported AUC includes them. Changing all 16 scores arbitrarily
can move aggregate AUC by at most `0.031`.

Deletion prechecks every affected boundary. A deleted fraction above
`1-nu=0.3` routes to full repack and is not certified as an incremental
decrement. On identical 4B keys, a frozen ν sweep raises mean support fraction
`30.7% → 50.5% → 70.3%` and reverse fallback `0% → 7% → 59%`, confirming
the retention-coverage versus incremental-headroom trade-off.

## Legacy: recovered 1B diagnostics

The reference path needs:

- accepted access to `google/gemma-3-1b-pt`;
- the recovered adapter at
  `outputs/gemma_sv_distill/lora_adapter/adapter_model.safetensors`;
- sufficient RAM for float64 certificate runs.

```bash
bash gemma_sv/reproducibility/run_headline.sh --paper
```

This runs the output-level certificate, behavioral efficacy/specificity,
adversarial elicitation, relearning, and LiRA evaluations using their reported
default scales. It can take hours. Inspect `PROVENANCE.md` before comparing
outputs with manuscript numbers.

### Three-seed recovery and matched deletion audit

The ICLR multiseed path pins the Gemma, FineWeb-Edu, and WikiText revisions,
seeds Python/NumPy/Torch and LoRA initialization, and fingerprints every
stage-2 batch. Each control skips the recovery calibration plus 2,000 stage-1
batches, then refuses to complete unless its 6,000-batch fingerprint exactly
matches the paired recovery:

```bash
caffeinate -dims bash gemma_sv/reproducibility/run_models.sh iclr-1b
```

The default `SEEDS="0 1 2"` run uses 400 WikiText blocks and writes
seed-scoped checkpoints and reports under `outputs/gemma_sv_multiseed/`.
Completed seeds are skipped safely on restart. On the reference M3 Ultra, the
three recovery/control pairs require roughly 25 accelerator-hours before
utility and deletion evaluation. Stage 1 needs
`PYTORCH_ENABLE_MPS_FALLBACK=1` because PyTorch does not implement
`cdist` backward natively on MPS; the orchestrator sets it explicitly.
The final summarizer writes the full local report to
`outputs/gemma_sv_multiseed/summary.json` and a compact, source-free public
summary to `gemma_sv/benchmarks/iclr_multiseed_v1.json`.

The predeclared five-seed extension retains the original \(0,1,2\) pairs and
adds seeds \(3,4\) at both 1B and 4B. Both new seeds remain in the analysis
regardless of cost, admission, loss trajectory, or interval direction:

```bash
SEEDS="3 4" caffeinate -dims \
  bash gemma_sv/reproducibility/run_models.sh recover-1b
SEEDS="3 4" MAX_PARALLEL_4B=1 caffeinate -dims \
  bash gemma_sv/reproducibility/run_models.sh recover-4b
```

Run the two scales sequentially. Completed recovery/control arms are skipped,
but an interrupted arm restarts from batch zero: optimizer, RNG, and data
position are not checkpointed mid-stage. The completed rerun must reproduce
the full stream fingerprint before aggregation.

The completed reference run gives recovered/control perplexity
`20.50 [18.23, 22.77]` versus `18.91 [18.32, 19.49]` and paired cost
`+8.42% [-1.63%, 18.47%]` (mean and 95% Student-t CI across seeds).
Across the eight-record prefill audit, exact/refit KL has floor-safe geometric
mean `5.86e-13 [1.12e-13, 3.07e-12]` and decay/refit is `1.70e-2`. The recorded
proxy/exact value `4.31e-3` used different bandwidth objectives and is retained
only as a legacy diagnostic; the corrected evaluator gives both paths the
prefill-frozen `SVDecodeSession.kpar` and `box_C`. The unchanged joint
target-plus-retained admission is `1/24`; target-only is `8/24`, deleted fields
are `44/72`, retained-neighbor availability is `8/24`, and fixed-C feasibility
is `24/24` contexts (`120/120` boundaries). Thus the matched baseline matrix is
reported as a mechanism diagnostic rather than efficacy evidence. The executed
exact path used refit fallback for `248/1,920` head-gates; its plotted
approximately 47-second update is shared decrement-plus-refit audit time, not
pure decrement latency.
Seed 2 recorded one transient stage-1 loss spike at logged step 500
(`3.61e6`); the next logged value returned to `0.0097`, all 6,000 stage-2
updates were stable, and the run is retained rather than silently replaced.

Individual resumable phases are:

```bash
bash gemma_sv/reproducibility/run_models.sh recover-1b
bash gemma_sv/reproducibility/run_models.sh utility-1b
bash gemma_sv/reproducibility/run_models.sh certificates-1b
bash gemma_sv/reproducibility/run_models.sh deletion-audit-1b
bash gemma_sv/reproducibility/run_models.sh summarize-1b
```

The deletion audit attempts all eight predeclared synthetic records, retains
rejection reasons, and evaluates the same tokenized context with full
repack/re-prefill, exact decrement, fixed-$C$ refit, the FP32 proxy,
cache-only delete-and-shift, coefficient decay, and faithful four-shot ICUL.
It reports deleted-record quality, retained-record drift, output KL, latency,
and deduplicated tensor storage together. A separate float64 runtime performs
the prefill-once exact/refit/proxy precision audit. KVEraser is recorded as an
excluded method rather than assigned a cross-model number: its released
Qwen3-8B/CUDA eraser does not support the grafted `SVDecodeSession` or the
benchmark's disjoint repeated spans.

After objective alignment, run the locked residual-qualified proxy frontier:

```bash
python -m gemma_sv.eval_proxy_precision_sweep \
  --adapters \
outputs/gemma_sv_multiseed/seed-0/recovered/lora_adapter,\
outputs/gemma_sv_multiseed/seed-1/recovered/lora_adapter,\
outputs/gemma_sv_multiseed/seed-2/recovered/lora_adapter \
  --devices cpu \
  --out outputs/gemma_sv_proxy_precision/frontier.json
```

Calibration uses manifest indices `{0,2,4,6}` across all three adapters and
sweeps the versioned FISTA, partition-cutoff, and residual-cutoff grids. The
fastest nontrivial policy (at least one proxy-qualified context) with zero
calibration contexts above
`KL(refit64 || hybrid) = 1e-3` is evaluated once on locked indices
`{1,3,5,7}`. If no policy passes, the report records a negative frontier and
does not run validation. This runner reports feasibility, KKT,
projected-gradient, Frank--Wolfe duality-gap, latency, storage, peak-memory,
retained-drift, and fallback diagnostics; it never labels the approximation a
certificate. For deletion inference it evaluates the feasible projected FISTA
iterate directly; the ridge-regularized differentiable KKT reconstruction used
during training is not substituted for the measured proxy solution. The MLX
power iteration is reset from the recorded context-level operation seed before
every policy, so solver-grid comparisons and reruns use identical initialization.
The completed reference sweep selected 160 iterations, partition cutoff
`3e-4`, and residual cutoff `3e-3`. Calibration had `1/12` fallback and max
hybrid/refit64 KL `4.27e-6`; locked validation had no `1e-3` violations,
`2/12` fallbacks, max KL `4.88e-6`, mean retained absolute drift `5.55e-4`
nats, and mean end-to-end time `17.98 s`. The compact source-free result is
`gemma_sv/benchmarks/proxy_frontier_v1.json`.

### Public 2WikiMultiHopQA RAG erasure benchmark

Install the optional local-only retrieval tier; it is deliberately excluded
from `requirements-core.txt`:

```bash
python -m pip install -r gemma_sv/requirements-rag.txt
```

Freeze the small manifest before loading Gemma:

```bash
python -m gemma_sv.rag_benchmark \
  --mode smoke --records 8 --candidate-limit 32 \
  --out outputs/gemma_sv_rag/2wiki_rag_erasure_smoke_v1.json
```

This uses pinned 2WikiMultiHopQA and `BAAI/bge-small-en-v1.5` revisions,
LlamaIndex `Document`/`SentenceSplitter`/`VectorStoreIndex`, stable IDs, fixed
`top_k`, and no hosted API. The versioned manifest contains source IDs, hashes,
retrieval scores/order, and exact Gemma-token ownership but no source text.
The builder also writes a complete resolved `environment-lock.txt` beside the
manifest. Candidate order and retrieval eligibility are fixed without Gemma
outputs.

After validating manifest rehydration and ownership, run one live-model record:

```bash
python -m gemma_sv.eval_context_erasure_qa \
  --manifest outputs/gemma_sv_rag/2wiki_rag_erasure_smoke_v1.json \
  --records 1 --warmup 0 --repeats 1 \
  --out outputs/gemma_sv_rag/context_erasure_smoke.json
```

The target admission gate requires the inserted distractor to change the
pre-deletion answer and literal full repack to restore the gold answer.
Retained-QA availability is a separate denominator. Scan all frozen gates
without running deletion methods:

```bash
python -m gemma_sv.eval_context_erasure_qa \
  --manifest outputs/gemma_sv_rag/2wiki_rag_erasure_smoke_v1.json \
  --admission-only --warmup 0 --repeats 1 \
  --out outputs/gemma_sv_rag/admission_scan.json
```

The result-bearing
`gemma_sv/benchmarks/ruler_context_erasure_v1.json` field
`manifest.natural_qa_trigger.joint_admission` records `0/8`, alongside `1/8`
target and `1/8` retained admission; its `manifest_integrity_sha256` binds the
result to the fixed protocol. The
one-record wiring run completed every deletion method and verified exact token
ownership, source immutability, fixed-C feasibility, and maximum
decrement/refit KL `7.74e-12`. Because natural-QA admission is weak, no
examples are replaced and no full 2Wiki method matrix is run.
The source-free frozen protocol/input manifest is versioned at
`gemma_sv/benchmarks/2wiki_rag_erasure_smoke_v1.json`; it contains no outcome
fields.

An exploratory MuSiQue v2 then tested a stronger construction on untouched
pinned data: a same-type donor answer replaces the gold span inside a real
supporting sentence, the conflicting sentence follows retrieved gold evidence,
and retention uses an unrelated record's original QA. Its strict admission
scan passed `0/8` target gates and `2/8` retained probes, so no full method
matrix was run and it is not a paper result. Reproduce with
`gemma_sv.musique_rag_benchmark` followed by
`gemma_sv.eval_musique_context_erasure --admission-only`.

The frozen natural-QA failure also triggers the controlled RULER multi-key
follow-up:

```bash
python -m gemma_sv.ruler_erasure_benchmark \
  --out outputs/gemma_sv_rag/ruler_multikey_erasure_v1.json
python -m gemma_sv.eval_ruler_context_erasure \
  --manifest outputs/gemma_sv_rag/ruler_multikey_erasure_v1.json \
  --admission-only --warmup 0 --repeats 1 \
  --out outputs/gemma_sv_rag/ruler_admission_scan.json
python -m gemma_sv.eval_ruler_context_erasure \
  --manifest outputs/gemma_sv_rag/ruler_multikey_erasure_v1.json \
  --warmup 0 --repeats 1 \
  --out outputs/gemma_sv_rag/ruler_context_erasure.json
```

At pinned RULER commit `ab17b785…ee13`, the manifest fixes eight 1,024-token
contexts with the official four-key/one-value/one-query complexity and one
declared conflicting owned needle. Target admission is `4/8`, retained
availability `8/8`, and joint admission `4/8`; all eight remain in the
population. The shared evaluator supports full repack, exact
decrement/fixed-C refit, feasible FP32 FISTA, cache delete/shift, decay, and
four-demonstration correction-style ICUL, with gold recovery, distractor
leakage, retained QA, output KL, synchronized latency, and deduplicated
storage. MuSiQue remains an optional held-out natural-QA replication.
The controlled manifest is versioned at
`gemma_sv/benchmarks/ruler_multikey_erasure_v1.json`.
The completed all-eight matrix is versioned at
`gemma_sv/benchmarks/ruler_context_erasure_v1.json`: exact/refit maximum KL is
`9.51e-12` with `44/640` disclosed refit fallbacks. Mean behavioral KL to
repack is `0.0144` for exact, `0.0146` for the FP32 proxy, `0.211` for cache
shift, `0.0225` for decay, and `1.434` for correction-style ICUL. The
single-repeat exact row (`59.12 s`) includes shared decrement/refit work; proxy
end-to-end time is `3.50 s`.

### Naturalistic longitudinal conversation deletion

The MemOps follow-up freezes 64 distinct `Remember` conversations before any
Gemma evaluation. Each context is a contiguous complete-session window with at
least 512 model tokens before the owned user/assistant exchange and 10,000
tokens after it. The paired retained fact comes from another exchange in the
same evidence segment. MemOps and its UltraChat carrier text are not
redistributed.

Build and validate the source-free manifest from pinned MemOps commit
`312af65e2c7b6d1b70f062ffa8b4cde32aaf6f35`:

```bash
python -m gemma_sv.memops_longitudinal \
  --source-root /path/to/MemOps \
  --records 64 \
  --out gemma_sv/benchmarks/memops_longitudinal_v1.json \
  --overwrite
```

Run a bounded development admission pilot before the frozen confirmation:

```bash
python -m gemma_sv.eval_memops_longitudinal \
  --source-root /path/to/MemOps \
  --admission-only --records 8 \
  --warmup 0 --repeats 1 \
  --out outputs/gemma_sv_memops/admission_pilot_8.json
```

After the prompt and gate contract remains unchanged, run all 64 records. The
runner writes after every record and supports `--resume`:

```bash
python -m gemma_sv.eval_memops_longitudinal \
  --source-root /path/to/MemOps \
  --warmup 0 --repeats 1 --resume \
  --out outputs/gemma_sv_memops/longitudinal_evaluation.json
python -m gemma_sv.publish_memops_summary \
  --report outputs/gemma_sv_memops/longitudinal_evaluation.json \
  --out gemma_sv/benchmarks/memops_longitudinal_result_v1.json
```

The references are intentionally distinct. The raw omitted repack is freshly
retokenized after removing the owned raw-text exchange, and behavioral metrics
use a fresh prefill of that stream. Exact decrement is certified only to the
fixed-\(C\) retained-key refit over already-contextualized keys. Token-row
deletion is diagnostic; it does not define the raw omitted reference. Publish
no compact result until every frozen record and method completes without source
text.

Run the prefill-once same-context bridge separately:

```bash
python -m gemma_sv.persistent_bridge_demo \
  --out outputs/gemma_sv_demo/persistent_state_bridge_v2.json
```

This ingests each memory once, applies one stored deletion, forks the edited
cache across probes, and reports decrement/refit KL, decrement/proxy KL, and a
shared elicitation prompt.

Probabilistic decoding and dependency-aware paired prompts are a separate,
especially expensive generation sweep:

```bash
python -m gemma_sv.eval_robust_unlearning \
  --mode both --targets 20 --paired-targets 20 --samples 200 \
  --lora outputs/gemma_sv_distill/lora_adapter
```

It reports the fixed recall-admission denominator, token ROUGE-L and exact
phrase as explicitly named Leak@k core metrics, and forget/retain scores on
TOFU-Pair. Use `--smoke` only to verify wiring; its two eight-token samples are
not benchmark evidence.

The implementation supports `--target-start`, `--paired-start`, and
`--conditions` so targets/conditions can be sharded into separate output files.
On MPS, batches above 16 are rejected because the Gemma output kernel becomes
invalid or severely degraded; the full 200-sample sweep should be sharded or
run on a validated CUDA host rather than launched as one oversized MPS batch.
Merge a complete shard grid with:

```bash
python -m gemma_sv.merge_robust_shards \
  outputs/gemma_sv_eval/robust_shard_*.json \
  --out outputs/gemma_sv_eval/robust_unlearning.json
```

The merger rejects mismatched sampling settings, conflicting rows, or targets
missing any condition present in the merged grid, then recomputes every
aggregate from per-sample scores.

For the reference Mac Studio, use the resumable orchestrator. It writes one
target/condition per JSON file, skips valid completed shards on restart, and
merges only after the full grid is present:

```bash
screen -dmS robust bash -c \
  'caffeinate -dims bash gemma_sv/reproducibility/run_robust_m3.sh --paper \
   >> outputs/gemma_sv_eval/robust_paper_run.log 2>&1'
```

Launch inside a detached `screen` (or `tmux`) session so editor or terminal
restarts cannot kill the multi-day tree; `caffeinate` keeps the machine awake.
Use `--pilot` for the small integration run. Interrupting the paper command
loses only the currently running shard; rerunning the same command resumes
from the completed shard files. A lock directory under the shard root refuses
concurrent orchestrators.

The evaluation probes with the question plus the answer stem and scores the
extracted secret span; naive full-question probes have no discriminative
power (the model reproduces the template and resamples the secret; see
`outputs/gemma_sv_eval/robust_naive_probe.json`). Admission gates require the
stored secret to be genuinely extractable, so set `LEAK_INDICES` and
`PAIRED_INDICES` to pre-screened admissible targets to avoid burning shard
time on targets every condition will reject; admission is still re-verified
inside each shard. The paper run used `LEAK_INDICES="1 2 3 4 5 8 11 14 16 17
18 20 21 22 23 24 27 29 30 31"` and `PAIRED_INDICES="4 5 6 8 19 25 30 31 36
45 67 70 74 78 81 82 83 89 92 94"` (first 20 admitted of 40 and 140 scanned).
Sampling uses cached prefill-and-decode by default (validated logit-equivalent;
`--no-kv-cache` restores full recompute). After the grid completes:

```bash
.venv311/bin/python -m gemma_sv.make_robust_figure
```

recomputes the bootstrap summary and the paper figure from the merged report.

### Atomic whole-record deletion

The record benchmark uses a versioned manifest fixed before execution. It
deletes every token owned by both copies of each three-field record, audits all
fields, and checks one unrelated retained neighbor:

```bash
.venv311/bin/python -m gemma_sv.eval_whole_record_unlearning \
  --samples 16 --k 1,2,4,8,16
.venv311/bin/python -m gemma_sv.make_whole_record_figure
.venv311/bin/python -m gemma_sv.certify_whole_record \
  --record-id case-zaffre
.venv311/bin/python -m gemma_sv.hero_demo
```

The manifest attempts eight records and keeps all rejection reasons; six pass
the all-field admission and fixed-$C$ feasibility gates. The record-scale
certificate deletes each record's coefficients together with a coupled block
decrement, post-verifies every affected head-gate, and falls back to exact
fixed-$C$ refit when a gate disagrees (44/320 hero gates; 12/272 for the Helios
record). Whole-record deletion is therefore exact and predominantly
incremental, with fallback counts reported rather than hidden.

For the admission-scaling study, use the true admission-only path and bind
model revision, adapter hash, manifest hash, packing geometry, gates, and seed
in every report. Ordinary-attention controls use the same manifest and
questions:

```bash
.venv311/bin/python -m gemma_sv.eval_whole_record_unlearning \
  --admission-only --model MODEL --model-revision REVISION \
  --lora ADAPTER --window WINDOW --prefix-fillers PREFIX \
  --seed SEED --out REPORT.json

.venv311/bin/python -m gemma_sv.eval_whole_record_unlearning \
  --admission-only --ungrafted --model MODEL --model-revision REVISION \
  --lora none --window WINDOW --prefix-fillers PREFIX \
  --seed 0 --out UNGRAFTED.json
```

Do not compare the three-seed prefill matrix's \(1/24\) denominator directly
with a single-checkpoint whole-record ratio. Aggregate only reports with
matching provenance using
`python -m gemma_sv.summarize_whole_record_multiseed`.

Credentialed MIMIC-IV-Ext-CDS validation is local and aggregate-only:

```bash
MIMIC_EXT_CDS_DIR=/local/credentialed/path \
  .venv311/bin/python -m gemma_sv.eval_mimic_whole_record \
  --records 16 --samples 32 --k 1,2,4,8,16
```

The command refuses to write inside the source directory and persists no
source text, identifiers, generations, or per-record rows. Public artifacts
remain fully synthetic.

## Tier 4: utility and scaling

```bash
bash gemma_sv/reproducibility/run_models.sh utility-1b
SEEDS="0 1 2" MAX_PARALLEL_4B=2 \
  bash gemma_sv/reproducibility/run_models.sh recover-4b
SEEDS="0 1 2" bash gemma_sv/reproducibility/run_models.sh summarize-4b
bash gemma_sv/reproducibility/run_models.sh scale-12b
```

The corrected 4B mode trains one recovered model and one exact-stream matched
control per seed, pins the 4B model revision, records stream fingerprints, and
rejects mismatched pairs. Seed pipelines are independent and can run in
parallel; recovery precedes its control within each seed. `scale-4b` remains a
legacy single-adapter evaluation command and is not the multiseed authority.
These large-model runs require model access and substantial accelerator time.

## Demo

Recorded reviewer path:

```bash
bash gemma_sv/reproducibility/run_demo.sh replay
```

Live M3 path:

```bash
bash gemma_sv/reproducibility/run_demo.sh live
```

The script starts the API only. Serve `gemma_sv/demo_site` separately on port
8000 as printed by the script. Never enter real credentials, patient data, or
private records.

The visible sample exchanges are synthetic padding, not benchmark evidence.
The two shipped presets are guided mechanism demonstrations. Quantitative
claims come from the paper evaluation scripts.

### Demo-only natural-language resolver

The resolver is server-side and opt-in. Disabled mode is the default and makes
no provider request. A live deployment additionally needs the maintained Google
Gen AI SDK:

```bash
python -m pip install "google-genai>=2.19,<3"
HERO_RESOLVER_MODE=gemini \
HERO_RESOLVER_MODEL=gemini-3.6-flash \
GEMINI_API_KEY=... \
  bash gemma_sv/reproducibility/run_demo.sh replay
```

`GOOGLE_API_KEY` is the fallback credential name. The key is retained only in
server settings; `/api/v1/config` publishes the resolver mode, model label, and
confirmation requirement but never the credential. Candidate prompts contain
only opaque record IDs plus user-visible labels and summaries. Private
ownership receipts, token/character ranges, complete unrelated history, and
certificate state are not sent to the resolver.

As of 2026-08-24, the official Gemini model documentation lists
`gemini-3.6-flash` as the current stable Flash model and does not list the
specification's requested `gemini-3.7-flash` identifier. The repository's
demo requirements also do not install `google-genai`. For that reason the
server defaults to the documented `gemini-3.6-flash`; set
`HERO_RESOLVER_MODEL` only to an identifier verified for the deployment's
account and API.

Recorded mode accepts one committed structured outcome through
`HERO_RESOLVER_RECORDED_JSON`, validates it against the current session catalog,
and performs no network request. Static presentations should load that recorded
outcome from a shipped local asset, label it `recorded Gemini resolver outcome`,
and require a confirmation click before advancing to the recorded deletion.
They must not call the resolver endpoint.

The server flow is:

1. `POST /api/v1/sessions/{session_id}/memory/resolve` creates a short-lived,
   catalog-hash-bound proposal;
2. the browser displays the returned complete owned exchange;
3. `POST /api/v1/sessions/{session_id}/memory/confirm` with `confirmed=true`
   consumes the proposal and routes its exact ID through the existing deletion
   engine.

Run the data-free resolver contracts with:

```bash
python -m pytest tests/test_demo_resolver.py tests/test_demo_api.py -q
```

## Paper

The submitted figure files are tracked, so compiling the manuscript does not
require rerunning experiments:

```bash
cd gemma_sv/paper
tectonic main.tex
```

`gather_figs.py` is only for refreshing tracked figures from verified
`outputs/gemma_sv_eval/` artifacts.
The admission/cost boundary figure is regenerated from tracked source-free
aggregates with:

```bash
python -m gemma_sv.make_audit_boundaries_figure
```

See:

- `PROVENANCE.md` for claim-to-command mapping;
- `HARDWARE.md` for runtime and numerical boundaries;
- `../SUBMISSION_HANDOFF.md` for anonymous ICLR packaging.

