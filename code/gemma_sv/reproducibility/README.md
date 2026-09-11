# Reproduce GemmaSV results

Run commands from `code/`. Start with the offline checks in `../../README.md`
and the completed decoded pipeline in `../../REPRODUCE_DECODED.md`.
`PROVENANCE.md` maps measured claims to their source-free artifacts and original
commands. `ENVIRONMENT.md` records the historical reference environment.

The retained `demo_server` modules implement the historical model runtime and
its state contracts; they are dependencies of the experiment runners. This
release does not include a hosted assistant service or website. Random-model
integration uses `python -m gemma_sv.smoke_test` with full dependencies.

The detailed commands below require independently obtained model/data assets
and, where stated, historical adapters and raw local reports. They are not
part of the offline verification and do not guarantee an identical run on a
different accelerator or software environment.

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

## Completed conversation and cost analyses

Use `../../REPRODUCE_DECODED.md` for the completed decoded LongMemEval pipeline
and `../../../journal_evidence/README.md` for the 8-record/6-arm cost comparison
and 96-history retained-answer reanalysis. The original protocols distinguish
completed observations, calibration evidence and terminated attempts.

`PROVENANCE.md` gives the remaining claim-to-command map, including earlier
Gemma behavior, certificate and natural-QA controls. Numerical fixtures are
checked under `../../tests/expected/`; manuscript compilation is not part of
this repository.
