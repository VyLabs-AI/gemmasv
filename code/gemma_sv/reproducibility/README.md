# Reproduce GemmaSV results

Run commands from `code/`. Start with the [quick start](../../README.md),
[decoded-study guide](../../REPRODUCE_DECODED.md), and
[claim-to-command map](PROVENANCE.md). The [reference environment](ENVIRONMENT.md)
and [hardware notes](HARDWARE.md) record model requirements and numerical limits.

The commands below require independently obtained model/data assets. The
protocols pin the model revisions, configuration, cohorts and measured
implementations. Runtime modules retain their recorded names and hashes;
`demo_server` supplies experiment dependencies, with no hosted service included.

## Training-free admission, quality and deletion

The primary Gemma configuration is frozen in
`gemma_sv/benchmarks/mass_preserving_boundary_v1.json`: `nu=0.7`, chunk
`128`, mass-preserving prefix readout, one `C=1/(nu*n_prefix)` per boundary,
deterministic bandwidth sampling, and solver seed `0`. It uses no adapter or
training data. The untouched confirmation cohort is
`whole_record_confirm_v2.json`. The source-free result is
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

## Conversation and cost analyses

The [decoded guide](../../REPRODUCE_DECODED.md) covers the 32-cluster/96-history
LongMemEval pipeline and its source-free human-validation chain.
The [study bundle](../../../journal_evidence/README.md) covers the 8-record/6-arm
cost comparison and retained-answer reanalysis. Each contains the exact
execution code needed for its measured results.

Run `bash gemma_sv/reproducibility/run_quick.sh` for the supported offline
result and contract checks. With full model dependencies, `python -m
gemma_sv.smoke_test` checks random-model wiring without pretrained weights.
Numerical result fixtures live under `../../tests/expected/`; no manuscript
compilation is required.
