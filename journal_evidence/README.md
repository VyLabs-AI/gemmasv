# GemmaSV: source-free journal study supplement

This includes the completed eight-record, six-arm TOFU cost/teacher-forced utility
bridge (48/48 arms), its frozen analysis plan and paired/source-block analysis,
the completed context-reconstruction artifact audit, and a separate post hoc
decoded-retained analysis of 96 existing histories in 32 clusters.

The bridge reuses an earlier cohort; it is not newly held out. The original
selection manifest describes its historical confirmation role, while the journal
protocol explicitly labels the reuse. FP32 masked refit is the measured proxy,
not timed exact decrement or physical KV erasure. Rebuild references delete the
literal token subsequence and are distinct from older padded never-stored
controls. The cache-shift arm is diagnostic only. Timing excludes model loading,
tokenization, lookup/selection/edit construction, and shared original prefill;
query time includes cloning, CPU float64 scoring, and the final unused decode
step. Shared-host timings and fixed architecture order are descriptive.

The retained reanalysis uses existing source-free correctness flags, not source
conversations, new generations, human semantic scoring, or an equivalence design.
All 96 histories and both existing scoring endpoints are retained. Its methods
note discloses that aggregates were inspected before the paired analysis.

## Contents and external requirements

Runnable Python modules preserve their repository-relative paths at the bundle
root. The runner and its 35 local import dependencies are byte-identical to the
main execution snapshot. Selected originals also remain under that run's
`source_snapshot/`. The original provenance lists all 529 captured files; the
other files are intentionally absent. Only the 36 Python modules in the import
closure plus protocol/manifest are distributed from that snapshot. Analyzer,
addendum and retained methods copies come from their separately frozen directories.

The numerical tables and both paired analyses can be reproduced without any
source dataset or model. Model inference requires the pinned external assets:

- `google/gemma-3-4b-pt`, revision `cc012e0a6d0787b4adcc0fa2c4da74402494554d`.
- `locuslab/TOFU`, revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`,
  `forget10` and `retain90` train splits.

The run provenance records Python 3.11.14, macOS arm64, Torch 2.12.1,
Transformers 5.12.1, MLX 0.32.0, Datasets 5.0.0, NumPy 2.4.6,
SciPy 1.17.1, Safetensors 0.8.0, and model/config/cache hashes. It explicitly
records that model weights were not rehashed. The tested inference route is
Torch MPS plus MLX on Apple Silicon. Upstream access is required to populate a
local model and dataset cache before the runner, which enforces offline use.
CVXPY and the numerical solver dependencies are also required for the graft.
Optional runtime imports such as PEFT are not needed by the frozen no-adapter
configuration; a full repository installation may be used for those optional
paths. This selected supplement is not an installation of all demo/server tools.

## Offline analysis commands

Run from this extracted directory with Python and NumPy. Each command requires a
new output directory and does not replace original observations. Timestamps and
absolute paths in new provenance will differ. `offline_verification.json` records
the supplement's own integrity and analysis checks, separately from the original
artifact audit.

```sh
python -m journal_studies.gemmasv.analyze_results   --plan outputs/journal_studies/gemmasv/analysis_plan_20260906_v1   --results outputs/journal_studies/gemmasv/main_20260906_v1/results.json   --out paired_reproduction
python -m journal_studies.gemmasv.analyze_decoded_retained --out retained_reproduction
```

For inference, after obtaining the external assets, use a new output directory:

```sh
python -m journal_studies.gemmasv.run_comparison   --model-snapshot /absolute/path/to/the/pinned/model/snapshot   --device mps --out main_reproduction
```

`summarize_main.py` and the completed `main_audit_20260906_v1/artifact_audit.json`
are included for transparency. That historical full audit requires all 529
original snapshot files and the original tokenizer/dataset cache paths; it cannot
be rerun unchanged from this selected supplement alone. Its recorded success is
not presented as a fresh full-snapshot audit here. Restore the complete execution
snapshot from the original repository/archive and the separately obtained cache
assets to replay that script. The supplement checks every distributed snapshot
file against the original provenance instead.

## Integrity and provenance

From this extracted directory, run `python verify_bundle.py` (standard library only).
`MANIFEST.json` records the SHA256, byte count, and original relative path of every
payload file. `MANIFEST.sha256` contains the same hashes in conventional format.
The manifests exclude themselves to avoid circular hashes; the ZIP has an external
SHA256 sidecar. Measured JSON and frozen code/protocols are copied byte for byte.
Absolute author-machine paths in original provenance are historical metadata, not
portable paths. New supporting files are explicitly marked in the manifest.

This is a supplement for the new journal studies, not a copy of all historical
experiments or the complete development repository. No model weights, dataset
source rows, private clinical data, cached conversations, generated responses,
token arrays from language-model corpora, credentials, or account files are
distributed. Synthetic solver inputs may be included. Existing source notices
remain in place; external models and datasets must be obtained under their own
access and license terms. The manifest and allowlist are an integrity/scope check,
not a claim of universal privacy or a new license grant.

The package is prepared locally for journal review. This does not establish a
public repository or DOI. Data-availability statements should say the supplement
provides the new studies' source-free measurements and analysis code, with
upstream data/models obtained separately; do not say all data or all historical
experiments are included or already publicly deposited.

## Existing license and attribution notices

The root `LICENSE` and `THIRD_PARTY_NOTICES.md` are copied unchanged from the
GemmaSV v3 release. The existing `gemma_sv/` license/notices remain in place.
`cp_svm/LICENSE` and `cp_svm/THIRD_PARTY_NOTICES.md` preserve the original
SV Attention v2 solver terms and Cauwenberghs reference-implementation
attribution/public-domain notice. These original notices also name style files,
figure tooling, datasets, and dependencies that are not distributed here;
their mention is not an inventory of this selected supplement. No original
MATLAB source or fixtures are included.
