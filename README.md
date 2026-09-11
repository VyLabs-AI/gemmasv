# GemmaSV: auditable deletion from addressable memory

Code and measured evidence for **Can an AI Assistant Really Forget? Auditable
Deletion from Addressable Memory**, by Vishwajith Ramesh, Vy Labs, Inc.
The [arXiv record](https://arxiv.org/abs/2607.27539) is the single evolving
manuscript citation; its public version may lag a submitted replacement.

GemmaSV installs a support-vector gate in frozen Gemma 3 and records which
stored rows belong to an exchange. Deletion excludes the exchange's rows and
checks the gate against a refit on retained keys. Rebuilding from the literal
retained conversation is a separate reference: surviving contextualized rows
can retain influence from the deleted exchange.

At the tested 4B configuration, admission matched the base model on 6 of 8
records with a 1.851% paired perplexity cost. The same configuration lost
admission at 1B and 12B. Behavioral tests show less disclosure than prompt-only
suppression and residual membership signal. The completed cost comparison found
no update-speed advantage for the current FP32 proxy; retained-answer matching
was below half in every condition. The evidence does not establish physical
erasure, general capability preservation, or raw-history equivalence.

## Repository contents

- `code/`: Gemma evaluation and analysis pipelines, required support-vector
  solver modules, source-free benchmark protocols/results, and contract tests.
  Modules retained under `gemma_sv/demo_server/` implement the historical
  experiment runtime; the hosted website and API are excluded.
- `journal_evidence/`: the frozen 8-record/6-arm cost comparison and post hoc
  retained-answer reanalysis over 96 histories in 32 clusters, with exact
  execution snapshots, measured results, protocols and offline verification.
  This directory name records when the studies were prepared; both belong to
  the same evolving manuscript.
- `code/tests/expected/`: three numerical macro fixtures used to check that
  reported values are reconstructed exactly. No TeX installation is needed.
- `RELEASE_MANIFEST.json`: release file hashes and historical provenance.

The completed decoded-study pipeline and its 12 exact historical runtime
bindings are documented in `code/REPRODUCE_DECODED.md`. The frozen study
snapshot is retained separately because those experiments used that exact
implementation. Ongoing experiments are not incorporated into these results.

## Offline checks

Use Python 3.11 or newer and install `code/requirements-core.txt`. From the
repository root:

```sh
python journal_evidence/verify_bundle.py
cd code
python -m pytest tests/test_boundary_evidence_publication_v3.py tests/test_publish_longmemeval_chat_result.py tests/test_exactness.py tests/test_build_longmemeval_chat_v3_paper_macros.py tests/test_build_longmemeval_chat_suffix_disclosure_crosstab_v2.py -q
```

Recompute the completed study analyses from saved measured results with NumPy,
from `journal_evidence/`:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m journal_studies.gemmasv.analyze_results --plan outputs/journal_studies/gemmasv/analysis_plan_20260906_v1 --results outputs/journal_studies/gemmasv/main_20260906_v1/results.json --out ../reproduction_paired
PYTHONDONTWRITEBYTECODE=1 python -m journal_studies.gemmasv.analyze_decoded_retained --out ../reproduction_retained
```

See `code/REPRODUCE_DECODED.md` for 28 further offline decoded-pipeline contract
checks and `code/gemma_sv/reproducibility/PROVENANCE.md` for the claim-to-command
map. `VALIDATION.md` records the checks completed for this release.

## Model reruns and data boundaries

Inference requires independently obtaining the pinned upstream model weights,
original benchmark inputs, and a compatible accelerator environment. The
completed study route used Apple Silicon MPS plus MLX. The decoded runner
preserves its original implementation and environment checks and may reject a
different machine. See `journal_evidence/README.md` and the decoded guide for
these requirements. A full historical audit also requires its original
529-file snapshot and caches, which are not all distributed. The selected
110-file study bundle independently verifies its own payloads and reproduces
the saved-result analyses.

This repository excludes source conversation collections, full generated
response sets, token arrays, model weights, credentials and private review
ledgers. Source-free flags and aggregates summarize measured model behavior.
The retained-answer study uses existing deterministic matching flags; it does
not evaluate semantic correctness or demonstrate equivalence. Explicitly
synthetic fixtures supporting implemented state contracts remain labeled.

Original code is supplied under Apache-2.0; third-party material retains its
notices and licenses. See `THIRD_PARTY_NOTICES.md` and the study bundle licenses.
Manuscript compilation, Node figure tooling and unrelated model experiments
are outside this repository's current scope. Existing Git history is preserved.
