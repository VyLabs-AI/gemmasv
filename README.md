# GemmaSV: auditable deletion from addressable memory

Code and evidence accompanying **Can an AI Assistant Really Forget? Auditable Deletion from Addressable Memory**, by Vishwajith Ramesh, Vy Labs, Inc. [Preprint](https://arxiv.org/abs/2607.27539).

GemmaSV installs a support-vector gate in frozen Gemma 3 and records which stored rows belong to an exchange. Deletion excludes the exchange's rows and checks the gate against a refit on retained keys. Rebuilding from the literal retained conversation is a separate reference: surviving contextualized rows can retain influence from the deleted exchange.

At the tested 4B configuration, admission matched the base model on 6 of 8 records with a 1.851% paired perplexity cost. The same configuration lost admission at 1B and 12B. The behavioral tests show less disclosure than prompt-only suppression and residual membership signal. The journal cost comparison found no update-speed advantage for the current FP32 proxy; retained-answer matching was below half in every condition. This is a scoped memory-editing study, not a guarantee of physical erasure, general capability preservation, or raw-history equivalence.

## Contents

- `code/`: the tracked implementation and contract-test snapshot maintained with the public GemmaSV v3 paper. The included Kimi, support-vector and synthetic-demo support modules preserve existing dependencies and historical tests; their presence does not make their separate results GemmaSV claims.
- `journal_evidence/`: the frozen 8-record/6-arm cost comparison and post hoc retained-answer analysis over 96 histories in 32 clusters. It includes measured numerical results, source snapshots, protocols, analysis code, manifests and verification scripts.
- `paper/`: editable public manuscript sources corresponding to the September 11, 2026 arXiv replacement.
- `journal_manuscript/`: the Neurocomputing manuscript, including both additional journal studies.
- `RELEASE_MANIFEST.json`: file hashes and provenance for this reviewed release snapshot.

The completed decoded-study pipeline and exact historical runtime bindings are documented in `code/REPRODUCE_DECODED.md`.

The code directory is a documented paper snapshot. It is not a mirror of ongoing experiments or the separate evolving assistant-demo application. The journal evidence contains the exact selected execution snapshot used by its additional studies; that frozen code is not silently overlaid onto the earlier implementation.

## Quick checks without model inference

Use Python 3.11 or newer. The following commands require NumPy, SciPy, CVXPY and pytest; the full contract environment is in `code/requirements-core.txt`.

```sh
python journal_evidence/verify_bundle.py
cd code
python -m pytest tests/test_boundary_evidence_publication_v3.py tests/test_publish_longmemeval_chat_result.py tests/test_exactness.py tests/test_build_longmemeval_chat_v3_paper_macros.py tests/test_build_longmemeval_chat_suffix_disclosure_crosstab_v2.py -q
```

Reproduce the journal numerical analyses from the saved measured results (NumPy only; run from `journal_evidence/`):

```sh
python -m journal_studies.gemmasv.analyze_results --plan outputs/journal_studies/gemmasv/analysis_plan_20260906_v1 --results outputs/journal_studies/gemmasv/main_20260906_v1/results.json --out reproduction_paired
python -m journal_studies.gemmasv.analyze_decoded_retained --out reproduction_retained
```

See `journal_evidence/README.md` for pinned model/dataset revisions and inference instructions. The measured journal route used Apple Silicon MPS plus MLX. Inference requires separately obtaining the original datasets and gated model weights under their upstream terms; no command here downloads them automatically. Repeating the full historical audit requires its original 529-file snapshot and caches, which are not all in this selected supplement. The released 110-file evidence bundle independently verifies its distributed files and reproduces the numerical analyses.

## Build the papers

Each paper directory is a self-contained TeX package with embedded figure PDFs, editable SVGs and a frozen bibliography. Compile `main.tex` with Tectonic, or XeLaTeX/BibTeX followed by two XeLaTeX passes. The arXiv manuscript and journal extension are distinct documents.

## Data and release boundaries

This repository excludes real source conversations, full generated response collections, token arrays, model weights, credentials and private review ledgers. Explicitly synthetic demonstration fixtures and the selected fictitious TOFU illustration remain labeled as such. Source-free flags and aggregates summarize measured model behavior; they are not new synthetic training data. Original benchmark inputs are obtained from TOFU and LongMemEval. The retained-answer study is a post hoc reanalysis using existing deterministic matching flags; it is not a semantic-correctness evaluation or an equivalence study.

Original code is supplied under Apache-2.0; third-party components retain their existing notices and licenses. See `THIRD_PARTY_NOTICES.md` and the journal evidence's license files. This is a clean release snapshot with no private repository history.
