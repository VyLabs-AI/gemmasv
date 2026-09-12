# GemmaSV: auditable deletion from addressable memory

Code and evidence for the current version of **Can an AI Assistant Really Forget?
Auditable Deletion from Addressable Memory**, by Vishwajith Ramesh, Vy Labs, Inc.
[Read the paper on arXiv](https://arxiv.org/abs/2607.27539).

GemmaSV installs a support-vector gate in frozen Gemma 3 and records which
stored rows belong to an exchange. Deletion excludes the exchange's rows and
checks the gate against a refit on retained keys. Rebuilding from the literal
retained conversation is a separate reference: surviving contextualized rows
can retain influence from the deleted exchange.

At the tested 4B configuration, admission matched the base model on 6 of 8
records with a 1.851% paired perplexity cost. Behavioral tests show less
disclosure than prompt-only suppression and residual membership signal.
The same configuration lost admission at 1B and 12B. The completed cost
comparison found no update-speed advantage for the FP32 proxy; retained-answer
matching was below half in every condition. These results do not establish
physical erasure, general capability preservation, or raw-history equivalence.

## Reproduce the results

Use Python 3.11 or newer. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-core.txt
python journal_evidence/verify_bundle.py
bash code/gemma_sv/reproducibility/run_quick.sh
```

The offline checks verify the evidence bundle, exact solver comparisons,
reported numerical values and decoded-pipeline contracts. They use saved
measurements and synthetic fixtures; they do not run model inference.

Recompute the paired cost and retained-answer analyses from measured results:

```sh
cd journal_evidence
PYTHONDONTWRITEBYTECODE=1 python -m journal_studies.gemmasv.analyze_results --plan outputs/journal_studies/gemmasv/analysis_plan_20260906_v1 --results outputs/journal_studies/gemmasv/main_20260906_v1/results.json --out ../reproduction_paired
PYTHONDONTWRITEBYTECODE=1 python -m journal_studies.gemmasv.analyze_decoded_retained --out ../reproduction_retained
```

For model reruns, start with the [claim-to-command map](code/gemma_sv/reproducibility/PROVENANCE.md),
[decoded-study guide](code/REPRODUCE_DECODED.md), and
[cost and retained-answer study instructions](journal_evidence/README.md).
[VALIDATION.md](VALIDATION.md) records completed verification and its limits.

## What is included

- `code/`: Gemma evaluation and analysis pipelines, support-vector solvers,
  protocols, aggregate results and tests.
- `journal_evidence/`: the 8-record/6-arm cost comparison and retained-answer
  reanalysis over 96 histories in 32 clusters, including measured results,
  execution code, protocols and a payload verifier.
- `code/tests/expected/`: three data-only numerical fixtures; no TeX
  installation or manuscript compilation is needed.
- `RELEASE_MANIFEST.json`: file hashes and execution provenance.

Each study retains the exact implementation and configuration used to produce
its reported results. Runtime modules under `code/gemma_sv/demo_server/` are
experiment dependencies; their paths and fingerprints are preserved. Detailed
version information belongs to the protocols and manifest.

## Model and data requirements

Inference requires the pinned upstream model weights, original benchmark
inputs and a compatible accelerator environment. The model studies used
Apple Silicon MPS plus MLX. The decoded runner checks its recorded environment
and may reject a different machine. The full execution audit also requires
original caches and a 529-file source snapshot that are not all distributed;
the included 110-file study payload independently verifies itself and
reproduces its saved-result analyses.

The repository excludes source conversation collections, full generated
response sets, token arrays, model weights and private review ledgers.
Retained-answer matching uses deterministic flags, not semantic correctness.
Synthetic fixtures remain labeled. No manuscript build or hosted service is
needed to reproduce the supplied results.

Original code is Apache-2.0; third-party material retains its own licenses.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the study bundle notices.
