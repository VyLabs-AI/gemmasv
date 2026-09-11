# GemmaSV replication code

This directory contains evaluation and result-reconstruction code for the
[evolving GemmaSV manuscript](https://arxiv.org/abs/2607.27539).
Run all Python module commands from this directory.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pytest tests/test_boundary_evidence_publication_v3.py tests/test_publish_longmemeval_chat_result.py tests/test_exactness.py tests/test_build_longmemeval_chat_v3_paper_macros.py tests/test_build_longmemeval_chat_suffix_disclosure_crosstab_v2.py -q
```

The result tests validate source-free aggregates and reconstruct the three
numeric fixtures in `tests/expected/` without compiling a manuscript.
`REPRODUCE_DECODED.md` documents the completed 32-cluster/96-history decoded
study, its source-free human-validation chain and exact historical runtime
bindings. The older response-generation v1 authorization was terminated and
contributes no outcome evidence. Raw responses, private judge ledgers and
human-review packets are excluded.

The modules under `gemma_sv/demo_server/` supply model runtime, boundary gates,
state and certificate contracts used by the experiment runners. Their names
are preserved to maintain historical fingerprints. A hosted demo service is
not included. `svattn/` retains only the transitive support-vector gate modules
needed by Gemma; `cp_svm/` supplies the underlying solvers.

Live-model reproduction requires separately obtained model and dataset assets
and the documented reference environment. The full and Apple requirement
files add model dependencies; offline checks use only the core requirements.
See `gemma_sv/reproducibility/PROVENANCE.md` for the claim-to-command map and
`../journal_evidence/README.md` for the completed cost and retained-answer studies.
