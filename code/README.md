# Gemma-SV replication code

Public replication snapshot for *Can an AI Assistant Really Forget? Auditable
Deletion from Addressable Memory*.

## Quick start

```bash
python3 -m venv .venv311
source .venv311/bin/activate
python -m pip install -r requirements-core.txt
python -m pytest tests/test_boundary_evidence_publication_v3.py tests/test_publish_longmemeval_chat_result.py tests/test_exactness.py tests/test_build_longmemeval_chat_v3_paper_macros.py tests/test_build_longmemeval_chat_suffix_disclosure_crosstab_v2.py -q
```

The data-free test suite covers exact solver contracts and request-scoped
Gemma gate state. The snapshot retains historical Kimi utilities for provenance,
but the v3 manuscript cites the separate KDA paper rather than using them as
evidence. Live-model evaluations require access to released Gemma weights.

The paper consumes a completed decoded follow-up with `K=32` clusters and
`n=96` histories,
the immutable Luna matcher-negative census, completed blinded human validation,
the strict human-adjudicated v4 derivative, and the sealed
suffix/disclosure cross-tab.
The older 16-record response-generation v1 authorization remains a terminated
historical protocol and contributes no evidence to the completed follow-up.
Full source-bearing response sets, Luna ledgers, and the blinded human packet
remain local-only. The manuscript's selected fictitious TOFU completions are
represented here only by a source-free prompt/count/replay binding. The
snapshot also includes the source-free human lock chain, final counts,
adjudicated v4 census, and superseding suffix cross-tab.

The source-free decoded summary, immutable Luna statistics, human-validation
lock chain, adjudicated census, suffix cross-tab, and both macro generators are
included. Twelve publication tests validate the exact input hashes, recompute
the cross-tab, and reproduce the committed decoded and suffix TeX macros
byte-for-byte. Generator modules and numerical input artifacts were copied
unchanged from the reviewed parent; the two tests only adjust the paper path
for this standalone directory layout. Complete source-bearing responses and
private human-review packets remain excluded by design.

See `gemma_sv/reproducibility/PROVENANCE.md` for the claim-to-command map.
