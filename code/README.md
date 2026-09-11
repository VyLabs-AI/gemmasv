# Gemma-SV replication code

Public replication snapshot for *Can an AI Assistant Really Forget? Auditable
Deletion from Addressable Memory*.

## Quick start

```bash
python3 -m venv .venv311
source .venv311/bin/activate
python -m pip install -r requirements-core.txt
bash gemma_sv/reproducibility/run_quick.sh
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

This snapshot is not release-complete until the source-free decoded, Luna,
human-validation, and adjudicated-census artifacts plus their macro generators
are copied from the reviewed parent implementation and validated
byte-for-byte.

See `../reproducibility/PROVENANCE.md` for the claim-to-command map.
