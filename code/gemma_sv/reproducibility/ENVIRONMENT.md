# Reference environment

The reported Apple-silicon runs used Python 3.11.14 on macOS 26.5. The
following direct package versions were recorded from the reference environment:

```text
numpy==2.4.6
scipy==1.17.1
cvxpy==1.9.2
pytest==9.1.1
torch==2.12.1
transformers==5.12.1
accelerate==1.14.0
datasets==5.0.0
peft==0.19.1
sentencepiece==0.2.1
huggingface-hub==1.20.1
mlx==0.32.0
mlx-lm==0.31.3
fastapi==0.139.0
uvicorn==0.51.0
httpx==0.28.1
```

The optional public-RAG tier additionally pins
`llama-index-core==0.14.23` and
`llama-index-embeddings-huggingface==0.7.0`. Retrieval uses
`BAAI/bge-small-en-v1.5` at
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; 2WikiMultiHopQA is loaded from
`framolfese/2WikiMultihopQA` at
`a5d42f3b40d57a8c59fa10b2ac0c1829e4f73aba`. Each live run still writes a full
resolved `pip freeze --all` beside its report.
The controlled fallback follows NVIDIA RULER commit
`ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13`; its synthetic generator is
dependency-free and records the same Gemma tokenizer revision used for exact
owned-token mapping.
The exploratory natural-QA v2 uses
`bdsaglam/musique@22873a405dd809893b22ada0b499299fb612d2df`
(`musique_ans_v1.0_dev.jsonl`, SHA-256
`15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`).

These versions document the environment; the tiered requirement files remain
the supported installation interface. To capture a complete transitive lock
from a reproduction environment, run:

```bash
python -m pip freeze --all > environment-lock.txt
python -VV >> environment-lock.txt
```

The Gemma model identifiers are `google/gemma-3-1b-pt`,
`google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, and
`google/gemma-3-4b-it`. Protocol files record their applicable revisions.
The reference Gemma-3-1B cache resolved to
`fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`. Reproduction reports should
retain the resolved model revision, adapter hash, package lock, device, dtype,
and framework versions. Model weights and adapters are not distributed.

The version list above records the original host. FastAPI, Uvicorn and HTTPX
were used by a separate service and are not installed by the offline result
requirements in this trimmed repository.
