# Completed decoded LongMemEval study

The repository includes the completed 32-cluster / 96-history cohort, response-generation v2 runner, deterministic v3 summary, matcher, model-judge protocols/runners, source-free human-validation artifacts, adjudicated v4 census, and suffix cross-tab. All 12 files fingerprinted by the response-generation protocol match their recorded SHA-256 values. These exact implementations reproduce the reported study; their fingerprints are preserved.

## Reproduce published values without model access

From `code/`:

```sh
python -m gemma_sv.build_longmemeval_chat_v3_paper_macros --validate-only
python -m gemma_sv.build_longmemeval_chat_suffix_disclosure_crosstab_v2 --check
python -m pytest tests/test_build_longmemeval_chat_v3_paper_macros.py tests/test_build_longmemeval_chat_suffix_disclosure_crosstab_v2.py -q
```

The tests reconstruct the committed numeric fixtures under `tests/expected/`, using the exact hash-bound decoded summary, immutable model-judge census and human-adjudicated derivative. They do not need raw conversations, response sets or a model.

## Inspect or rerun the pipeline

The entry points are `longmemeval_chat_cohort_v3.py`, `longmemeval_chat_response_generation_audit_v2.py`, `validate_longmemeval_chat_response_generation_audit_v2.py`, `summarize_longmemeval_chat_v3.py`, `longmemeval_chat_leakage_recall_openai_v2.py`, and `summarize_longmemeval_chat_leakage_recall_census_v3.py`. Each accepts `--help`; none is run during release verification. The inference runner requires an explicit locally obtained pinned LongMemEval oracle, the original Gemma-3-4B-IT checkpoint, and a supported accelerator environment. Its `--run` path requires the exact source-bearing-output acknowledgement printed by help.

The preserved authorization deliberately pins original implementation, OS/package environment and MPS availability, and checks committed inputs before live execution. It may reject a different machine or OS. Those checks preserve the recorded execution conditions; the run is not turnkey on an arbitrary machine. A new experiment must use a separately reviewed protocol and preserve its own provenance; it must not overwrite historical observations. Model weights, complete source conversations, all decoded responses, provider ledgers and private human-review packets are not distributed.

The data-free cohort/aggregation/response-runner contract suite is:

```sh
python -m pytest tests/test_longmemeval_chat_cohort_v3.py tests/test_summarize_longmemeval_chat_v3.py tests/test_longmemeval_chat_response_generation_audit_v2.py -k 'not rebuild_is_value and not rehydration and not real_exact and not real_sample and not real_outputs' -q
```

Five omitted cases require original upstream sources or model/tokenizer access. The fake-runtime tests isolate accelerator RNG seeding and replay the recorded environment when checking immutable lock construction. They verify count/order accounting, strict provenance, corrupt-shard rejection, no-overwrite behavior and resumability; they do not verify a live GPU/model run.
