# Recorded LongMemEval forgetting replay

This is a presentation-safe replay of one predeclared, measured case. It makes
no API calls, loads no model, and does not rerun the roughly 20-hour
method matrix.

## Run the recorded walkthrough

From the repository root:

```bash
.venv311/bin/python -m http.server 8000 --directory gemma_sv/demo_site
```

Open:

```text
http://127.0.0.1:8000/longmemeval_forgetting_replay.html
```

Advance through:

1. exact public excerpts from separate target, retained-control, and gap examples
   selected and assembled by the benchmark;
2. recorded pre-forget recall;
3. the natural-language forget request;
4. a deterministic catalog-bound resolver proposal, explicitly labelled as a
   fixture with no provider call;
5. complete-exchange review and a dedicated confirmation click;
6. the out-of-band evaluation deletion action added by us;
7. recorded target re-query;
8. recorded retained-control query;
9. the exact-decrement/fixed-\(C\)-refit certificate.

Every model-facing stage displays numeric measurements rather than a fabricated
assistant response. The target, retained, and gap excerpts are not one organic
continuous chat. Each displayed excerpt is an exact complete source sentence or
line with its public source-turn hash and span; the official LongMemEval
questions are replayed by the evaluator.
The recorded resolver fixture is outside the certificate boundary and makes no
network or API request.

## Rebuild the final 16/16 replay

The public LongMemEval oracle is pinned by repository, revision, size, and
SHA-256. Pass a local copy with `--data-path`, or omit that option to resolve
the pinned public artifact through the Hugging Face cache:

```bash
PYTHONPATH=. .venv311/bin/python \
  -m gemma_sv.build_longmemeval_chat_forgetting_payload \
  --method-report outputs/gemma_sv_rag/longmemeval_chat_geometry_methods_finalized_v1.json \
  --compact-out paper_viz/payloads/longmemeval_chat_forgetting_compact_final.json \
  --payload-out gemma_sv/demo_site/assets/longmemeval_forgetting_final.json \
  --data-path /path/to/longmemeval_oracle.json

cd paper_viz
npm test
npm run render:longmemeval-forgetting
```

This regenerates:

- `paper_viz/payloads/longmemeval_chat_forgetting_compact_final.json`
- `gemma_sv/demo_site/assets/longmemeval_forgetting_final.json`
- `paper_viz/previews/longmemeval_chat_forgetting_final.{svg,pdf,png}`

The reusable dissertation-style paper caption is stored at
`paper_viz/payloads/longmemeval_chat_forgetting_caption.tex`.

To consume an already frozen compact report, rerun the builder with the same
`--method-report` plus
`--compact-input paper_viz/payloads/longmemeval_chat_forgetting_compact_final.json`.
The compact input is accepted only when it reproduces exactly from that bound
all16 report.

Final mode fails closed unless all 16 authorized slots are attempted and the
report is bound to the committed partial lock. The case remains the first
pre-method-scoring jointly admitted record; it is not reselected from method
outcomes. The replay says `16/16 complete`, while still making no cohort
aggregate claim. The earlier partial15 assets remain explicitly labeled
`INCOMPLETE CASE STUDY PREVIEW` for provenance and are not the default.

## Claim boundary

- Behavioral comparator: fresh raw round-omitted repack.
- Numerical certificate: exact deletion path versus fixed-\(C\) retained-key
  refit at the two registered first-token distributions.
- This case used fixed-\(C\) refit fallback for 522/560 attempted affected
  solves; 960 head-gates were enumerated.
- It is not a raw-repack state certificate, privacy/compliance guarantee, or
  official LongMemEval leaderboard result.
