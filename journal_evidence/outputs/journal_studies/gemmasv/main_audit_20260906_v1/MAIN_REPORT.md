# GemmaSV completed cost and utility bridge

Status: **completed**. Audited 48/48 completed arms across all eight records. Context and probe token hashes were independently reconstructed from the pinned local tokenizer and dataset for all eight records.

This implementation showed no update-speed advantage over either rebuild comparator. The FP32 proxy's paired update ratio was 1.012 relative to rebuilding the graft and 35.79 relative to rebuilding the ungrafted base. Its update-plus-four-probe ratio was 1.080 and 4.43, respectively. These are geometric means of within-record ratios, not ratios of the table medians. The proxy's retained-target mean log-probability difference was -0.524 nats/token versus the base rebuild and +0.378 versus the graft rebuild; the base comparison includes architecture differences. The cache-shift diagnostic's retained-target difference was -5.045 nats/token versus graft rebuilding. All-boundary gate recomputation is an implementation cost, not a fundamental complexity bound.

**These costs measure deletion-state construction and four teacher-forced audit probes. They are not production request latency or exact-decrement timings.** Record lookup, tokenization, locating deletion positions, and building the literal edited token list occur outside the timer. Model loading and shared original prefill are separate. The FP32 masked-refit arm is a proxy; cache shifting is a diagnostic.

**The proxy implements logical exclusion, not physical erasure.** It forks the original persistent memory, retains copied token IDs and K/V arrays, applies a drop mask, and recomputes every eligible stored boundary/head in every graft layer. The implementation does not restrict gate recomputation to affected boundaries. It reuses contextualized keys and does not re-ingest the language-model input unless boundary feasibility triggers full rebuilding. The study retains original controls and does not evaluate physical erasure from storage.

The cohort reuses eight records from four source blocks. One warmup and three timing repetitions were used. The table reports the median across available per-record median times, with each arm's completion denominator shown; separate medians need not sum. No main-cohort claims are made from smoke measurements.

| Arm | Completed | Update (s) | Four-probe audit (s) | Update + audit (s) | Boundary full-rebuild fallbacks |
|---|---:|---:|---:|---:|---:|
| Base present control | 8/8 | 0.000 | 8.100 | 8.100 | 0 |
| Base literal rebuild | 8/8 | 0.701 | 8.080 | 8.770 | 0 |
| Graft present control | 8/8 | 0.000 | 9.041 | 9.041 | 0 |
| Graft literal rebuild | 8/8 | 24.882 | 8.905 | 33.134 | 0 |
| Graft FP32 masked refit | 8/8 | 25.141 | 11.904 | 36.886 | 0 |
| Graft cache-shift diagnostic | 8/8 | 0.051 | 9.045 | 9.098 | 0 |

Paired estimates below average records, with four-source-block bootstrap sensitivity intervals. A cost ratio above one means the first arm is slower; a positive probability difference means its target continuation is more likely. The small reused cohort and only four source clusters do not support calibrated population or significance claims.

| First arm versus second | Deleted target Δ LP/token | Retained target Δ LP/token | Update ratio | Update + audit ratio |
|---|---:|---:|---:|---:|
| Graft present control versus Base present control | -0.3893 [-0.4762, -0.3325] | -0.8175 [-1.6215, -0.1274] | unavailable | 1.1566 [1.1150, 1.1997] |
| Graft FP32 masked refit versus Graft literal rebuild | 0.0431 [0.0046, 0.0816] | 0.3785 [0.1798, 0.5772] | 1.0120 [0.9882, 1.0367] | 1.0800 [1.0537, 1.1082] |
| Graft cache-shift diagnostic versus Graft literal rebuild | -4.9887 [-5.4856, -4.3880] | -5.0455 [-5.5243, -4.5501] | 0.0021 [0.0020, 0.0021] | 0.2569 [0.2451, 0.2727] |
| Graft literal rebuild versus Base literal rebuild | -0.0525 [-0.0694, -0.0264] | -0.9024 [-1.7084, -0.3258] | 35.3637 [34.3253, 36.4335] | 4.0995 [3.7511, 4.3951] |
| Graft FP32 masked refit versus Base literal rebuild | -0.0093 [-0.0519, 0.0187] | -0.5240 [-1.1366, -0.0628] | 35.7866 [35.1790, 36.5365] | 4.4274 [4.1571, 4.6311] |
| Graft present control versus Graft literal rebuild | 3.4226 [2.5615, 4.2128] | 0.0826 [-0.1511, 0.4086] | unavailable | 0.2581 [0.2423, 0.2768] |
| Base present control versus Base literal rebuild | 3.7594 [2.8501, 4.5508] | -0.0023 [-0.0055, -0.0004] | unavailable | 0.9147 [0.9105, 0.9209] |

All individual probe total/per-token log probabilities, paired absolute costs, and record-bootstrap sensitivity are retained in the frozen analyzer's JSON/Markdown output. Incomplete comparisons have their full-cohort estimates and intervals withheld.

## Interpretation limits

- The fresh reference is a literal edited token sequence, shorter than the historical padded never-stored control. Results are not interchangeable with that earlier reference.
- Probabilities are teacher-forced target-continuation scores, not decoded leakage, semantic correctness, or general model quality. Within-architecture KL is full-vocabulary first-token `KL(fresh rebuild || arm)`; cross-architecture KL also includes architecture differences.
- Boundary-precheck rebuild fallback in this proxy is different from the historical exact-decrement head/boundary fallback rate. Neither a proxy speed result nor reference self-KL proves exact-solver acceleration or conformance.
- Per-probe querying includes state cloning, CPU float64 log-softmax, and sequential model steps, including a step after the final scored target token. The benchmark describes this audit implementation.
- Base and graft run sequentially; graft order rotates across records. Timing repetitions do not establish independent observations or remove host drift.
- The study cannot substantiate the manuscript claim that the exact audit is cheap enough for every deletion request. Practical deployment budgets require separate evidence.
