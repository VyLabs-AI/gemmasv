# Existing decoded histories: retained-answer correctness

This is post hoc reanalysis of existing decoded outputs. Aggregate retained correctness and source structure were inspected before this note; the analysis is not outcome-blind, newly held out, newly generated, or a replacement for locked continuous-utility studies.

All 96 histories in 32 target clusters remain in the denominator. Repeated decodes contribute one canonical result per history. Missing or unscorable responses count as incorrect in the operational endpoint and are counted separately.

| Existing endpoint | Condition | Retained matches / 96 | Scorable |
|---|---|---:|---:|
| deterministic_any | exact_decrement_or_refit_policy | 44/96 | 96/96 |
| deterministic_any | fresh_raw_omission | 46/96 | 96/96 |
| deterministic_any | present | 46/96 | 96/96 |
| legacy_registered_casefold_substring | exact_decrement_or_refit_policy | 38/96 | 96/96 |
| legacy_registered_casefold_substring | fresh_raw_omission | 39/96 | 96/96 |
| legacy_registered_casefold_substring | present | 39/96 | 96/96 |

Paired differences below are percentage points, left minus right. Intervals resample the 32 target clusters and preserve all three variants together.

| Endpoint | Left versus right | Difference (percentage points) | Cluster bootstrap 95% | Left-only / right-only matches | Unscorable pairs |
|---|---|---:|---:|---:|---:|
| deterministic_any | exact_decrement_or_refit_policy versus fresh_raw_omission | -2.08 | [-6.25, 2.08] | 2 / 4 | 0 |
| deterministic_any | exact_decrement_or_refit_policy versus present | -2.08 | [-8.33, 4.17] | 3 / 5 | 0 |
| deterministic_any | fresh_raw_omission versus present | 0.00 | [-6.25, 6.25] | 5 / 5 | 0 |
| legacy_registered_casefold_substring | exact_decrement_or_refit_policy versus fresh_raw_omission | -1.04 | [-4.17, 2.08] | 2 / 3 | 0 |
| legacy_registered_casefold_substring | exact_decrement_or_refit_policy versus present | -1.04 | [-5.21, 3.12] | 2 / 3 | 0 |
| legacy_registered_casefold_substring | fresh_raw_omission versus present | 0.00 | [-5.21, 5.21] | 4 / 4 | 0 |

No equivalence margin was specified. A small difference or an interval containing zero does not establish equivalence, noninferiority, preserved general capability, low decoded disclosure, or universal deletion. Low absolute retained correctness remains visible even when two methods agree.

The six intervals are descriptive and have no multiplicity correction. Agreement between responses is not correctness. These artifacts contain no forced-choice margin, gold-token rank, or continuation-KL measurement.
