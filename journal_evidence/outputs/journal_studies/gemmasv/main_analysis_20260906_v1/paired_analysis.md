# GemmaSV journal bridge: paired results

All log-probability deltas are left arm minus right arm; positive means greater target probability. All latency ratios are left arm divided by right arm; values above one mean the left arm is slower.

Only four source clusters exist, and their independence is not established. Both intervals are descriptive sensitivity summaries for this reused cohort, not calibrated population coverage or significance claims.

Teacher-forced target probabilities are disclosure and retained-utility proxies; these results do not measure decoded leakage or exact-decrement speed.

## graft_present versus base_present

Complete pairs: 8/8. Full main cohort: True.

| Record | Source block | Deleted target Δ LP/token | Retained target Δ LP/token | Update ratio | End-to-end ratio |
|---|---:|---:|---:|---:|---:|
| tofu-final-320-a | 320 | -0.3333 | -0.2899 | n/a | 1.140 |
| tofu-final-320-b | 320 | -0.3751 | -3.9216 | n/a | 1.113 |
| tofu-final-340-a | 340 | -0.5403 | -0.0104 | n/a | 1.088 |
| tofu-final-340-b | 340 | -0.1036 | -0.1616 | n/a | 1.119 |
| tofu-final-360-a | 360 | -0.4009 | -0.0711 | n/a | 1.171 |
| tofu-final-360-b | 360 | -0.3274 | -0.2666 | n/a | 1.216 |
| tofu-final-380-a | 380 | -0.4898 | -0.0520 | n/a | 1.181 |
| tofu-final-380-b | 380 | -0.5440 | -1.7671 | n/a | 1.233 |

| Cohort measure | Estimate | Record bootstrap 95% | Source-block bootstrap 95% |
|---|---:|---:|---:|
| deleted_mean_token_log_probability_delta | -0.3893 | [-0.4739, -0.2897] | [-0.4762, -0.3325] |
| retained_mean_token_log_probability_delta | -0.8175 | [-1.7741, -0.1120] | [-1.6215, -0.1274] |
| update_seconds_difference | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] |
| update_seconds_ratio | n/a | n/a | n/a |
| query_seconds_difference | 1.2115 | [0.9076, 1.5403] | [0.9162, 1.5068] |
| query_seconds_ratio | 1.1566 | [1.1249, 1.1899] | [1.1150, 1.1997] |
| end_to_end_seconds_difference | 1.2115 | [0.9076, 1.5403] | [0.9162, 1.5068] |
| end_to_end_seconds_ratio | 1.1566 | [1.1249, 1.1899] | [1.1150, 1.1997] |

## graft_masked_refit_proxy versus graft_full_repack

Complete pairs: 8/8. Full main cohort: True.

| Record | Source block | Deleted target Δ LP/token | Retained target Δ LP/token | Update ratio | End-to-end ratio |
|---|---:|---:|---:|---:|---:|
| tofu-final-320-a | 320 | 0.2398 | -0.0196 | 1.044 | 1.113 |
| tofu-final-320-b | 320 | -0.0462 | 1.1543 | 1.054 | 1.124 |
| tofu-final-340-a | 340 | 0.0281 | -0.0004 | 0.977 | 1.043 |
| tofu-final-340-b | 340 | -0.0577 | 1.1745 | 1.027 | 1.115 |
| tofu-final-360-a | 360 | 0.1196 | 0.1132 | 0.972 | 1.031 |
| tofu-final-360-b | 360 | 0.0133 | 0.1629 | 1.073 | 1.130 |
| tofu-final-380-a | 380 | 0.0097 | -0.0122 | 0.965 | 1.019 |
| tofu-final-380-b | 380 | 0.0385 | 0.4553 | 0.990 | 1.072 |

| Cohort measure | Estimate | Record bootstrap 95% | Source-block bootstrap 95% |
|---|---:|---:|---:|
| deleted_mean_token_log_probability_delta | 0.0431 | [-0.0120, 0.1104] | [0.0046, 0.0816] |
| retained_mean_token_log_probability_delta | 0.3785 | [0.0790, 0.7275] | [0.1798, 0.5772] |
| update_seconds_difference | 0.3046 | [-0.3606, 0.9956] | [-0.2956, 0.8966] |
| update_seconds_ratio | 1.0120 | [0.9858, 1.0398] | [0.9882, 1.0367] |
| query_seconds_difference | 2.4475 | [1.9776, 2.8460] | [2.2280, 2.7242] |
| query_seconds_ratio | 1.2772 | [1.2459, 1.3088] | [1.2446, 1.3108] |
| end_to_end_seconds_difference | 2.7449 | [1.6932, 3.7508] | [1.9106, 3.6282] |
| end_to_end_seconds_ratio | 1.0800 | [1.0498, 1.1092] | [1.0537, 1.1082] |

## graft_cache_delete_shift versus graft_full_repack

Complete pairs: 8/8. Full main cohort: True.

| Record | Source block | Deleted target Δ LP/token | Retained target Δ LP/token | Update ratio | End-to-end ratio |
|---|---:|---:|---:|---:|---:|
| tofu-final-320-a | 320 | -5.4013 | -5.8663 | 0.002 | 0.291 |
| tofu-final-320-b | 320 | -4.4722 | -2.7029 | 0.002 | 0.273 |
| tofu-final-340-a | 340 | -4.9345 | -6.6980 | 0.002 | 0.213 |
| tofu-final-340-b | 340 | -5.8887 | -4.7061 | 0.002 | 0.279 |
| tofu-final-360-a | 360 | -5.7601 | -2.7586 | 0.002 | 0.222 |
| tofu-final-360-b | 360 | -5.3589 | -6.9388 | 0.002 | 0.298 |
| tofu-final-380-a | 380 | -4.9653 | -3.9555 | 0.002 | 0.214 |
| tofu-final-380-b | 380 | -3.1284 | -6.7376 | 0.002 | 0.284 |

| Cohort measure | Estimate | Record bootstrap 95% | Source-block bootstrap 95% |
|---|---:|---:|---:|
| deleted_mean_token_log_probability_delta | -4.9887 | [-5.4986, -4.3648] | [-5.4856, -4.3880] |
| retained_mean_token_log_probability_delta | -5.0455 | [-6.1648, -3.8967] | [-5.5243, -4.5501] |
| update_seconds_difference | -24.7926 | [-25.3256, -24.3006] | [-25.4962, -24.0889] |
| update_seconds_ratio | 0.0021 | [0.0020, 0.0022] | [0.0020, 0.0021] |
| query_seconds_difference | -0.0320 | [-0.1840, 0.1117] | [-0.2026, 0.1255] |
| query_seconds_ratio | 0.9958 | [0.9765, 1.0128] | [0.9750, 1.0142] |
| end_to_end_seconds_difference | -24.8149 | [-25.3991, -24.2331] | [-25.6232, -24.0065] |
| end_to_end_seconds_ratio | 0.2569 | [0.2325, 0.2803] | [0.2451, 0.2727] |

## graft_full_repack versus base_full_repack

Complete pairs: 8/8. Full main cohort: True.

| Record | Source block | Deleted target Δ LP/token | Retained target Δ LP/token | Update ratio | End-to-end ratio |
|---|---:|---:|---:|---:|---:|
| tofu-final-320-a | 320 | -0.0142 | -0.2892 | 33.430 | 3.538 |
| tofu-final-320-b | 320 | -0.1291 | -3.8339 | 34.108 | 3.688 |
| tofu-final-340-a | 340 | -0.1008 | -0.0095 | 34.362 | 4.612 |
| tofu-final-340-b | 340 | -0.0157 | -1.2880 | 35.431 | 3.826 |
| tofu-final-360-a | 360 | -0.1368 | -0.1747 | 36.198 | 4.627 |
| tofu-final-360-b | 360 | 0.0024 | -0.2290 | 36.076 | 3.736 |
| tofu-final-380-a | 380 | 0.0152 | -0.0255 | 37.257 | 5.090 |
| tofu-final-380-b | 380 | -0.0409 | -1.3697 | 36.216 | 3.938 |

| Cohort measure | Estimate | Record bootstrap 95% | Source-block bootstrap 95% |
|---|---:|---:|---:|
| deleted_mean_token_log_probability_delta | -0.0525 | [-0.0934, -0.0153] | [-0.0694, -0.0264] |
| retained_mean_token_log_probability_delta | -0.9024 | [-1.8270, -0.2302] | [-1.7084, -0.3258] |
| update_seconds_difference | 24.1415 | [23.6461, 24.6781] | [23.4347, 24.8482] |
| update_seconds_ratio | 35.3637 | [34.5212, 36.2081] | [34.3253, 36.4335] |
| query_seconds_difference | 1.1510 | [0.8676, 1.4580] | [0.7885, 1.5135] |
| query_seconds_ratio | 1.1515 | [1.1143, 1.1899] | [1.1008, 1.2046] |
| end_to_end_seconds_difference | 25.2830 | [24.5297, 26.0424] | [24.2134, 26.3526] |
| end_to_end_seconds_ratio | 4.0995 | [3.7828, 4.4863] | [3.7511, 4.3951] |

## graft_masked_refit_proxy versus base_full_repack

Complete pairs: 8/8. Full main cohort: True.

| Record | Source block | Deleted target Δ LP/token | Retained target Δ LP/token | Update ratio | End-to-end ratio |
|---|---:|---:|---:|---:|---:|
| tofu-final-320-a | 320 | 0.2256 | -0.3088 | 34.896 | 3.937 |
| tofu-final-320-b | 320 | -0.1753 | -2.6797 | 35.938 | 4.146 |
| tofu-final-340-a | 340 | -0.0727 | -0.0099 | 33.559 | 4.808 |
| tofu-final-340-b | 340 | -0.0733 | -0.1135 | 36.392 | 4.267 |
| tofu-final-360-a | 360 | -0.0172 | -0.0615 | 35.197 | 4.770 |
| tofu-final-360-b | 360 | 0.0157 | -0.0661 | 38.725 | 4.221 |
| tofu-final-380-a | 380 | 0.0249 | -0.0377 | 35.942 | 5.187 |
| tofu-final-380-b | 380 | -0.0024 | -0.9144 | 35.853 | 4.223 |

| Cohort measure | Estimate | Record bootstrap 95% | Source-block bootstrap 95% |
|---|---:|---:|---:|
| deleted_mean_token_log_probability_delta | -0.0093 | [-0.0775, 0.0690] | [-0.0519, 0.0187] |
| retained_mean_token_log_probability_delta | -0.5240 | [-1.1779, -0.0727] | [-1.1366, -0.0628] |
| update_seconds_difference | 24.4460 | [23.8033, 25.1534] | [23.7192, 25.0464] |
| update_seconds_ratio | 35.7866 | [34.9090, 36.7747] | [35.1790, 36.5365] |
| query_seconds_difference | 3.5985 | [3.0277, 4.1610] | [3.3093, 3.8057] |
| query_seconds_ratio | 1.4707 | [1.4379, 1.5062] | [1.4428, 1.4991] |
| end_to_end_seconds_difference | 28.0279 | [26.9289, 29.1651] | [27.0185, 28.8324] |
| end_to_end_seconds_ratio | 4.4274 | [4.1813, 4.7182] | [4.1571, 4.6311] |

## graft_present versus graft_full_repack

Complete pairs: 8/8. Full main cohort: True.

| Record | Source block | Deleted target Δ LP/token | Retained target Δ LP/token | Update ratio | End-to-end ratio |
|---|---:|---:|---:|---:|---:|
| tofu-final-320-a | 320 | 5.1479 | -0.0007 | n/a | 0.299 |
| tofu-final-320-b | 320 | 3.9064 | -0.1018 | n/a | 0.278 |
| tofu-final-340-a | 340 | 4.5921 | -0.0010 | n/a | 0.212 |
| tofu-final-340-b | 340 | 2.8416 | 1.1248 | n/a | 0.271 |
| tofu-final-360-a | 360 | 1.9148 | 0.1019 | n/a | 0.227 |
| tofu-final-360-b | 360 | 2.4379 | -0.0377 | n/a | 0.303 |
| tofu-final-380-a | 380 | 2.3530 | -0.0269 | n/a | 0.207 |
| tofu-final-380-b | 380 | 4.1869 | -0.3975 | n/a | 0.290 |

| Cohort measure | Estimate | Record bootstrap 95% | Source-block bootstrap 95% |
|---|---:|---:|---:|
| deleted_mean_token_log_probability_delta | 3.4226 | [2.6602, 4.1940] | [2.5615, 4.2128] |
| retained_mean_token_log_probability_delta | 0.0826 | [-0.1496, 0.4169] | [-0.1511, 0.4086] |
| update_seconds_difference | -24.8438 | [-25.3786, -24.3502] | [-25.5494, -24.1382] |
| update_seconds_ratio | n/a | n/a | n/a |
| query_seconds_difference | 0.0759 | [-0.0441, 0.1839] | [-0.0131, 0.1817] |
| query_seconds_ratio | 1.0063 | [0.9924, 1.0186] | [0.9961, 1.0180] |
| end_to_end_seconds_difference | -24.7586 | [-25.3470, -24.2085] | [-25.5364, -23.9808] |
| end_to_end_seconds_ratio | 0.2581 | [0.2325, 0.2837] | [0.2423, 0.2768] |

## base_present versus base_full_repack

Complete pairs: 8/8. Full main cohort: True.

| Record | Source block | Deleted target Δ LP/token | Retained target Δ LP/token | Update ratio | End-to-end ratio |
|---|---:|---:|---:|---:|---:|
| tofu-final-320-a | 320 | 5.4670 | -0.0001 | n/a | 0.927 |
| tofu-final-320-b | 320 | 4.1523 | -0.0141 | n/a | 0.921 |
| tofu-final-340-a | 340 | 5.0316 | -0.0001 | n/a | 0.897 |
| tofu-final-340-b | 340 | 2.9296 | -0.0015 | n/a | 0.925 |
| tofu-final-360-a | 360 | 2.1789 | -0.0017 | n/a | 0.897 |
| tofu-final-360-b | 360 | 2.7676 | -0.0001 | n/a | 0.930 |
| tofu-final-380-a | 380 | 2.8580 | -0.0004 | n/a | 0.894 |
| tofu-final-380-b | 380 | 4.6900 | -0.0002 | n/a | 0.927 |

| Cohort measure | Estimate | Record bootstrap 95% | Source-block bootstrap 95% |
|---|---:|---:|---:|
| deleted_mean_token_log_probability_delta | 3.7594 | [2.9698, 4.5641] | [2.8501, 4.5508] |
| retained_mean_token_log_probability_delta | -0.0023 | [-0.0058, -0.0003] | [-0.0055, -0.0004] |
| update_seconds_difference | -0.7023 | [-0.7125, -0.6930] | [-0.7136, -0.6922] |
| update_seconds_ratio | n/a | n/a | n/a |
| query_seconds_difference | 0.0154 | [-0.0026, 0.0299] | [0.0035, 0.0272] |
| query_seconds_ratio | 1.0018 | [0.9994, 1.0037] | [1.0000, 1.0037] |
| end_to_end_seconds_difference | -0.6871 | [-0.7070, -0.6672] | [-0.7070, -0.6673] |
| end_to_end_seconds_ratio | 0.9147 | [0.9036, 0.9239] | [0.9105, 0.9209] |

