# Model D temporal behavioral graph baseline

Model D is Model B plus exactly six non-target, event-time temporal behavioral graph/topology features. It uses a full-card-signature node (only when `card1`–`card6` are all present), `DeviceInfo`, and `addr1`; no other reusable graph nodes are created.

## Causality and leakage controls

The graph builder receives only `TransactionID`, `TransactionDT`, card fields, `DeviceInfo`, and `addr1`; it never receives `isFraud`. Events are sorted by time and processed in complete timestamp batches: every feature reads state from strictly earlier timestamps, then the entire batch updates state. Therefore an event cannot see itself or another event with the same `TransactionDT`. The train and validation intervals come only from `validation_boundaries.py`; no other partition is materialized or evaluated.

No target encoding, fraud statistics, V-family, historical labels, or future events are used. Model B categorical mappings and missing-value handling remain unchanged.

## Validation comparison

| Metric | Model A | Model B | Model D | D − B |
| --- | ---: | ---: | ---: | ---: |
| PR-AUC / average precision | 0.559602944785 | 0.568626882128 | 0.568111297885 | -0.000515584243 |
| ROC-AUC | 0.909745891888 | 0.911016440071 | 0.912237744459 | +0.001221304388 |
| Precision at selected threshold | 0.644533200200 | 0.642999006951 | 0.637329286798 | -0.005669720153 |
| Recall at selected threshold | 0.505679592636 | 0.507246376812 | 0.493537015276 | -0.013709361535 |
| F1 at selected threshold | 0.566725197542 | 0.567111889643 | 0.556291390728 | -0.010820498915 |
| Recall at FPR ≤1% | 0.505679592636 | 0.507246376812 | 0.493537015276 | -0.013709361535 |
| Recall at FPR ≤0.1% | 0.215041128085 | 0.243243243243 | 0.248726987857 | +0.005483744614 |

Model D selects threshold `0.749799628164` exactly once by maximizing validation recall under FPR ≤1%. Its confusion matrix `[[TN, FP], [FN, TP]]` is `[[71217, 717], [1293, 1260]]`.

## Graph feature importance

| Feature | Gain | Splits |
| --- | ---: | ---: |
| `tb_graph_card_amt_zscore` | 23670.999332 | 117 |
| `tb_graph_card_recency` | 20024.524799 | 162 |
| `tb_graph_card_amt_mean` | 13388.468231 | 136 |
| `tb_graph_pair_recency` | 7453.794847 | 52 |
| `tb_graph_card_address_stability` | 3865.546223 | 40 |
| `tb_graph_card_velocity` | 1860.699114 | 31 |
| `tb_graph_pair_velocity` | 1419.253502 | 28 |
| `tb_graph_card_device_stability` | 1340.311319 | 19 |
| `tb_graph_is_novel_card_device` | 0.000000 | 0 |

The inherited fixed LightGBM configuration uses 500 iterations, learning rate 0.05, 31 leaves, no row/feature subsampling, deterministic execution, seed 20260826, and train-derived `scale_pos_weight=27.528364632238`.
