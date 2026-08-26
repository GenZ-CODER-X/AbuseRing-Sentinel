# Model C causal graph baseline

Model C is Model B plus exactly eight non-target, event-time graph/topology features. It uses a full-card-signature node (only when `card1`–`card6` are all present), `DeviceInfo`, and `addr1`; no other reusable graph nodes are created.

## Causality and leakage controls

The graph builder receives only `TransactionID`, `TransactionDT`, card fields, `DeviceInfo`, and `addr1`; it never receives `isFraud`. Events are sorted by time and processed in complete timestamp batches: every feature reads state from strictly earlier timestamps, then the entire batch updates state. Therefore an event cannot see itself or another event with the same `TransactionDT`. The train and validation intervals come only from `validation_boundaries.py`; no other partition is materialized or evaluated.

No target encoding, fraud statistics, V-family, historical labels, or future events are used. Model B categorical mappings and missing-value handling remain unchanged.

## Validation comparison

| Metric | Model A | Model B | Model C | C − B |
| --- | ---: | ---: | ---: | ---: |
| PR-AUC / average precision | 0.559602944785 | 0.568626882128 | 0.565242927150 | -0.003383954978 |
| ROC-AUC | 0.909745891888 | 0.911016440071 | 0.908071875108 | -0.002944564963 |
| Precision at selected threshold | 0.644533200200 | 0.642999006951 | 0.641753861485 | -0.001245145467 |
| Recall at selected threshold | 0.505679592636 | 0.507246376812 | 0.504504504505 | -0.002741872307 |
| F1 at selected threshold | 0.566725197542 | 0.567111889643 | 0.564912280702 | -0.002199608941 |
| Recall at FPR ≤1% | 0.505679592636 | 0.507246376812 | 0.504504504505 | -0.002741872307 |
| Recall at FPR ≤0.1% | 0.215041128085 | 0.243243243243 | 0.230708969839 | -0.012534273404 |

Model C selects threshold `0.744966673224` exactly once by maximizing validation recall under FPR ≤1%. Its confusion matrix `[[TN, FP], [FN, TP]]` is `[[71215, 719], [1265, 1288]]`.

## Graph feature importance

| Feature | Gain | Splits |
| --- | ---: | ---: |
| `graph_prior_card_transaction_count` | 51410.177851 | 289 |
| `graph_prior_address_distinct_cards` | 20647.080196 | 76 |
| `graph_prior_device_transaction_count` | 17999.674192 | 103 |
| `graph_prior_card_distinct_devices` | 16710.747772 | 122 |
| `graph_prior_device_distinct_cards` | 14879.533556 | 72 |
| `graph_prior_address_transaction_count` | 8262.853649 | 85 |
| `graph_prior_card_distinct_addresses` | 3383.628487 | 43 |
| `graph_prior_shared_entity_connectivity` | 2262.633316 | 8 |

The inherited fixed LightGBM configuration uses 500 iterations, learning rate 0.05, 31 leaves, no row/feature subsampling, deterministic execution, seed 20260826, and train-derived `scale_pos_weight=27.528364632238`.
