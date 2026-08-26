# Model A baseline

## Objective

Model A is the first reproducible, leakage-safe LightGBM fraud-risk benchmark for AbuseRing Sentinel. It deliberately uses event-time transaction and entity fields only, so later experiments can measure the incremental value of richer feature sets.

## Temporal split and leakage controls

The script imports the locked boundaries from `scripts/validation_boundaries.py`; it does not recreate them. Training uses `TransactionDT` 86,400–11,059,199 (437,511 rows) and validation uses 11,059,200–13,391,999 (74,487 rows). There is no random split or shuffle. The experiment materializes and evaluates only those two partitions; no other partition is scored, counted, or used for metrics or tuning.

All category mappings are fitted from training values only. No target encoding, fraud/entity rate, historical/velocity/frequency aggregate, graph, identity, device, V-family, or other derived feature is used. `isFraud` is used only as the training target and validation label.

## Features

The 50 approved features are: `TransactionAmt`, `ProductCD`, `TransactionDT`, `dist1`, `dist2`, `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `C1`, `C2`, `C4`, `C5`, `C6`, `C7`, `C8`, `C9`, `C10`, `C11`, `C12`, `C13`, `C14`, `D1`, `D2`, `D3`, `D4`, `D5`, `D6`, `D7`, `D8`, `D9`, `D10`, `D11`, `D12`, `D13`, `D14`, `D15`, `M1`, `M2`, `M3`, `M4`, `M5`, `M6`, `M7`, `M8`, `M9`.

Categorical features are: `ProductCD`, `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M1`, `M2`, `M3`, `M4`, `M5`, `M6`, `M7`, `M8`, `M9`. Numeric features are: `TransactionAmt`, `TransactionDT`, `dist1`, `dist2`, `C1`, `C2`, `C4`, `C5`, `C6`, `C7`, `C8`, `C9`, `C10`, `C11`, `C12`, `C13`, `C14`, `D1`, `D2`, `D3`, `D4`, `D5`, `D6`, `D7`, `D8`, `D9`, `D10`, `D11`, `D12`, `D13`, `D14`, `D15`.

## Preprocessing and model

Numeric nulls remain `NaN` for LightGBM's native missing-value handling. Training categories are assigned deterministic positive integer codes; missing categorical values receive `-1` (LightGBM categorical missing), and validation-only categories receive the fixed reserved code `0`. This means numeric-coded card and address fields are handled categorically, not as ordered measurements. Training category cardinalities are recorded in the experiment metadata.

The predeclared model is LightGBM GBDT with 500 fixed boosting iterations, learning rate 0.05, 31 leaves, `min_child_samples=100`, no row/feature subsampling, deterministic column-wise execution, seed 20,260,826, and `scale_pos_weight=27.528364632238` derived only from the train target prevalence. There is no early stopping or iterative validation tuning.

## Validation results

| Metric | Value |
| --- | ---: |
| PR-AUC / average precision | 0.559602944785 |
| ROC-AUC | 0.909745891888 |
| Validation fraud prevalence | 3.427443715011% |
| Selected threshold | 0.749676415812 |
| Precision at threshold | 0.644533200200 |
| Recall at threshold | 0.505679592636 |
| F1 at threshold | 0.566725197542 |
| Recall at FPR ≤1% | 0.505679592636 |
| Recall at FPR ≤0.1% | 0.215041128085 |
| Validation transactions flagged | 2,003 |
| False positives | 712 |
| False negatives | 1,262 |

Confusion matrix (`[[TN, FP], [FN, TP]]`): `[[71222, 712], [1262, 1291]]`.

## Threshold selection

The operating threshold was selected exactly once from validation predictions: maximize recall under validation FPR ≤1%; ties use the higher threshold to reduce unnecessary flags. The chosen threshold has FPR 0.009897962021 and recall 0.505679592636. The 0.1% FPR result is reported as a separate constrained-recall diagnostic, not as a tuning loop.

## Memory behavior and limitations

Polars lazily scans the CSV and projects only approved fields plus the target. It materializes the necessary train and validation selected-column frames separately. Observed selected-frame/matrix sizes were 144,637,704/87,502,200 bytes for train and 24,683,391/14,897,400 bytes for validation; the full 590,540-row transaction table is never collected.

Model A contains no V-family, identity/device, behavioral temporal, graph, target-encoding, historical aggregate, or production-serving components. It is a deliberately bounded benchmark, not a final fraud model.
