# Model B baseline

Model B is a controlled extension of Model A: it adds only raw categorical `DeviceInfo`, `DeviceType`, `card4`, and `card6`. All Model A features, locked chronological boundaries, fixed 500-iteration LightGBM configuration, train-only categorical mappings, and validation-only threshold methodology are retained.

No graph, composite multi-transaction, target-encoded, historical, velocity, frequency, V-family, or other derived features are included. Numeric nulls remain `NaN`; categorical missing values use `-1`, and validation-only categorical values use reserved code `0`.

## A vs. B validation comparison

| Metric | Model A | Model B | B − A |
| --- | ---: | ---: | ---: |
| PR-AUC / average precision | 0.559602944785 | 0.568626882128 | +0.009023937343 |
| ROC-AUC | 0.909745891888 | 0.911016440071 | +0.001270548183 |
| Precision at selected threshold | 0.644533200200 | 0.642999006951 | -0.001534193248 |
| Recall at selected threshold | 0.505679592636 | 0.507246376812 | +0.001566784175 |
| F1 at selected threshold | 0.566725197542 | 0.567111889643 | +0.000386692101 |

## Model B operating point

Threshold `0.739109767178` maximizes validation recall under FPR ≤1%, with ties resolved by higher threshold. It yields precision 0.642999006951, recall 0.507246376812, F1 0.567111889643, and FPR 0.009995273445. Recall at FPR ≤0.1% is 0.243243243243.

Confusion matrix `[[TN, FP], [FN, TP]]`: `[[71215, 719], [1258, 1295]]`.

The fixed model configuration retains `n_estimators=500`, `learning_rate=0.05`, `num_leaves=31`, deterministic execution, no row/feature subsampling, and train-derived `scale_pos_weight=27.528364632238`. Selected train/validation frame and matrix sizes are 151,381,230/94,502,376 and 25,744,294/16,089,192 bytes.
