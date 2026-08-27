# Model E Baseline (Entity Features Final)

Model E is Model B plus nine entity-level temporal features (card1-6 + addr1).

## Hypothesis
Aggregating historical behavior at the entity level using UID_G = card1 + card2 + card3 + card4 + card5 + card6 + addr1 will capture stable behavioral patterns that transaction-level features alone miss, thereby improving fraud detection performance—specifically, Recall@FPR≤1%—without introducing leakage.

## Entity Definition
UID_G = card1|card2|card3|card4|card5|card6|addr1 (concatenated string of the seven fields, only when all are non-null).

## Exact Features Added
1. `tb_entity_prior_count` – number of strictly prior transactions for the entity.
2. `tb_entity_amt_mean` – mean TransactionAmt over strictly prior entity transactions.
3. `tb_entity_amt_std` – standard deviation of TransactionAmt over strictly prior entity transactions.
4. `tb_entity_amt_entropy` – Shannon entropy of discretized historical TransactionAmt values (10 equal-width bins from 0 to 2000).
5. `tb_entity_txn_velocity` – prior transaction count divided by elapsed entity lifetime in days (elapsed = now - first_seen_timestamp).
6. `tb_entity_amt_velocity` – sum of prior TransactionAmt divided by elapsed entity lifetime in days.
7. `tb_entity_first_seen` – seconds since first strictly prior transaction for the entity (0 for unseen entities).
8. `tb_entity_unique_product_count` – number of distinct ProductCD values previously observed for the entity.
9. `tb_entity_amt_trend` – historical transaction-amount trend/slope over time (online linear regression slope of amount vs. timestamp).

## Temporal Semantics & Leakage Controls
- For every TransactionDT timestamp batch:
  1. Read ONLY state from TransactionDT values strictly earlier than the current timestamp.
  2. Generate features for the COMPLETE current timestamp batch.
  3. Only AFTER all rows in that timestamp batch have been scored, update entity state using that batch.
- Therefore:
  - a transaction cannot see itself
  - transactions at the same timestamp cannot see one another
  - future transactions cannot affect past rows
- The feature builder does NOT receive `isFraud` or any label-derived information.
- Label information is used ONLY by LightGBM for training and evaluation.

## Validation Boundaries
Exactly the same train/validation split as Model B and Model D, defined in `validation_boundaries.py`:
- Train: TransactionDT in [86400, 11059199]
- Validation: TransactionDT in [11059200, 13391999]

## Model B vs Model E Metrics

| Metric | Model B | Model E | Delta (E − B) |
|--------|---------|---------|---------------|
| PR‑AUC | 0.5686268821 | 0.5508057463 | -0.01782114 |
| ROC‑AUC | 0.9110164401 | 0.9171894862 | +0.00617305 |
| Recall@FPR≤1% | 0.5072463768 | 0.4500587544 | -0.05718762 |
| Recall@FPR≤0.1% | 0.2432432432 | 0.2314923619 | -0.01175088 |
| F1 | 0.5671118896 | 0.5224706664 | -0.04464122 |
| Precision | 0.6429990070 | 0.6008146640 | -0.04218434 |
| Selected Threshold | 0.7391097672 | 0.2040926191 | -0.53501715 |

## Feature Importance (Top 5 Entity Features by Gain)
All nine entity features received non‑zero gain, indicating the model used them:
1. `tb_entity_amt_mean`: gain=14492.61998, split=657
2. `tb_entity_txn_velocity`: gain=10009.09718, split=424
3. `tb_entity_prior_count`: gain=8126.21736, split=432
4. `tb_entity_amt_velocity`: gain=7896.99471, split=388
5. `tb_entity_amt_std`: gain=7517.56070, split=405

Full feature importance: `artifacts/models/model_e_final/feature_importance.csv`

## Leakage Tests Passed
All unit tests for the entity feature constructor pass (12/12):
- First-seen entity
- Repeated entity
- Same‑timestamp rows do not see each other
- Future‑row invariance
- Missing field handling
- No label access
- Deterministic execution
- Chronological train→validation behavior
- Zero/one prior association
- Insufficient history for std/z-score
- Entropy with different bins
- Velocity zero elapsed
- Trend constant amount

## Hypothesis Verdict
**NOT SUPPORTED** for the primary objective.
- Model E **degrades** the primary metric Recall@FPR≤1% by 0.0572 (relative drop ~11.3%).
- Although ROC‑AUC improves (+0.0062), indicating better ranking quality, the operational point at FPR≤1% worsens.
- The entity features provide useful signal but not of the right kind to improve recall at the chosen false‑positive threshold.

## Limitations
- The entity definition (card1‑6 + addr1) may be too coarse or too fine for capturing fraudulent behavior.
- The temporal features (mean, std, entropy, velocity, trend) may not be predictive of fraud at the FPR≤1% operating point.
- The linearity assumption in `tb_entity_amt_trend` may not capture complex temporal patterns.
- Entropy binning (0‑2000) may not be optimal for the TransactionAmt distribution.

## Final Decision
**Model hunt ends.** Model B remains the champion. No further model (F/G/H) will be invented. The negative result is scientifically valuable: it shows that simply aggregating historical behavior at the entity level—without targeting the specific behavioral changes that signal fraud at the chosen operating threshold—is insufficient to improve Recall@FPR≤1% in this dataset.

## Next Phase
Move to evaluation, investigation workflow, explainability, testing, deployment, and documentation of the champion model (Model B).