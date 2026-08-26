# Model D-card Experiment Summary

## Objective
To test whether removing `addr1` from the UID_G field (i.e., using only `card1-6` for the UID) improves or degrades the temporal identity signal compared to the original Model D (which uses `card1-6 + addr1`).

## Frozen References
- Model B PR-AUC: 0.5686268821
- Model D (card1-6 + addr1) PR-AUC: 0.5695818780

## Experiment: Model D-card (card1-6 only)
- UID_G construct: `card1|card2|card3|card4|card5|card6` (addr1 excluded)
- Five strict-prior temporal features:
  - `tb_uid_prior_count`
  - `tb_uid_amt_mean`
  - `tb_uid_amt_std`
  - `tb_uid_recency`
  - `tb_uid_amt_zscore`
- All other aspects identical to Model D:
  - Same Model B feature set
  - Same LightGBM configuration (learning_rate=0.05, num_leaves=63, etc.)
  - Same train/validation boundaries from `validation_boundaries.py`
  - Same evaluation metrics (PR-AUC, ROC-AUC, Recall@FPR<=1%/0.1%, F1, precision)
  - Same artifact isolation (output directory: `artifacts/models/model_d_card`)

## Results
### Validation Metrics (Model D-card)
- PR-AUC: 0.5602746224535499
- ROC-AUC: 0.9195638528206063
- Recall@FPR<=1%: 0.46533490011750883
- Recall@FPR<=0.1%: 0.23305914610262438
- F1: 0.5435395899938585
- Precision: 0.5227930535455861
- Selected Threshold: 0.13059614901178407
- Confusion Matrix:
  - TN: 70615
  - FP: 1319
  - FN: 1108
  - TP: 1445

### Top 5 UID Features by Gain
1. `tb_uid_amt_mean`: gain=22968.899926, split=1373
2. `tb_uid_prior_count`: gain=21463.572041, split=1412
3. `tb_uid_amt_std`: gain=16005.869198, split=1007
4. `tb_uid_amt_zscore`: gain=12772.981594, split=971
5. `tb_uid_recency`: gain=10850.727491, split=879

## Comparison with Baselines
| Model | PR-AUC | Δ vs Model D-card |
|-------|--------|-------------------|
| Model B | 0.5686268821 | +0.0083522596 |
| Model D (card1-6 + addr1) | 0.5695818780 | +0.0093072555 |
| Model D-card (this experiment) | 0.5602746225 | — |

## Interpretation
- Removing `addr1` from the UID_G field results in a decrease in PR-AUC of approximately 0.0084 relative to Model B and 0.0093 relative to Model D.
- This suggests that the `addr1` component contributes positively to the temporal identity signal in the UID_G feature set.
- The ROC-AUC also decreases slightly (Model D-card: 0.91956 vs Model D: 0.92368 from `docs/MODEL_D_BASELINE.md`).

## Artifacts Generated
- Model: `artifacts/models/model_d_card/model.lgb`
- Feature list: `artifacts/models/model_d_card/feature_list.json`
- Validation metrics: `docs/MODEL_D_CARD_BASELINE.md`
- Configuration metadata: `artifacts/models/model_d_card/config_metadata.json` (not shown in output but created by the script)

## Conclusion
The controlled experiment confirms that the `addr1` field provides a small but measurable improvement to the temporal identity signal when included in the UID_G construct. The card-only UID (card1-6) is a valid isolation experiment, and all implementation requirements were met:
- No modifications to protected files (Model A/B/C, validation boundaries, causal graph features, TBGF, or existing Model D).
- Strict streaming/leakage rules preserved.
- Identical train/validation boundaries and LightGBM configuration.
- Comprehensive unit tests pass for the new UID card feature construction.

Next steps: Await further instructions on whether to proceed with additional analysis or experiments.