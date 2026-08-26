{
  "evidence": {
    "causality": "Features for a TransactionDT batch read state from strictly earlier TransactionDT values; all same-timestamp rows update state only after feature emission.",
    "features": [
      "tb_uid_prior_count",
      "tb_uid_amt_mean",
      "tb_uid_amt_std",
      "tb_uid_recency",
      "tb_uid_amt_zscore"
    ],
    "forbidden_inputs": [
      "isFraud",
      "target encoding",
      "fraud rates",
      "fraud counts",
      "future transactions"
    ],
    "graph_entity_nodes": [
      "card1",
      "card2",
      "card3",
      "card4",
      "card5",
      "card6",
      "addr1"
    ],
    "graph_input_columns": [
      "TransactionID",
      "TransactionDT",
      "card1",
      "card2",
      "card3",
      "card4",
      "card5",
      "card6",
      "addr1",
      "TransactionAmt"
    ],
    "label_column_passed_to_graph_builder": false,
    "same_timestamp_batching": true,
    "train_uid_event_row_count": 437511,
    "uid_event_row_count": 511998,
    "validation_uid_event_row_count": 74487
  },
  "experiment": "model_d",
  "feature_importance": [
    {
      "feature": "C1",
      "importance_gain": 79439.61821007729,
      "importance_split": 678
    },
    {
      "feature": "C13",
      "importance_gain": 51502.95904421806,
      "importance_split": 827
    },
    {
      "feature": "C14",
      "importance_gain": 40249.47725558281,
      "importance_split": 396
    },
    {
      "feature": "card1",
      "importance_gain": 32696.62684893608,
      "importance_split": 1968
    },
    {
      "feature": "C8",
      "importance_gain": 32178.70988059044,
      "importance_split": 216
    },
    {
      "feature": "TransactionDT",
      "importance_gain": 31111.690933942795,
      "importance_split": 1858
    },
    {
      "feature": "C4",
      "importance_gain": 30818.915367603302,
      "importance_split": 91
    },
    {
      "feature": "D2",
      "importance_gain": 28293.982002019882,
      "importance_split": 904
    },
    {
      "feature": "D3",
      "importance_gain": 27897.577304840088,
      "importance_split": 507
    },
    {
      "feature": "TransactionAmt",
      "importance_gain": 27544.187950134277,
      "importance_split": 1499
    },
    {
      "feature": "card2",
      "importance_gain": 27198.986416339874,
      "importance_split": 1575
    },
    {
      "feature": "addr1",
      "importance_gain": 23013.734753608704,
      "importance_split": 1370
    },
    {
      "feature": "C7",
      "importance_gain": 22670.922481536865,
      "importance_split": 42
    },
    {
      "feature": "D15",
      "importance_gain": 21982.85106420517,
      "importance_split": 1084
    },
    {
      "feature": "R_emaildomain",
      "importance_gain": 21218.73264813423,
      "importance_split": 431
    },
    {
      "feature": "D8",
      "importance_gain": 20790.68268418312,
      "importance_split": 713
    },
    {
      "feature": "tb_uid_amt_mean",
      "importance_gain": 20124.019989013672,
      "importance_split": 1221
    },
    {
      "feature": "C5",
      "importance_gain": 17883.42877316475,
      "importance_split": 270
    },
    {
      "feature": "tb_uid_prior_count",
      "importance_gain": 16679.47060728073,
      "importance_split": 1103
    },
    {
      "feature": "C11",
      "importance_gain": 15620.033174037933,
      "importance_split": 368
    },
    {
      "feature": "P_emaildomain",
      "importance_gain": 14742.369837999344,
      "importance_split": 795
    },
    {
      "feature": "D4",
      "importance_gain": 14565.17091846466,
      "importance_split": 811
    },
    {
      "feature": "C2",
      "importance_gain": 14276.136792182922,
      "importance_split": 521
    },
    {
      "feature": "DeviceInfo",
      "importance_gain": 13699.813871860504,
      "importance_split": 582
    },
    {
      "feature": "D10",
      "importance_gain": 13521.103386640549,
      "importance_split": 710
    },
    {
      "feature": "tb_uid_amt_std",
      "importance_gain": 12920.688715934753,
      "importance_split": 826
    },
    {
      "feature": "D1",
      "importance_gain": 12465.32534956932,
      "importance_split": 605
    },
    {
      "feature": "C6",
      "importance_gain": 12053.930432796478,
      "importance_split": 429
    },
    {
      "feature": "card6",
      "importance_gain": 11799.556919813156,
      "importance_split": 180
    },
    {
      "feature": "dist1",
      "importance_gain": 11070.67797279358,
      "importance_split": 751
    },
    {
      "feature": "tb_uid_recency",
      "importance_gain": 10789.269189119339,
      "importance_split": 770
    },
    {
      "feature": "card5",
      "importance_gain": 10776.658880710602,
      "importance_split": 614
    },
    {
      "feature": "C9",
      "importance_gain": 10638.579471826553,
      "importance_split": 369
    },
    {
      "feature": "M4",
      "importance_gain": 10487.0463950634,
      "importance_split": 316
    },
    {
      "feature": "M5",
      "importance_gain": 8957.46749997139,
      "importance_split": 271
    },
    {
      "feature": "tb_uid_amt_zscore",
      "importance_gain": 8832.218647956848,
      "importance_split": 728
    },
    {
      "feature": "D5",
      "importance_gain": 7970.791506528854,
      "importance_split": 399
    },
    {
      "feature": "card3",
      "importance_gain": 7317.386569738388,
      "importance_split": 275
    },
    {
      "feature": "D11",
      "importance_gain": 6834.604514122009,
      "importance_split": 456
    },
    {
      "feature": "D14",
      "importance_gain": 5798.100889444351,
      "importance_split": 333
    },
    {
      "feature": "addr2",
      "importance_gain": 5592.365891695023,
      "importance_split": 42
    },
    {
      "feature": "dist2",
      "importance_gain": 5391.644459247589,
      "importance_split": 456
    },
    {
      "feature": "M6",
      "importance_gain": 4635.490978717804,
      "importance_split": 224
    },
    {
      "feature": "D7",
      "importance_gain": 4243.58021235466,
      "importance_split": 115
    },
    {
      "feature": "D9",
      "importance_gain": 3965.754988193512,
      "importance_split": 368
    },
    {
      "feature": "C12",
      "importance_gain": 3771.4101202487946,
      "importance_split": 189
    },
    {
      "feature": "D6",
      "importance_gain": 3551.425199985504,
      "importance_split": 200
    },
    {
      "feature": "D13",
      "importance_gain": 3439.8887145519257,
      "importance_split": 219
    },
    {
      "feature": "D12",
      "importance_gain": 3363.4551453590393,
      "importance_split": 195
    },
    {
      "feature": "card4",
      "importance_gain": 3350.9379901885986,
      "importance_split": 240
    },
    {
      "feature": "C10",
      "importance_gain": 3067.365731239319,
      "importance_split": 160
    },
    {
      "feature": "ProductCD",
      "importance_gain": 1850.5489614009857,
      "importance_split": 75
    },
    {
      "feature": "DeviceType",
      "importance_gain": 1202.9977972507477,
      "importance_split": 106
    },
    {
      "feature": "M3",
      "importance_gain": 1186.6185455322266,
      "importance_split": 67
    },
    {
      "feature": "M9",
      "importance_gain": 790.91308426857,
      "importance_split": 62
    },
    {
      "feature": "M7",
      "importance_gain": 777.6739490032196,
      "importance_split": 71
    },
    {
      "feature": "M2",
      "importance_gain": 556.4954288005829,
      "importance_split": 40
    },
    {
      "feature": "M8",
      "importance_gain": 489.41439867019653,
      "importance_split": 50
    },
    {
      "feature": "M1",
      "importance_gain": 145.62452220916748,
      "importance_split": 11
    }
  ],
  "metrics": {
    "F1": 0.5480671928754366,
    "PR-AUC": 0.5695818780374736,
    "ROC-AUC": 0.9236835304190526,
    "Recall@FPR<=0.1%": 0.22483352918135527,
    "Recall@FPR<=1%": 0.47473560517038776,
    "confusion_matrix": {
      "fn": 1199,
      "fp": 1034,
      "tn": 70900,
      "tp": 1354
    },
    "precision": 0.567001675041876,
    "selected_threshold": 0.15589992374736117
  },
  "model": "Model D (UID)",
  "row_counts": {
    "train": 437511,
    "validation": 74487
  },
  "uid_feature_importance": [
    {
      "feature": "tb_uid_amt_mean",
      "importance_gain": 20124.019989013672,
      "importance_split": 1221
    },
    {
      "feature": "tb_uid_prior_count",
      "importance_gain": 16679.47060728073,
      "importance_split": 1103
    },
    {
      "feature": "tb_uid_amt_std",
      "importance_gain": 12920.688715934753,
      "importance_split": 826
    },
    {
      "feature": "tb_uid_recency",
      "importance_gain": 10789.269189119339,
      "importance_split": 770
    },
    {
      "feature": "tb_uid_amt_zscore",
      "importance_gain": 8832.218647956848,
      "importance_split": 728
    }
  ]
}
