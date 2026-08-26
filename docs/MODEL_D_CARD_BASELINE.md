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
      "card6"
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
      "TransactionAmt"
    ],
    "label_column_passed_to_graph_builder": false,
    "same_timestamp_batching": true,
    "train_uid_event_row_count": 437511,
    "uid_event_row_count": 511998,
    "validation_uid_event_row_count": 74487
  },
  "experiment": "model_d_card",
  "feature_importance": [
    {
      "feature": "C1",
      "importance_gain": 78262.70794320107,
      "importance_split": 629
    },
    {
      "feature": "C13",
      "importance_gain": 50602.866559267044,
      "importance_split": 794
    },
    {
      "feature": "C14",
      "importance_gain": 43786.69230055809,
      "importance_split": 429
    },
    {
      "feature": "C4",
      "importance_gain": 39394.263409376144,
      "importance_split": 77
    },
    {
      "feature": "TransactionDT",
      "importance_gain": 31866.194214105606,
      "importance_split": 1825
    },
    {
      "feature": "D2",
      "importance_gain": 29972.284006357193,
      "importance_split": 884
    },
    {
      "feature": "card1",
      "importance_gain": 29924.946539640427,
      "importance_split": 1712
    },
    {
      "feature": "C8",
      "importance_gain": 28394.528749465942,
      "importance_split": 184
    },
    {
      "feature": "D3",
      "importance_gain": 27658.365862607956,
      "importance_split": 447
    },
    {
      "feature": "card2",
      "importance_gain": 26554.976240873337,
      "importance_split": 1355
    },
    {
      "feature": "TransactionAmt",
      "importance_gain": 24782.760145187378,
      "importance_split": 1309
    },
    {
      "feature": "addr1",
      "importance_gain": 24180.569854736328,
      "importance_split": 1308
    },
    {
      "feature": "tb_uid_amt_mean",
      "importance_gain": 22968.899926424026,
      "importance_split": 1373
    },
    {
      "feature": "C7",
      "importance_gain": 21826.3485789299,
      "importance_split": 49
    },
    {
      "feature": "tb_uid_prior_count",
      "importance_gain": 21463.57204055786,
      "importance_split": 1412
    },
    {
      "feature": "D15",
      "importance_gain": 19662.993153095245,
      "importance_split": 977
    },
    {
      "feature": "C5",
      "importance_gain": 19027.02695608139,
      "importance_split": 277
    },
    {
      "feature": "R_emaildomain",
      "importance_gain": 17935.626346826553,
      "importance_split": 401
    },
    {
      "feature": "C11",
      "importance_gain": 16549.00090932846,
      "importance_split": 366
    },
    {
      "feature": "D8",
      "importance_gain": 16411.34456014633,
      "importance_split": 670
    },
    {
      "feature": "tb_uid_amt_std",
      "importance_gain": 16005.86919760704,
      "importance_split": 1007
    },
    {
      "feature": "P_emaildomain",
      "importance_gain": 15238.95421719551,
      "importance_split": 730
    },
    {
      "feature": "C2",
      "importance_gain": 13582.743639469147,
      "importance_split": 524
    },
    {
      "feature": "D4",
      "importance_gain": 13551.010133981705,
      "importance_split": 753
    },
    {
      "feature": "tb_uid_amt_zscore",
      "importance_gain": 12772.981594324112,
      "importance_split": 971
    },
    {
      "feature": "D1",
      "importance_gain": 12680.633615255356,
      "importance_split": 593
    },
    {
      "feature": "DeviceInfo",
      "importance_gain": 12363.22743344307,
      "importance_split": 528
    },
    {
      "feature": "D10",
      "importance_gain": 12146.039729833603,
      "importance_split": 675
    },
    {
      "feature": "C6",
      "importance_gain": 11520.617058753967,
      "importance_split": 414
    },
    {
      "feature": "dist1",
      "importance_gain": 10887.851819992065,
      "importance_split": 740
    },
    {
      "feature": "tb_uid_recency",
      "importance_gain": 10850.727490663528,
      "importance_split": 879
    },
    {
      "feature": "card6",
      "importance_gain": 9709.369405508041,
      "importance_split": 161
    },
    {
      "feature": "M4",
      "importance_gain": 9550.709003448486,
      "importance_split": 287
    },
    {
      "feature": "M5",
      "importance_gain": 9370.43509054184,
      "importance_split": 246
    },
    {
      "feature": "card5",
      "importance_gain": 9319.130010128021,
      "importance_split": 497
    },
    {
      "feature": "C9",
      "importance_gain": 8903.477302312851,
      "importance_split": 351
    },
    {
      "feature": "D5",
      "importance_gain": 7740.958116054535,
      "importance_split": 352
    },
    {
      "feature": "card3",
      "importance_gain": 6286.064444303513,
      "importance_split": 254
    },
    {
      "feature": "D11",
      "importance_gain": 6110.8249888420105,
      "importance_split": 379
    },
    {
      "feature": "addr2",
      "importance_gain": 5526.575661659241,
      "importance_split": 42
    },
    {
      "feature": "D14",
      "importance_gain": 4966.11839556694,
      "importance_split": 285
    },
    {
      "feature": "dist2",
      "importance_gain": 4479.871362686157,
      "importance_split": 415
    },
    {
      "feature": "C12",
      "importance_gain": 4448.457449436188,
      "importance_split": 173
    },
    {
      "feature": "D7",
      "importance_gain": 4393.864100933075,
      "importance_split": 100
    },
    {
      "feature": "M6",
      "importance_gain": 3854.70929145813,
      "importance_split": 184
    },
    {
      "feature": "D6",
      "importance_gain": 3756.9795372486115,
      "importance_split": 204
    },
    {
      "feature": "card4",
      "importance_gain": 3585.723284482956,
      "importance_split": 227
    },
    {
      "feature": "D9",
      "importance_gain": 3405.8710582256317,
      "importance_split": 310
    },
    {
      "feature": "D12",
      "importance_gain": 3194.929501771927,
      "importance_split": 179
    },
    {
      "feature": "D13",
      "importance_gain": 3120.79763007164,
      "importance_split": 211
    },
    {
      "feature": "C10",
      "importance_gain": 2190.754441022873,
      "importance_split": 148
    },
    {
      "feature": "ProductCD",
      "importance_gain": 1584.6199405193329,
      "importance_split": 60
    },
    {
      "feature": "M3",
      "importance_gain": 1268.9122183322906,
      "importance_split": 68
    },
    {
      "feature": "DeviceType",
      "importance_gain": 1120.473938703537,
      "importance_split": 78
    },
    {
      "feature": "M7",
      "importance_gain": 820.937686920166,
      "importance_split": 66
    },
    {
      "feature": "M9",
      "importance_gain": 774.4715042114258,
      "importance_split": 58
    },
    {
      "feature": "M2",
      "importance_gain": 321.1349210739136,
      "importance_split": 26
    },
    {
      "feature": "M8",
      "importance_gain": 309.55946254730225,
      "importance_split": 33
    },
    {
      "feature": "M1",
      "importance_gain": 234.0912103652954,
      "importance_split": 12
    }
  ],
  "metrics": {
    "F1": 0.5435395899938585,
    "PR-AUC": 0.5602746224535499,
    "ROC-AUC": 0.9195638528206063,
    "Recall@FPR<=0.1%": 0.23305914610262438,
    "Recall@FPR<=1%": 0.46533490011750883,
    "confusion_matrix": {
      "fn": 1108,
      "fp": 1319,
      "tn": 70615,
      "tp": 1445
    },
    "precision": 0.5227930535455861,
    "selected_threshold": 0.13059614901178407
  },
  "model": "Model D-card (UID card1-6 only)",
  "row_counts": {
    "train": 437511,
    "validation": 74487
  },
  "uid_feature_importance": [
    {
      "feature": "tb_uid_amt_mean",
      "importance_gain": 22968.899926424026,
      "importance_split": 1373
    },
    {
      "feature": "tb_uid_prior_count",
      "importance_gain": 21463.57204055786,
      "importance_split": 1412
    },
    {
      "feature": "tb_uid_amt_std",
      "importance_gain": 16005.86919760704,
      "importance_split": 1007
    },
    {
      "feature": "tb_uid_amt_zscore",
      "importance_gain": 12772.981594324112,
      "importance_split": 971
    },
    {
      "feature": "tb_uid_recency",
      "importance_gain": 10850.727490663528,
      "importance_split": 879
    }
  ]
}
