{
  "entity_feature_importance": [
    {
      "feature": "tb_entity_amt_mean",
      "importance_gain": 14492.61997961998,
      "importance_split": 657
    },
    {
      "feature": "tb_entity_txn_velocity",
      "importance_gain": 10009.097179412842,
      "importance_split": 424
    },
    {
      "feature": "tb_entity_prior_count",
      "importance_gain": 8126.217358589172,
      "importance_split": 432
    },
    {
      "feature": "tb_entity_amt_velocity",
      "importance_gain": 7896.99471449852,
      "importance_split": 388
    },
    {
      "feature": "tb_entity_amt_std",
      "importance_gain": 7517.560701608658,
      "importance_split": 405
    },
    {
      "feature": "tb_entity_amt_entropy",
      "importance_gain": 7424.794735193253,
      "importance_split": 370
    },
    {
      "feature": "tb_entity_first_seen",
      "importance_gain": 7360.012335300446,
      "importance_split": 397
    },
    {
      "feature": "tb_entity_amt_trend",
      "importance_gain": 7314.323886871338,
      "importance_split": 462
    },
    {
      "feature": "tb_entity_unique_product_count",
      "importance_gain": 5078.684554576874,
      "importance_split": 159
    }
  ],
  "evidence": {
    "causality": "Features for a TransactionDT batch read state from strictly earlier TransactionDT values; all same-timestamp rows update state only after feature emission.",
    "entity_event_row_count": 511998,
    "features": [
      "tb_entity_prior_count",
      "tb_entity_amt_mean",
      "tb_entity_amt_std",
      "tb_entity_amt_entropy",
      "tb_entity_txn_velocity",
      "tb_entity_amt_velocity",
      "tb_entity_first_seen",
      "tb_entity_unique_product_count",
      "tb_entity_amt_trend"
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
      "TransactionAmt",
      "ProductCD"
    ],
    "label_column_passed_to_graph_builder": false,
    "same_timestamp_batching": true,
    "train_entity_event_row_count": 437511,
    "validation_entity_event_row_count": 74487
  },
  "experiment": "model_e",
  "feature_importance": [
    {
      "feature": "C1",
      "importance_gain": 71803.39422059059,
      "importance_split": 485
    },
    {
      "feature": "C13",
      "importance_gain": 52842.322539806366,
      "importance_split": 557
    },
    {
      "feature": "C14",
      "importance_gain": 38418.38343858719,
      "importance_split": 330
    },
    {
      "feature": "C8",
      "importance_gain": 38374.22200059891,
      "importance_split": 165
    },
    {
      "feature": "D3",
      "importance_gain": 30183.73160147667,
      "importance_split": 354
    },
    {
      "feature": "D2",
      "importance_gain": 27504.727506160736,
      "importance_split": 601
    },
    {
      "feature": "card1",
      "importance_gain": 25858.42377758026,
      "importance_split": 1237
    },
    {
      "feature": "TransactionDT",
      "importance_gain": 25277.604001760483,
      "importance_split": 1071
    },
    {
      "feature": "C4",
      "importance_gain": 24959.228481531143,
      "importance_split": 77
    },
    {
      "feature": "card2",
      "importance_gain": 24343.25920200348,
      "importance_split": 959
    },
    {
      "feature": "TransactionAmt",
      "importance_gain": 23573.091086387634,
      "importance_split": 992
    },
    {
      "feature": "C7",
      "importance_gain": 21812.927416324615,
      "importance_split": 25
    },
    {
      "feature": "addr1",
      "importance_gain": 20446.52858710289,
      "importance_split": 845
    },
    {
      "feature": "D15",
      "importance_gain": 18574.954612493515,
      "importance_split": 650
    },
    {
      "feature": "D8",
      "importance_gain": 17785.111139059067,
      "importance_split": 498
    },
    {
      "feature": "R_emaildomain",
      "importance_gain": 17024.247464179993,
      "importance_split": 296
    },
    {
      "feature": "C5",
      "importance_gain": 16993.481929302216,
      "importance_split": 163
    },
    {
      "feature": "C11",
      "importance_gain": 15048.033238887787,
      "importance_split": 264
    },
    {
      "feature": "D4",
      "importance_gain": 14740.509785413742,
      "importance_split": 571
    },
    {
      "feature": "tb_entity_amt_mean",
      "importance_gain": 14492.61997961998,
      "importance_split": 657
    },
    {
      "feature": "P_emaildomain",
      "importance_gain": 13544.006432056427,
      "importance_split": 546
    },
    {
      "feature": "D1",
      "importance_gain": 11902.475890874863,
      "importance_split": 432
    },
    {
      "feature": "C2",
      "importance_gain": 11681.282247543335,
      "importance_split": 340
    },
    {
      "feature": "D10",
      "importance_gain": 11669.099157571793,
      "importance_split": 490
    },
    {
      "feature": "card6",
      "importance_gain": 11105.92299413681,
      "importance_split": 141
    },
    {
      "feature": "tb_entity_txn_velocity",
      "importance_gain": 10009.097179412842,
      "importance_split": 424
    },
    {
      "feature": "M5",
      "importance_gain": 9722.526892662048,
      "importance_split": 207
    },
    {
      "feature": "C6",
      "importance_gain": 9705.413658618927,
      "importance_split": 309
    },
    {
      "feature": "DeviceInfo",
      "importance_gain": 9633.503413200378,
      "importance_split": 401
    },
    {
      "feature": "C9",
      "importance_gain": 8811.18095445633,
      "importance_split": 282
    },
    {
      "feature": "dist1",
      "importance_gain": 8601.05914568901,
      "importance_split": 469
    },
    {
      "feature": "tb_entity_prior_count",
      "importance_gain": 8126.217358589172,
      "importance_split": 432
    },
    {
      "feature": "tb_entity_amt_velocity",
      "importance_gain": 7896.99471449852,
      "importance_split": 388
    },
    {
      "feature": "card3",
      "importance_gain": 7771.039249181747,
      "importance_split": 245
    },
    {
      "feature": "M4",
      "importance_gain": 7591.974071979523,
      "importance_split": 198
    },
    {
      "feature": "tb_entity_amt_std",
      "importance_gain": 7517.560701608658,
      "importance_split": 405
    },
    {
      "feature": "tb_entity_amt_entropy",
      "importance_gain": 7424.794735193253,
      "importance_split": 370
    },
    {
      "feature": "tb_entity_first_seen",
      "importance_gain": 7360.012335300446,
      "importance_split": 397
    },
    {
      "feature": "tb_entity_amt_trend",
      "importance_gain": 7314.323886871338,
      "importance_split": 462
    },
    {
      "feature": "card5",
      "importance_gain": 7094.777166366577,
      "importance_split": 357
    },
    {
      "feature": "D7",
      "importance_gain": 5708.62069606781,
      "importance_split": 62
    },
    {
      "feature": "D11",
      "importance_gain": 5482.405994653702,
      "importance_split": 254
    },
    {
      "feature": "D5",
      "importance_gain": 5468.40273809433,
      "importance_split": 210
    },
    {
      "feature": "tb_entity_unique_product_count",
      "importance_gain": 5078.684554576874,
      "importance_split": 159
    },
    {
      "feature": "D14",
      "importance_gain": 5068.8500900268555,
      "importance_split": 226
    },
    {
      "feature": "addr2",
      "importance_gain": 4681.545614719391,
      "importance_split": 37
    },
    {
      "feature": "D12",
      "importance_gain": 4430.691617965698,
      "importance_split": 138
    },
    {
      "feature": "dist2",
      "importance_gain": 4371.001514196396,
      "importance_split": 300
    },
    {
      "feature": "C10",
      "importance_gain": 3798.763972759247,
      "importance_split": 99
    },
    {
      "feature": "M6",
      "importance_gain": 3668.5004992485046,
      "importance_split": 153
    },
    {
      "feature": "C12",
      "importance_gain": 3486.6622545719147,
      "importance_split": 122
    },
    {
      "feature": "D6",
      "importance_gain": 3328.8944602012634,
      "importance_split": 135
    },
    {
      "feature": "D13",
      "importance_gain": 2980.017986536026,
      "importance_split": 155
    },
    {
      "feature": "D9",
      "importance_gain": 2972.190711736679,
      "importance_split": 202
    },
    {
      "feature": "card4",
      "importance_gain": 2563.3842248916626,
      "importance_split": 142
    },
    {
      "feature": "ProductCD",
      "importance_gain": 1453.4344911575317,
      "importance_split": 44
    },
    {
      "feature": "DeviceType",
      "importance_gain": 1263.403451681137,
      "importance_split": 73
    },
    {
      "feature": "M3",
      "importance_gain": 1139.1455459594727,
      "importance_split": 49
    },
    {
      "feature": "M9",
      "importance_gain": 496.12000799179077,
      "importance_split": 32
    },
    {
      "feature": "M8",
      "importance_gain": 435.3810691833496,
      "importance_split": 32
    },
    {
      "feature": "M7",
      "importance_gain": 400.5264916419983,
      "importance_split": 27
    },
    {
      "feature": "M2",
      "importance_gain": 307.69134044647217,
      "importance_split": 19
    },
    {
      "feature": "M1",
      "importance_gain": 152.07908964157104,
      "importance_split": 8
    }
  ],
  "metrics": {
    "F1": 0.5224706663709939,
    "PR-AUC": 0.5508057463474152,
    "ROC-AUC": 0.9171894861929567,
    "Recall@FPR<=0.1%": 0.23149236192714454,
    "Recall@FPR<=1%": 0.4500587544065805,
    "confusion_matrix": {
      "fn": 1373,
      "fp": 784,
      "tn": 71150,
      "tp": 1180
    },
    "precision": 0.6008146639511202,
    "selected_threshold": 0.20409261910516727
  },
  "model": "Model E (Entity Features Final)",
  "row_counts": {
    "train": 437511,
    "validation": 74487
  }
}
