{
  "evidence": {
    "causality": "Features for a TransactionDT batch read state from strictly earlier TransactionDT values; all same-timestamp rows update state only after feature emission.",
    "features": [
      "tb_card1_device_prior_unique_count",
      "tb_addr1_card1_prior_unique_count"
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
      "DeviceInfo",
      "addr1"
    ],
    "graph_input_columns": [
      "TransactionID",
      "TransactionDT",
      "card1",
      "DeviceInfo",
      "addr1"
    ],
    "label_column_passed_to_graph_builder": false,
    "rel_event_row_count": 511998,
    "same_timestamp_batching": true,
    "train_rel_event_row_count": 437511,
    "validation_rel_event_row_count": 74487
  },
  "experiment": "model_c_graph",
  "feature_importance": [
    {
      "feature": "C1",
      "importance_gain": 84076.38519978523,
      "importance_split": 741
    },
    {
      "feature": "C14",
      "importance_gain": 53371.4634206295,
      "importance_split": 472
    },
    {
      "feature": "C13",
      "importance_gain": 35532.09885787964,
      "importance_split": 872
    },
    {
      "feature": "card1",
      "importance_gain": 35221.19918727875,
      "importance_split": 2042
    },
    {
      "feature": "TransactionDT",
      "importance_gain": 33807.40720701218,
      "importance_split": 2053
    },
    {
      "feature": "C8",
      "importance_gain": 33460.890568733215,
      "importance_split": 220
    },
    {
      "feature": "TransactionAmt",
      "importance_gain": 32667.80506825447,
      "importance_split": 1875
    },
    {
      "feature": "D2",
      "importance_gain": 30733.476839780807,
      "importance_split": 975
    },
    {
      "feature": "C4",
      "importance_gain": 30613.71723484993,
      "importance_split": 96
    },
    {
      "feature": "card2",
      "importance_gain": 29624.258108377457,
      "importance_split": 1707
    },
    {
      "feature": "addr1",
      "importance_gain": 26531.42524123192,
      "importance_split": 1409
    },
    {
      "feature": "D3",
      "importance_gain": 26324.654405117035,
      "importance_split": 512
    },
    {
      "feature": "D15",
      "importance_gain": 25229.454243183136,
      "importance_split": 1245
    },
    {
      "feature": "tb_addr1_card1_prior_unique_count",
      "importance_gain": 22822.026376724243,
      "importance_split": 1518
    },
    {
      "feature": "C7",
      "importance_gain": 22547.20224761963,
      "importance_split": 46
    },
    {
      "feature": "D8",
      "importance_gain": 20643.36232793331,
      "importance_split": 772
    },
    {
      "feature": "D10",
      "importance_gain": 20562.10203933716,
      "importance_split": 830
    },
    {
      "feature": "C5",
      "importance_gain": 19520.00393986702,
      "importance_split": 270
    },
    {
      "feature": "R_emaildomain",
      "importance_gain": 19089.07354950905,
      "importance_split": 403
    },
    {
      "feature": "D4",
      "importance_gain": 16573.766882419586,
      "importance_split": 864
    },
    {
      "feature": "P_emaildomain",
      "importance_gain": 15579.741013288498,
      "importance_split": 836
    },
    {
      "feature": "D1",
      "importance_gain": 14967.98704123497,
      "importance_split": 639
    },
    {
      "feature": "C2",
      "importance_gain": 13947.44232416153,
      "importance_split": 575
    },
    {
      "feature": "C11",
      "importance_gain": 13862.652635097504,
      "importance_split": 439
    },
    {
      "feature": "card6",
      "importance_gain": 11988.493672132492,
      "importance_split": 217
    },
    {
      "feature": "dist1",
      "importance_gain": 11830.676273345947,
      "importance_split": 776
    },
    {
      "feature": "C6",
      "importance_gain": 11578.436150550842,
      "importance_split": 482
    },
    {
      "feature": "card5",
      "importance_gain": 11531.672435998917,
      "importance_split": 657
    },
    {
      "feature": "DeviceInfo",
      "importance_gain": 10978.413621664047,
      "importance_split": 641
    },
    {
      "feature": "C9",
      "importance_gain": 10219.001169681549,
      "importance_split": 357
    },
    {
      "feature": "M5",
      "importance_gain": 9868.389161586761,
      "importance_split": 315
    },
    {
      "feature": "M4",
      "importance_gain": 9385.155638694763,
      "importance_split": 304
    },
    {
      "feature": "D5",
      "importance_gain": 8916.682946920395,
      "importance_split": 432
    },
    {
      "feature": "card3",
      "importance_gain": 7748.620539665222,
      "importance_split": 301
    },
    {
      "feature": "D11",
      "importance_gain": 7358.494227647781,
      "importance_split": 479
    },
    {
      "feature": "tb_card1_device_prior_unique_count",
      "importance_gain": 6799.319659471512,
      "importance_split": 518
    },
    {
      "feature": "dist2",
      "importance_gain": 5491.805701494217,
      "importance_split": 510
    },
    {
      "feature": "D14",
      "importance_gain": 5483.611860990524,
      "importance_split": 350
    },
    {
      "feature": "M6",
      "importance_gain": 5226.423583984375,
      "importance_split": 254
    },
    {
      "feature": "D6",
      "importance_gain": 4354.134313583374,
      "importance_split": 234
    },
    {
      "feature": "card4",
      "importance_gain": 3769.541923046112,
      "importance_split": 268
    },
    {
      "feature": "D9",
      "importance_gain": 3511.103577852249,
      "importance_split": 358
    },
    {
      "feature": "C12",
      "importance_gain": 3509.187882423401,
      "importance_split": 171
    },
    {
      "feature": "D13",
      "importance_gain": 3301.6482214927673,
      "importance_split": 236
    },
    {
      "feature": "C10",
      "importance_gain": 3073.2970900535583,
      "importance_split": 175
    },
    {
      "feature": "D12",
      "importance_gain": 3010.3872821331024,
      "importance_split": 234
    },
    {
      "feature": "D7",
      "importance_gain": 2863.316751241684,
      "importance_split": 103
    },
    {
      "feature": "addr2",
      "importance_gain": 2257.3893699645996,
      "importance_split": 29
    },
    {
      "feature": "ProductCD",
      "importance_gain": 1382.9916331768036,
      "importance_split": 55
    },
    {
      "feature": "M3",
      "importance_gain": 1208.2702934741974,
      "importance_split": 65
    },
    {
      "feature": "DeviceType",
      "importance_gain": 1166.0915157794952,
      "importance_split": 103
    },
    {
      "feature": "M7",
      "importance_gain": 956.0701541900635,
      "importance_split": 86
    },
    {
      "feature": "M9",
      "importance_gain": 618.3742485046387,
      "importance_split": 57
    },
    {
      "feature": "M2",
      "importance_gain": 477.4525671005249,
      "importance_split": 30
    },
    {
      "feature": "M8",
      "importance_gain": 340.0118713378906,
      "importance_split": 37
    },
    {
      "feature": "M1",
      "importance_gain": 132.24433016777039,
      "importance_split": 11
    }
  ],
  "graph_rel_feature_importance": [
    {
      "feature": "tb_addr1_card1_prior_unique_count",
      "importance_gain": 22822.026376724243,
      "importance_split": 1518
    },
    {
      "feature": "tb_card1_device_prior_unique_count",
      "importance_gain": 6799.319659471512,
      "importance_split": 518
    }
  ],
  "metrics": {
    "F1": 0.548868778280055,
    "PR-AUC": 0.5696927433540214,
    "ROC-AUC": 0.9231989281291721,
    "Recall@FPR<=0.1%": 0.21425773599686643,
    "Recall@FPR<=1%": 0.48217783000391695,
    "confusion_matrix": {
      "fn": 1340,
      "fp": 654,
      "tn": 71280,
      "tp": 1213
    },
    "precision": 0.6497054097482592,
    "selected_threshold": 0.22438126776414322
  },
  "model": "Model C-Graph (Graph Relationship)",
  "row_counts": {
    "train": 437511,
    "validation": 74487
  }
}
