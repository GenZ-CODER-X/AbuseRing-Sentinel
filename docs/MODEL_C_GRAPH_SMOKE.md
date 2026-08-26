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
    "rel_event_row_count": 200,
    "same_timestamp_batching": true,
    "train_rel_event_row_count": 120,
    "validation_rel_event_row_count": 80
  },
  "experiment": "model_c_graph",
  "feature_importance": [
    {
      "feature": "C1",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "C10",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "C11",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "C12",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "C13",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "C14",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "C2",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "C4",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "C5",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "C6",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "C7",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "C8",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "C9",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D1",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D10",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D11",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D12",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D13",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D14",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D15",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D2",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D3",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D4",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D5",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D6",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D7",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D8",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "D9",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "DeviceInfo",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "DeviceType",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "M1",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "M2",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "M3",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "M4",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "M5",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "M6",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "M7",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "M8",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "M9",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "P_emaildomain",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "ProductCD",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "R_emaildomain",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "TransactionAmt",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "TransactionDT",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "addr1",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "addr2",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "card1",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "card2",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "card3",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "card4",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "card5",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "card6",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "dist1",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "dist2",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "tb_addr1_card1_prior_unique_count",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "tb_card1_device_prior_unique_count",
      "importance_gain": 0.0,
      "importance_split": 0
    }
  ],
  "graph_rel_feature_importance": [
    {
      "feature": "tb_addr1_card1_prior_unique_count",
      "importance_gain": 0.0,
      "importance_split": 0
    },
    {
      "feature": "tb_card1_device_prior_unique_count",
      "importance_gain": 0.0,
      "importance_split": 0
    }
  ],
  "metrics": {
    "F1": 0.11764705882341868,
    "PR-AUC": 0.0625,
    "ROC-AUC": 0.5,
    "Recall@FPR<=0.1%": 0.0,
    "Recall@FPR<=1%": 0.0,
    "confusion_matrix": {
      "fn": 0,
      "fp": 75,
      "tn": 0,
      "tp": 5
    },
    "precision": 0.0625,
    "selected_threshold": 1.0000000036274914e-15
  },
  "model": "Model C-Graph (Graph Relationship)",
  "row_counts": {
    "train": 120,
    "validation": 80
  }
}
