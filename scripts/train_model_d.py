#!/usr/bin/env python3
"""Train Model D: Model B plus five UID features (card1-6 + addr1)."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import sklearn

import train_model_b as model_b
from uid_features import (
    UID_FEATURES,
    UID_INPUT_COLUMNS,
    build_uid_features,
    prepare_uid_events,
    uid_feature_specification,
)
from validation_boundaries import LABEL_FIELD, PARTITION_BOUNDARIES, partition_filter, validate_boundaries


EXPERIMENT = "model_d"
FEATURES = model_b.FEATURES + UID_FEATURES
CATEGORICAL_FEATURES = model_b.CATEGORICAL_FEATURES
NUMERIC_FEATURES = tuple(feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models/model_d"))
    parser.add_argument("--report", type=Path, default=Path("docs/MODEL_D_BASELINE.md"))
    parser.add_argument("--model-a-metrics", type=Path, default=Path("artifacts/models/model_a/validation_metrics.json"))
    parser.add_argument("--model-b-metrics", type=Path, default=Path("artifacts/models/model_b/validation_metrics.json"))
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found at {path}")


def scan_selected_columns(transaction_path: Path, identity_path: Path) -> pl.LazyFrame:
    """Lazily scan Model B fields, event key, and raw device fields only."""
    transaction_columns = [feature for feature in model_b.FEATURES if feature not in {"DeviceInfo", "DeviceType"}]
    transactions = pl.scan_csv(transaction_path).select("TransactionID", *transaction_columns, LABEL_FIELD)
    identities = pl.scan_csv(identity_path).select("TransactionID", "DeviceInfo", "DeviceType")
    frame = transactions.join(identities, on="TransactionID", how="left").select("TransactionID", *model_b.FEATURES, LABEL_FIELD)
    available = set(frame.collect_schema().names())
    required = set(model_b.FEATURES) | {LABEL_FIELD}
    missing = required.difference(available)
    if missing:
        raise ValueError(f"Input CSVs lack required columns: {sorted(missing)}")
    return frame


def collect_partition(frame: pl.LazyFrame, boundary: Any) -> pl.DataFrame:
    return frame.filter(partition_filter(boundary)).collect(engine="streaming")


_missing = -1
_unknown = 0


def training_category_mappings(train: pl.DataFrame) -> dict[str, dict[Any, int]]:
    """Fit deterministic positive categorical codes from train rows only."""
    mappings: dict[str, dict[Any, int]] = {}
    for feature in CATEGORICAL_FEATURES:
        values = train.get_column(feature).drop_nulls().unique().sort().to_list()
        mappings[feature] = {value: index for index, value in enumerate(values, start=1)}
    return mappings


def encode_categoricals(frame: pl.DataFrame, mappings: dict[str, dict[Any, int]]) -> pl.DataFrame:
    """Use -1 for missing and reserved code 0 for validation-only categories."""
    return frame.with_columns(
        *[
            pl.when(pl.col(feature).is_null())
            .then(pl.lit(_missing))
            .otherwise(pl.col(feature).replace_strict(mappings[feature], default=_unknown))
            .cast(pl.Int32)
            .alias(feature)
            for feature in CATEGORICAL_FEATURES
        ]
    )


def attach_uid_features(train: pl.DataFrame, validation: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Build state over train then validation events; labels never enter this function."""
    # Select the columns needed for UID: TransactionID, TransactionDT, card1-6, addr1, TransactionAmt
    uid_train = train.select(*UID_INPUT_COLUMNS)
    uid_validation = validation.select(*UID_INPUT_COLUMNS)
    uid_events = pl.concat([uid_train, uid_validation], how="vertical")
    uid_features = build_uid_features(prepare_uid_events(uid_events))
    evidence = uid_feature_specification() | {
        "uid_event_row_count": uid_events.height,
        "train_uid_event_row_count": uid_train.height,
        "validation_uid_event_row_count": uid_validation.height,
        "label_column_passed_to_graph_builder": False,
        "same_timestamp_batching": True,
    }
    return (
        train.join(uid_features, on="TransactionID", how="left"),
        validation.join(uid_features, on="TransactionID", how="left"),
        evidence,
    )


def matrix_and_target(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Keep numeric nulls as NaN when producing the LightGBM float32 matrix."""
    matrix = frame.select(*[pl.col(feature).cast(pl.Float32) for feature in FEATURES]).to_numpy()
    target = frame.get_column(LABEL_FIELD).cast(pl.Int8).to_numpy()
    return matrix, target


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def feature_importance_rows(model: lgb.Booster) -> list[dict[str, Any]]:
    """Extract feature importance from a LightGBM Booster."""
    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    return sorted(
        [{"feature": feature, "importance_gain": float(gain_value), "importance_split": int(split_value)} for feature, gain_value, split_value in zip(FEATURES, gain, split)],
        key=lambda row: (-row["importance_gain"], row["feature"]),
    )


def write_feature_importance(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "importance_gain", "importance_split"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    require_file(args.transaction, "Transaction CSV")
    require_file(args.identity, "Identity CSV")

    print(f"Machine: {platform.node()} ({platform.processor()})")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python {platform.python_version()}")

    # Validate boundaries
    validate_boundaries()

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    lazy_frame = scan_selected_columns(args.transaction, args.identity)

    # Split into train and validation
    train_boundary = next(b for b in PARTITION_BOUNDARIES if b.name == "train")
    val_boundary = next(b for b in PARTITION_BOUNDARIES if b.name == "validation")
    train_data = collect_partition(lazy_frame, train_boundary)
    validation_data = collect_partition(lazy_frame, val_boundary)

    print(f"Train rows: {train_data.height}")
    print(f"Validation rows: {validation_data.height}")

    # Attach UID features
    train_with_uid, validation_with_uid, evidence = attach_uid_features(train_data, validation_data)

    # Compute categorical mappings from training data only
    cat_mappings = training_category_mappings(train_with_uid)
    # Encode categoricals in both train and validation
    train_encoded = encode_categoricals(train_with_uid, cat_mappings)
    validation_encoded = encode_categoricals(validation_with_uid, cat_mappings)

    # Feature and target matrices
    X_train, y_train = matrix_and_target(train_encoded)
    X_valid, y_valid = matrix_and_target(validation_encoded)

    # LightGBM dataset
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

    # Parameters (same as Model B)
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "seed": 42,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    # Train model
    print("Training Model D...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)],
    )

    # Predict and evaluate
    valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score

    roc_auc = roc_auc_score(y_valid, valid_pred)
    precision, recall, thresholds = precision_recall_curve(y_valid, valid_pred)
    pr_auc = average_precision_score(y_valid, valid_pred)
    # Find threshold for Recall@FPR<=1%
    fpr, tpr, thr = sklearn.metrics.roc_curve(y_valid, valid_pred)
    # Find the threshold where FPR <= 0.01, maximize TPR
    idx = np.where(fpr <= 0.01)[0]
    if len(idx) > 0:
        best_idx = idx[np.argmax(tpr[idx])]
        recall_at_fpr_1 = tpr[best_idx]
        threshold_1 = thr[best_idx]
    else:
        recall_at_fpr_1 = 0.0
        threshold_1 = 0.5
    # Recall@FPR<=0.1%
    idx = np.where(fpr <= 0.001)[0]
    if len(idx) > 0:
        best_idx = idx[np.argmax(tpr[idx])]
        recall_at_fpr_01 = tpr[best_idx]
        threshold_01 = thr[best_idx]
    else:
        recall_at_fpr_01 = 0.0
        threshold_01 = 0.5
    # F1 and precision at threshold 0.5 (or we can use the threshold that maximizes F1)
    # Let's compute F1 for each threshold and pick the best
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    idx_best_f1 = np.argmax(f1_scores)
    best_threshold = thresholds[idx_best_f1]
    best_f1 = f1_scores[idx_best_f1]
    best_precision = precision[idx_best_f1]
    best_recall = recall[idx_best_f1]

    # Confusion matrix at best_threshold
    y_pred = (valid_pred >= best_threshold).astype(int)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_valid, y_pred).ravel()

    # Feature importance
    importance = feature_importance_rows(model)

    # UID feature importance
    uid_importance = [row for row in importance if row["feature"] in UID_FEATURES]

    # Prepare results
    results = {
        "model": "Model D (UID)",
        "experiment": EXPERIMENT,
        "row_counts": {
            "train": int(train_data.height),
            "validation": int(validation_data.height),
        },
        "metrics": {
            "PR-AUC": float(pr_auc),
            "ROC-AUC": float(roc_auc),
            "Recall@FPR<=1%": float(recall_at_fpr_1),
            "Recall@FPR<=0.1%": float(recall_at_fpr_01),
            "F1": float(best_f1),
            "precision": float(best_precision),
            "selected_threshold": float(best_threshold),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        },
        "feature_importance": importance,
        "uid_feature_importance": uid_importance,
        "evidence": evidence,
    }

    # Write metrics
    report_path = args.report
    write_json(report_path, results)
    print(f"Validation metrics written to {report_path}")

    # Write feature list
    feature_list_path = args.output_dir / "feature_list.json"
    write_json(feature_list_path, {
        "features": list(FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
    })
    print(f"Feature list written to {feature_list_path}")

    # Write model
    model_path = args.output_dir / "model.lgb"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Print summary
    print("\n=== Model D Validation Metrics ===")
    for k, v in results["metrics"].items():
        if k != "confusion_matrix":
            print(f"{k}: {v}")
    print("Confusion Matrix:")
    print(results["metrics"]["confusion_matrix"])
    print("\nTop 5 UID Features by Gain:")
    for row in sorted(uid_importance, key=lambda x: -x["importance_gain"])[:5]:
        print(f"  {row['feature']}: gain={row['importance_gain']:.6f}, split={row['importance_split']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"train_model_d.py: {error}", file=sys.stderr)
        sys.exit(1)
