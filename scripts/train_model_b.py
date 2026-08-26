#!/usr/bin/env python3
"""Train Model B: Model A plus four raw categorical device/card fields.

The experiment uses only the train and validation definitions imported from
validation_boundaries. It does not materialize or evaluate another partition.
"""

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

import train_model_a as model_a
from validation_boundaries import LABEL_FIELD, PARTITION_BOUNDARIES, partition_filter, validate_boundaries


EXPERIMENT = "model_b"
ADDED_RAW_CATEGORICAL_FEATURES = ("DeviceInfo", "DeviceType", "card4", "card6")
FEATURES = model_a.FEATURES + ADDED_RAW_CATEGORICAL_FEATURES
CATEGORICAL_FEATURES = model_a.CATEGORICAL_FEATURES + ADDED_RAW_CATEGORICAL_FEATURES
NUMERIC_FEATURES = tuple(feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models/model_b"))
    parser.add_argument("--report", type=Path, default=Path("docs/MODEL_B_BASELINE.md"))
    parser.add_argument(
        "--model-a-metrics",
        type=Path,
        default=Path("artifacts/models/model_a/validation_metrics.json"),
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV does not exist or is not a file: {path}")


def selected_boundaries() -> tuple[Any, Any]:
    """Get train and validation exclusively from the locked source of truth."""
    validate_boundaries(PARTITION_BOUNDARIES)
    by_name = {boundary.name: boundary for boundary in PARTITION_BOUNDARIES}
    return by_name["train"], by_name["validation"]


def scan_selected_columns(transaction_path: Path, identity_path: Path) -> pl.LazyFrame:
    """Lazily project Model B columns and left-join raw identity device fields."""
    transaction_columns = [feature for feature in FEATURES if feature not in {"DeviceInfo", "DeviceType"}]
    transactions = pl.scan_csv(transaction_path).select("TransactionID", *transaction_columns, LABEL_FIELD)
    identities = pl.scan_csv(identity_path).select("TransactionID", "DeviceInfo", "DeviceType")
    frame = transactions.join(identities, on="TransactionID", how="left").select(*FEATURES, LABEL_FIELD)
    available = set(frame.collect_schema().names())
    required = set(FEATURES) | {LABEL_FIELD}
    missing = required.difference(available)
    if missing:
        raise ValueError(f"Input CSVs lack required columns: {sorted(missing)}")
    return frame


def collect_partition(frame: pl.LazyFrame, boundary: Any) -> pl.DataFrame:
    return frame.filter(partition_filter(boundary)).collect(engine="streaming")


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
            .then(pl.lit(-1))
            .otherwise(pl.col(feature).replace_strict(mappings[feature], default=0))
            .cast(pl.Int32)
            .alias(feature)
            for feature in CATEGORICAL_FEATURES
        ]
    )


def matrix_and_target(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Keep numeric nulls as NaN when producing the LightGBM float32 matrix."""
    matrix = frame.select(*[pl.col(feature).cast(pl.Float32) for feature in FEATURES]).to_numpy()
    target = frame.get_column(LABEL_FIELD).cast(pl.Int8).to_numpy()
    return matrix, target


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_feature_importance(path: Path, model: lgb.LGBMClassifier) -> None:
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    rows = sorted(
        (
            {"feature": feature, "importance_gain": float(feature_gain), "importance_split": int(feature_split)}
            for feature, feature_gain, feature_split in zip(FEATURES, gain, split)
        ),
        key=lambda row: (-row["importance_gain"], row["feature"]),
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "importance_gain", "importance_split"])
        writer.writeheader()
        writer.writerows(rows)


def comparison(model_a_metrics: dict[str, Any], model_b_metrics: dict[str, Any]) -> dict[str, Any]:
    metric_names = (
        "pr_auc_average_precision",
        "roc_auc",
        "precision_at_selected_threshold",
        "recall_at_selected_threshold",
        "f1_at_selected_threshold",
    )
    return {
        name: {
            "model_a": model_a_metrics[name],
            "model_b": model_b_metrics[name],
            "difference_b_minus_a": model_b_metrics[name] - model_a_metrics[name],
        }
        for name in metric_names
    }


def write_report(path: Path, model_b_metrics: dict[str, Any], model_a_metrics: dict[str, Any], params: dict[str, Any], memory: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compare = comparison(model_a_metrics, model_b_metrics)
    matrix = model_b_metrics["confusion_matrix"]["rows"]
    path.write_text(
        f"# Model B baseline\n\n"
        f"Model B is a controlled extension of Model A: it adds only raw categorical `DeviceInfo`, `DeviceType`, `card4`, and `card6`. All Model A features, locked chronological boundaries, fixed 500-iteration LightGBM configuration, train-only categorical mappings, and validation-only threshold methodology are retained.\n\n"
        f"No graph, composite multi-transaction, target-encoded, historical, velocity, frequency, V-family, or other derived features are included. Numeric nulls remain `NaN`; categorical missing values use `-1`, and validation-only categorical values use reserved code `0`.\n\n"
        f"## A vs. B validation comparison\n\n"
        f"| Metric | Model A | Model B | B − A |\n| --- | ---: | ---: | ---: |\n"
        f"| PR-AUC / average precision | {compare['pr_auc_average_precision']['model_a']:.12f} | {compare['pr_auc_average_precision']['model_b']:.12f} | {compare['pr_auc_average_precision']['difference_b_minus_a']:+.12f} |\n"
        f"| ROC-AUC | {compare['roc_auc']['model_a']:.12f} | {compare['roc_auc']['model_b']:.12f} | {compare['roc_auc']['difference_b_minus_a']:+.12f} |\n"
        f"| Precision at selected threshold | {compare['precision_at_selected_threshold']['model_a']:.12f} | {compare['precision_at_selected_threshold']['model_b']:.12f} | {compare['precision_at_selected_threshold']['difference_b_minus_a']:+.12f} |\n"
        f"| Recall at selected threshold | {compare['recall_at_selected_threshold']['model_a']:.12f} | {compare['recall_at_selected_threshold']['model_b']:.12f} | {compare['recall_at_selected_threshold']['difference_b_minus_a']:+.12f} |\n"
        f"| F1 at selected threshold | {compare['f1_at_selected_threshold']['model_a']:.12f} | {compare['f1_at_selected_threshold']['model_b']:.12f} | {compare['f1_at_selected_threshold']['difference_b_minus_a']:+.12f} |\n\n"
        f"## Model B operating point\n\n"
        f"Threshold `{model_b_metrics['threshold_selection']['selected_threshold']:.12f}` maximizes validation recall under FPR ≤1%, with ties resolved by higher threshold. It yields precision {model_b_metrics['precision_at_selected_threshold']:.12f}, recall {model_b_metrics['recall_at_selected_threshold']:.12f}, F1 {model_b_metrics['f1_at_selected_threshold']:.12f}, and FPR {model_b_metrics['threshold_selection']['selected_threshold_fpr']:.12f}. Recall at FPR ≤0.1% is {model_b_metrics['recall_at_fpr_le_point_1_percent']['recall']:.12f}.\n\n"
        f"Confusion matrix `[[TN, FP], [FN, TP]]`: `[[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]]`.\n\n"
        f"The fixed model configuration retains `n_estimators={params['n_estimators']}`, `learning_rate={params['learning_rate']}`, `num_leaves={params['num_leaves']}`, deterministic execution, no row/feature subsampling, and train-derived `scale_pos_weight={params['scale_pos_weight']:.12f}`. Selected train/validation frame and matrix sizes are {memory['train_frame_bytes']:,}/{memory['train_matrix_bytes']:,} and {memory['validation_frame_bytes']:,}/{memory['validation_matrix_bytes']:,} bytes.\n"
    )


def main() -> None:
    args = parse_args()
    require_file(args.transaction, "Transaction")
    require_file(args.identity, "Identity")
    if not args.model_a_metrics.is_file():
        raise FileNotFoundError(f"Model A validation metrics are required for comparison: {args.model_a_metrics}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_boundary, validation_boundary = selected_boundaries()
    transactions = scan_selected_columns(args.transaction, args.identity)
    train = collect_partition(transactions, train_boundary)
    validation = collect_partition(transactions, validation_boundary)
    if train.height == 0 or validation.height == 0:
        raise ValueError("Train and validation partitions must both contain rows")

    mappings = training_category_mappings(train)
    category_cardinalities = {feature: len(mapping) for feature, mapping in mappings.items()}
    train_frame_bytes = train.estimated_size()
    validation_frame_bytes = validation.estimated_size()
    train_matrix, train_target = matrix_and_target(encode_categoricals(train, mappings))
    validation_matrix, validation_target = matrix_and_target(encode_categoricals(validation, mappings))
    del train, validation
    train_positive = int(train_target.sum())
    train_negative = int(train_target.size - train_positive)
    params = model_a.predeclared_model_parameters(train_negative / train_positive)
    model = lgb.LGBMClassifier(**params)
    model.fit(train_matrix, train_target, feature_name=list(FEATURES), categorical_feature=list(CATEGORICAL_FEATURES))
    metrics = model_a.validation_metrics(validation_target, model.predict_proba(validation_matrix)[:, 1])
    model_a_metrics = json.loads(args.model_a_metrics.read_text())
    if model_a_metrics.get("evaluation_partition") != "validation":
        raise ValueError("Model A comparison artifact must contain validation metrics only")

    memory = {
        "train_frame_bytes": train_frame_bytes,
        "train_matrix_bytes": train_matrix.nbytes,
        "validation_frame_bytes": validation_frame_bytes,
        "validation_matrix_bytes": validation_matrix.nbytes,
    }
    metadata = {
        "experiment": EXPERIMENT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "transaction_source": str(args.transaction.resolve()),
        "identity_source": str(args.identity.resolve()),
        "added_raw_categorical_features": list(ADDED_RAW_CATEGORICAL_FEATURES),
        "train_boundary": {"min": train_boundary.start, "max": train_boundary.end},
        "validation_boundary": {"min": validation_boundary.start, "max": validation_boundary.end},
        "train_row_count": int(train_target.size),
        "validation_row_count": int(validation_target.size),
        "train_fraud_count": train_positive,
        "validation_fraud_count": int(validation_target.sum()),
        "random_seed": model_a.RANDOM_SEED,
        "model_parameters": params,
        "categorical_handling": {
            "features": list(CATEGORICAL_FEATURES),
            "missing_code": -1,
            "unknown_validation_code": 0,
            "category_cardinalities_from_train_only": category_cardinalities,
        },
        "numeric_missing_values": "Preserved as NaN for LightGBM native missing-value handling.",
        "memory": memory,
        "package_versions": {"lightgbm": lgb.__version__, "numpy": np.__version__, "polars": pl.__version__, "scikit_learn": sklearn.__version__},
        "execution": {"python": platform.python_version(), "platform": platform.platform(), "num_threads": model_a.NUM_THREADS, "training_iterations_fixed_before_validation": model_a.TRAINING_ITERATIONS},
    }
    write_json(args.output_dir / "config_metadata.json", metadata)
    write_json(args.output_dir / "feature_list.json", {"features": list(FEATURES), "categorical_features": list(CATEGORICAL_FEATURES), "numeric_features": list(NUMERIC_FEATURES)})
    write_json(args.output_dir / "validation_metrics.json", metrics)
    write_json(args.output_dir / "model_a_vs_model_b_validation.json", comparison(model_a_metrics, metrics))
    write_json(args.output_dir / "reproducibility.json", {"random_seed": model_a.RANDOM_SEED, "package_versions": metadata["package_versions"], "python": metadata["execution"]["python"], "platform": metadata["execution"]["platform"], "fixed_training_iterations": model_a.TRAINING_ITERATIONS, "no_random_split_or_shuffle": True, "no_row_or_feature_subsampling": True})
    model.booster_.save_model(str(args.output_dir / "model.txt"))
    write_feature_importance(args.output_dir / "feature_importance.csv", model)
    write_report(args.report, metrics, model_a_metrics, params, memory)
    print(json.dumps({"model_b_validation_metrics": metrics, "model_a_vs_model_b": comparison(model_a_metrics, metrics)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"train_model_b.py: {error}", file=sys.stderr)
        raise
