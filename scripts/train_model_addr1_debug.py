#!/usr/bin/env python3
"""Train Model Addr1: Model B plus five addr1 temporal features.

The experiment uses only the train and validation definitions imported from
validation_boundaries. It does not materialize or evaluate another partition.
"""

from __future__ import annotations

import argparse
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
import addr_features as af
from validation_boundaries import LABEL_FIELD, PARTITION_BOUNDARIES, partition_filter, validate_boundaries


EXPERIMENT = "model_addr1"
ADDED_RAW_CATEGORICAL_FEATURES = ("DeviceInfo", "DeviceType", "card4", "card6")
ADDR_FEATURES = (
    "tb_addr_prior_count",
    "tb_addr_amt_mean",
    "tb_addr_amt_std",
    "tb_addr_recency",
    "tb_addr_amt_zscore",
)
FEATURES = model_a.FEATURES + ADDED_RAW_CATEGORICAL_FEATURES + ADDR_FEATURES
CATEGORICAL_FEATURES = model_a.CATEGORICAL_FEATURES + ADDED_RAW_CATEGORICAL_FEATURES
NUMERIC_FEATURES = tuple(feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models/model_addr1"))
    parser.add_argument("--report", type=Path, default=Path("docs/MODEL_ADDR1_BASELINE.md"))
    parser.add_argument(
        "--model-b-metrics",
        type=Path,
        default=Path("artifacts/models/model_b/validation_metrics.json"),
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
    """Lazily project columns needed for Model B raw features and left-join raw identity device fields."""
    # Model B raw features: all FEATURES except those that come from identity (DeviceInfo, DeviceType)
    raw_features = [f for f in FEATURES if f not in {"DeviceInfo", "DeviceType"}]
    transactions = pl.scan_csv(transaction_path).select("TransactionID", *raw_features, LABEL_FIELD)
    identities = pl.scan_csv(identity_path).select("TransactionID", "DeviceInfo", "DeviceType")
    frame = transactions.join(identities, on="TransactionID", how="left")
    # Ensure we have all FEATURES plus label
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def comparison(model_b_metrics: dict[str, Any], model_addr1_metrics: dict[str, Any]) -> dict[str, Any]:
    metric_names = (
        "pr_auc_average_precision",
        "roc_auc",
        "precision_at_selected_threshold",
        "recall_at_selected_threshold",
        "f1_at_selected_threshold",
    )
    return {
        name: {
            "model_b": model_b_metrics[name],
            "model_addr1": model_addr1_metrics[name],
            "difference_addr1_minus_b": model_addr1_metrics[name] - model_b_metrics[name],
        }
        for name in metric_names
    }


def write_report(
    path: Path,
    model_addr1_metrics: dict[str, Any],
    model_b_metrics: dict[str, Any],
    params: dict[str, Any],
    memory: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compare = comparison(model_b_metrics, model_addr1_metrics)
    matrix = model_addr1_metrics["confusion_matrix"]["rows"]
    path.write_text(
        f"# Model Addr1 baseline\n\n"
        f"Model Addr1 is a controlled extension of Model B: it adds only five temporal addr1 features. All Model B features, locked chronological boundaries, fixed 500-iteration LightGBM configuration, train-only categorical mappings, and validation-only threshold methodology are retained.\n\n"
        f"No graph, composite multi-transaction, target-encoded, historical, velocity, frequency, V-family, or other derived features are included. Numeric nulls remain `NaN`; categorical missing values use `-1`, and validation-only categorical values use reserved code `0`.\n\n"
        f"## B vs. Addr1 validation comparison\n\n"
        f"| Metric | Model B | Model Addr1 | Addr1 − B |\n| --- | ---: | ---: | ---: |\n"
        f"| PR-AUC / average precision | {compare['pr_auc_average_precision']['model_b']:.12f} | {compare['pr_auc_average_precision']['model_addr1']:.12f} | {compare['pr_auc_average_precision']['difference_addr1_minus_b']:+.12f} |\n"
        f"| ROC-AUC | {compare['roc_auc']['model_b']:.12f} | {compare['roc_auc']['model_addr1']:.12f} | {compare['roc_auc']['difference_addr1_minus_b']:+.12f} |\n"
        f"| Precision at selected threshold | {compare['precision_at_selected_threshold']['model_b']:.12f} | {compare['precision_at_selected_threshold']['model_addr1']:.12f} | {compare['precision_at_selected_threshold']['difference_addr1_minus_b']:+.12f} |\n"
        f"| Recall at selected threshold | {compare['recall_at_selected_threshold']['model_b']:.12f} | {compare['recall_at_selected_threshold']['model_addr1']:.12f} | {compare['recall_at_selected_threshold']['difference_addr1_minus_b']:+.12f} |\n"
        f"| F1 at selected threshold | {compare['f1_at_selected_threshold']['model_b']:.12f} | {compare['f1_at_selected_threshold']['model_addr1']:.12f} | {compare['f1_at_selected_threshold']['f1_at_selected_threshold']:+.12f} |\n\n"
        f"## Model Addr1 operating point\n\n"
        f"Threshold `{model_addr1_metrics['threshold_selection']['selected_threshold']:.12f}` maximizes validation recall under FPR ≤1%, with ties resolved by higher threshold. It yields precision {model_addr1_metrics['precision_at_selected_threshold']:.12f}, recall {model_addr1_metrics['recall_at_selected_threshold']:.12f}, F1 {model_addr1_metrics['f1_at_selected_threshold']:.12f}, and FPR {model_addr1_metrics['threshold_selection']['selected_threshold_fpr']:.12f}. Recall at FPR ≤0.1% is {model_addr1_metrics['recall_at_fpr_le_point_1_percent']['recall']:.12f}.\n\n"
        f"Confusion matrix `[[TN, FP], [FN, TP]]`: `[[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]]`.\n\n"
        f"The fixed model configuration retains `n_estimators={params['n_estimators']}`, `learning_rate={params['learning_rate']}`, `num_leaves={params['num_leaves']}`, deterministic execution, no row/feature subsampling, and train-derived `scale_pos_weight={params['scale_pos_weight']:.12f}`. Selected train/validation frame and matrix sizes are {memory['train_frame_bytes']:,}/{memory['train_matrix_bytes']:,} and {memory['validation_frame_bytes']:,}/{memory['validation_matrix_bytes']:,} bytes.\n"
    )


def main() -> None:
    args = parse_args()
    require_file(args.transaction, "Transaction")
    require_file(args.identity, "Identity")
    if not args.model_b_metrics.is_file():
        raise FileNotFoundError(f"Model B validation metrics are required for comparison: {args.model_b_metrics}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_boundary, validation_boundary = selected_boundaries()
    validation_end = validation_boundary.end  # inclusive upper bound for validation
    # Scan and join
    transactions = scan_selected_columns(args.transaction, args.identity)
    # Keep only rows up to validation_end (train + validation)
    transactions = transactions.filter(pl.col("TransactionDT") <= validation_end)
    # Collect to DataFrame for feature computation
    df = transactions.collect(engine="streaming")
    # Sort by TransactionDT, TransactionID for strict-prior processing
    df = df.sort(["TransactionDT", "TransactionID"])
    print(f"Initial df shape: {df.shape}")
    print(f"Initial df columns: {df.columns}")
    # Compute addr1 features
    addr_df = af.build_addr_features(df.select(["TransactionID", "TransactionDT", "addr1", "TransactionAmt"]))
    print(f"Addr df shape: {addr_df.shape}")
    print(f"Addr df columns: {addr_df.columns}")
    if addr_df.width == 0:
        print("Addr df is empty!")
    # Join addr1 features back
    df = df.join(addr_df, on="TransactionID", how="left")
    print(f"After join df shape: {df.shape}")
    print(f"After join df columns: {df.columns}")
    # Now split into train and validation
    train = collect_partition(df.lazy(), train_boundary)
    validation = collect_partition(df.lazy(), validation_boundary)
    if train.height == 0 or validation.height == 0:
        raise ValueError("Train and validation partitions must both contain rows")

    # Categorical mappings from train only
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
    # Validation metrics
    from train_model_a import validation_metrics
    metrics = validation_metrics(validation_target, model.predict_proba(validation_matrix)[:, 1])
    model_b_metrics = json.loads(args.model_b_metrics.read_text())
    if model_b_metrics.get("evaluation_partition") != "validation":
        raise ValueError("Model B comparison artifact must contain validation metrics only")

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
        "added_addr_features": list(ADDR_FEATURES),
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
    write_json(args.output_dir / "model_b_vs_model_addr1_validation.json", comparison(model_b_metrics, metrics))
    write_json(args.output_dir / "reproducibility.json", {"random_seed": model_a.RANDOM_SEED, "package_versions": metadata["package_versions"], "python": metadata["execution"]["python"], "platform": metadata["execution"]["platform"], "fixed_training_iterations": model_a.TRAINING_ITERATIONS, "no_random_split_or_shuffle": True, "no_row_or_feature_subsampling": True})
    model.booster_.save_model(str(args.output_dir / "model.txt"))
    write_feature_importance(args.output_dir / "feature_importance.csv", model)
    write_report(args.report, metrics, model_b_metrics, params, memory)
    print(json.dumps({"model_addr1_validation_metrics": metrics, "model_b_vs_model_addr1": comparison(model_b_metrics, metrics)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"train_model_addr1.py: {error}", file=sys.stderr)
        raise
