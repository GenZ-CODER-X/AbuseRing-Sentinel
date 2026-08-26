#!/usr/bin/env python3
"""Train Model C: Model B plus eight label-free causal graph features."""

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
import train_model_b as model_b
from causal_graph_features import (
    GRAPH_FEATURES,
    GRAPH_INPUT_COLUMNS,
    build_causal_graph_features,
    feature_specification,
    prepare_graph_events,
)
from validation_boundaries import LABEL_FIELD, PARTITION_BOUNDARIES, partition_filter, validate_boundaries


EXPERIMENT = "model_c"
FEATURES = model_b.FEATURES + GRAPH_FEATURES
CATEGORICAL_FEATURES = model_b.CATEGORICAL_FEATURES
NUMERIC_FEATURES = tuple(feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models/model_c"))
    parser.add_argument("--report", type=Path, default=Path("docs/MODEL_C_BASELINE.md"))
    parser.add_argument("--model-a-metrics", type=Path, default=Path("artifacts/models/model_a/validation_metrics.json"))
    parser.add_argument("--model-b-metrics", type=Path, default=Path("artifacts/models/model_b/validation_metrics.json"))
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV does not exist or is not a file: {path}")


def selected_boundaries() -> tuple[Any, Any]:
    validate_boundaries(PARTITION_BOUNDARIES)
    by_name = {boundary.name: boundary for boundary in PARTITION_BOUNDARIES}
    return by_name["train"], by_name["validation"]


def scan_selected_columns(transaction_path: Path, identity_path: Path) -> pl.LazyFrame:
    """Lazily scan Model B fields, event key, and raw device fields only."""
    transaction_columns = [feature for feature in model_b.FEATURES if feature not in {"DeviceInfo", "DeviceType"}]
    transactions = pl.scan_csv(transaction_path).select("TransactionID", *transaction_columns, LABEL_FIELD)
    identities = pl.scan_csv(identity_path).select("TransactionID", "DeviceInfo", "DeviceType")
    frame = transactions.join(identities, on="TransactionID", how="left").select("TransactionID", *model_b.FEATURES, LABEL_FIELD)
    missing = (set(model_b.FEATURES) | {"TransactionID", LABEL_FIELD}).difference(frame.collect_schema().names())
    if missing:
        raise ValueError(f"Input CSVs lack required columns: {sorted(missing)}")
    return frame


def collect_partition(frame: pl.LazyFrame, boundary: Any) -> pl.DataFrame:
    return frame.filter(partition_filter(boundary)).collect(engine="streaming")


def attach_causal_graph_features(train: pl.DataFrame, validation: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Build state over train then validation events; labels never enter this function."""
    graph_events = pl.concat(
        [train.select(*GRAPH_INPUT_COLUMNS), validation.select(*GRAPH_INPUT_COLUMNS)],
        how="vertical",
    )
    graph_features = build_causal_graph_features(prepare_graph_events(graph_events))
    evidence = feature_specification() | {
        "graph_event_row_count": graph_events.height,
        "train_graph_event_row_count": train.height,
        "validation_graph_event_row_count": validation.height,
        "label_column_passed_to_graph_builder": False,
        "same_timestamp_batching": True,
    }
    return (
        train.join(graph_features, on="TransactionID", how="left"),
        validation.join(graph_features, on="TransactionID", how="left"),
        evidence,
    )


def matrix_and_target(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    matrix = frame.select(*[pl.col(feature).cast(pl.Float32) for feature in FEATURES]).to_numpy()
    target = frame.get_column(LABEL_FIELD).cast(pl.Int8).to_numpy()
    return matrix, target


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def feature_importance_rows(model: lgb.LGBMClassifier) -> list[dict[str, Any]]:
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    return sorted(
        [{"feature": feature, "importance_gain": float(gain_value), "importance_split": int(split_value)} for feature, gain_value, split_value in zip(FEATURES, gain, split)],
        key=lambda row: (-row["importance_gain"], row["feature"]),
    )


def write_feature_importance(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "importance_gain", "importance_split"])
        writer.writeheader()
        writer.writerows(rows)


def metric_value(metrics: dict[str, Any], name: str) -> float:
    if name == "recall_at_fpr_le_1_percent":
        return float(metrics[name]["recall"])
    if name == "recall_at_fpr_le_point_1_percent":
        return float(metrics[name]["recall"])
    return float(metrics[name])


COMPARISON_METRICS = (
    "pr_auc_average_precision",
    "roc_auc",
    "precision_at_selected_threshold",
    "recall_at_selected_threshold",
    "f1_at_selected_threshold",
    "recall_at_fpr_le_1_percent",
    "recall_at_fpr_le_point_1_percent",
)


def comparison(model_a_metrics: dict[str, Any], model_b_metrics: dict[str, Any], model_c_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "model_a": metric_value(model_a_metrics, name),
            "model_b": metric_value(model_b_metrics, name),
            "model_c": metric_value(model_c_metrics, name),
            "incremental_c_minus_b": metric_value(model_c_metrics, name) - metric_value(model_b_metrics, name),
        }
        for name in COMPARISON_METRICS
    }


def write_report(path: Path, metrics_a: dict[str, Any], metrics_b: dict[str, Any], metrics_c: dict[str, Any], comparison_rows: dict[str, Any], graph_importance: list[dict[str, Any]], params: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table_names = {
        "pr_auc_average_precision": "PR-AUC / average precision",
        "roc_auc": "ROC-AUC",
        "precision_at_selected_threshold": "Precision at selected threshold",
        "recall_at_selected_threshold": "Recall at selected threshold",
        "f1_at_selected_threshold": "F1 at selected threshold",
        "recall_at_fpr_le_1_percent": "Recall at FPR ≤1%",
        "recall_at_fpr_le_point_1_percent": "Recall at FPR ≤0.1%",
    }
    comparison_table = "".join(
        f"| {table_names[name]} | {row['model_a']:.12f} | {row['model_b']:.12f} | {row['model_c']:.12f} | {row['incremental_c_minus_b']:+.12f} |\n"
        for name, row in comparison_rows.items()
    )
    importance_table = "".join(
        f"| `{row['feature']}` | {row['importance_gain']:.6f} | {row['importance_split']:,} |\n"
        for row in graph_importance
    )
    matrix = metrics_c["confusion_matrix"]["rows"]
    path.write_text(
        f"# Model C causal graph baseline\n\n"
        f"Model C is Model B plus exactly eight non-target, event-time graph/topology features. It uses a full-card-signature node (only when `card1`–`card6` are all present), `DeviceInfo`, and `addr1`; no other reusable graph nodes are created.\n\n"
        f"## Causality and leakage controls\n\n"
        f"The graph builder receives only `TransactionID`, `TransactionDT`, card fields, `DeviceInfo`, and `addr1`; it never receives `isFraud`. Events are sorted by time and processed in complete timestamp batches: every feature reads state from strictly earlier timestamps, then the entire batch updates state. Therefore an event cannot see itself or another event with the same `TransactionDT`. The train and validation intervals come only from `validation_boundaries.py`; no other partition is materialized or evaluated.\n\n"
        f"No target encoding, fraud statistics, V-family, historical labels, or future events are used. Model B categorical mappings and missing-value handling remain unchanged.\n\n"
        f"## Validation comparison\n\n"
        f"| Metric | Model A | Model B | Model C | C − B |\n| --- | ---: | ---: | ---: | ---: |\n{comparison_table}\n"
        f"Model C selects threshold `{metrics_c['threshold_selection']['selected_threshold']:.12f}` exactly once by maximizing validation recall under FPR ≤1%. Its confusion matrix `[[TN, FP], [FN, TP]]` is `[[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]]`.\n\n"
        f"## Graph feature importance\n\n| Feature | Gain | Splits |\n| --- | ---: | ---: |\n{importance_table}\n"
        f"The inherited fixed LightGBM configuration uses 500 iterations, learning rate 0.05, 31 leaves, no row/feature subsampling, deterministic execution, seed {model_a.RANDOM_SEED}, and train-derived `scale_pos_weight={params['scale_pos_weight']:.12f}`.\n"
    )


def main() -> None:
    args = parse_args()
    require_file(args.transaction, "Transaction")
    require_file(args.identity, "Identity")
    for path in (args.model_a_metrics, args.model_b_metrics):
        if not path.is_file():
            raise FileNotFoundError(f"Required baseline validation artifact is missing: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_boundary, validation_boundary = selected_boundaries()
    source = scan_selected_columns(args.transaction, args.identity)
    train_raw = collect_partition(source, train_boundary)
    validation_raw = collect_partition(source, validation_boundary)
    train_with_graph, validation_with_graph, causality_evidence = attach_causal_graph_features(train_raw, validation_raw)
    mappings = model_b.training_category_mappings(train_with_graph)
    category_cardinalities = {feature: len(mapping) for feature, mapping in mappings.items()}
    train_frame_bytes = train_with_graph.estimated_size()
    validation_frame_bytes = validation_with_graph.estimated_size()
    train_matrix, train_target = matrix_and_target(model_b.encode_categoricals(train_with_graph, mappings))
    validation_matrix, validation_target = matrix_and_target(model_b.encode_categoricals(validation_with_graph, mappings))
    del train_raw, validation_raw, train_with_graph, validation_with_graph
    train_positive = int(train_target.sum())
    train_negative = int(train_target.size - train_positive)
    params = model_a.predeclared_model_parameters(train_negative / train_positive)
    model = lgb.LGBMClassifier(**params)
    model.fit(train_matrix, train_target, feature_name=list(FEATURES), categorical_feature=list(CATEGORICAL_FEATURES))
    metrics_c = model_a.validation_metrics(validation_target, model.predict_proba(validation_matrix)[:, 1])
    metrics_a = json.loads(args.model_a_metrics.read_text())
    metrics_b = json.loads(args.model_b_metrics.read_text())
    if metrics_a.get("evaluation_partition") != "validation" or metrics_b.get("evaluation_partition") != "validation":
        raise ValueError("Baseline comparison artifacts must contain validation metrics only")
    comparison_rows = comparison(metrics_a, metrics_b, metrics_c)
    importance = feature_importance_rows(model)
    graph_importance = [row for row in importance if row["feature"] in GRAPH_FEATURES]
    memory = {"train_frame_bytes": train_frame_bytes, "train_matrix_bytes": train_matrix.nbytes, "validation_frame_bytes": validation_frame_bytes, "validation_matrix_bytes": validation_matrix.nbytes}
    metadata = {
        "experiment": EXPERIMENT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "transaction_source": str(args.transaction.resolve()),
        "identity_source": str(args.identity.resolve()),
        "base_experiment": "model_b",
        "added_graph_features": list(GRAPH_FEATURES),
        "train_boundary": {"min": train_boundary.start, "max": train_boundary.end},
        "validation_boundary": {"min": validation_boundary.start, "max": validation_boundary.end},
        "train_row_count": int(train_target.size),
        "validation_row_count": int(validation_target.size),
        "train_fraud_count": train_positive,
        "validation_fraud_count": int(validation_target.sum()),
        "random_seed": model_a.RANDOM_SEED,
        "model_parameters": params,
        "categorical_handling": {"features": list(CATEGORICAL_FEATURES), "missing_code": -1, "unknown_validation_code": 0, "category_cardinalities_from_train_only": category_cardinalities},
        "numeric_missing_values": "Preserved as NaN for LightGBM native missing-value handling.",
        "memory": memory,
        "package_versions": {"lightgbm": lgb.__version__, "numpy": np.__version__, "polars": pl.__version__, "scikit_learn": sklearn.__version__},
        "execution": {"python": platform.python_version(), "platform": platform.platform(), "num_threads": model_a.NUM_THREADS, "training_iterations_fixed_before_validation": model_a.TRAINING_ITERATIONS},
    }
    write_json(args.output_dir / "config_metadata.json", metadata)
    write_json(args.output_dir / "feature_list.json", {"features": list(FEATURES), "categorical_features": list(CATEGORICAL_FEATURES), "numeric_features": list(NUMERIC_FEATURES)})
    write_json(args.output_dir / "validation_metrics.json", metrics_c)
    write_json(args.output_dir / "model_a_vs_b_vs_c_validation.json", comparison_rows)
    write_json(args.output_dir / "causality_evidence.json", causality_evidence)
    write_json(args.output_dir / "graph_feature_importance.json", {"features": graph_importance})
    write_json(args.output_dir / "reproducibility.json", {"random_seed": model_a.RANDOM_SEED, "package_versions": metadata["package_versions"], "python": metadata["execution"]["python"], "platform": metadata["execution"]["platform"], "fixed_training_iterations": model_a.TRAINING_ITERATIONS, "same_timestamp_batching": True, "no_random_split_or_shuffle": True, "no_row_or_feature_subsampling": True})
    model.booster_.save_model(str(args.output_dir / "model.txt"))
    write_feature_importance(args.output_dir / "feature_importance.csv", importance)
    write_report(args.report, metrics_a, metrics_b, metrics_c, comparison_rows, graph_importance, params)
    print(json.dumps({"model_c_validation_metrics": metrics_c, "comparison": comparison_rows, "graph_feature_importance": graph_importance}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"train_model_c.py: {error}", file=sys.stderr)
        raise
