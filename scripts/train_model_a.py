#!/usr/bin/env python3
"""Train and evaluate the leakage-safe Model A LightGBM baseline.

Only the train and validation boundaries imported from validation_boundaries are
used here. No other partition is materialized, scored, or reported.
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from validation_boundaries import (
    CHRONOLOGICAL_FIELD,
    LABEL_FIELD,
    PARTITION_BOUNDARIES,
    partition_filter,
    validate_boundaries,
)


RANDOM_SEED = 20_260_826
TRAINING_ITERATIONS = 500
NUM_THREADS = 4

CORE_FEATURES = (
    "TransactionAmt",
    "ProductCD",
    CHRONOLOGICAL_FIELD,
    "dist1",
    "dist2",
)
ENTITY_FEATURES = (
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
)
C_FEATURES = tuple(f"C{number}" for number in range(1, 15) if number != 3)
D_FEATURES = tuple(f"D{number}" for number in range(1, 16))
M_FEATURES = tuple(f"M{number}" for number in range(1, 10))
FEATURES = CORE_FEATURES + ENTITY_FEATURES + C_FEATURES + D_FEATURES + M_FEATURES
CATEGORICAL_FEATURES = (
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    *M_FEATURES,
)
NUMERIC_FEATURES = tuple(feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models/model_a"))
    parser.add_argument("--report", type=Path, default=Path("docs/MODEL_A_BASELINE.md"))
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Transaction CSV does not exist or is not a file: {path}")


def selected_boundaries() -> tuple[Any, Any]:
    """Obtain the only two partitions used by this experiment from the source of truth."""
    validate_boundaries(PARTITION_BOUNDARIES)
    by_name = {boundary.name: boundary for boundary in PARTITION_BOUNDARIES}
    return by_name["train"], by_name["validation"]


def scan_selected_columns(path: Path) -> pl.LazyFrame:
    """Lazily scan only approved Model A fields and the training target."""
    frame = pl.scan_csv(path).select(*FEATURES, LABEL_FIELD)
    available = set(frame.collect_schema().names())
    required = set(FEATURES) | {LABEL_FIELD}
    missing = required.difference(available)
    if missing:
        raise ValueError(f"Transaction CSV lacks required columns: {sorted(missing)}")
    return frame


def collect_partition(frame: pl.LazyFrame, boundary: Any) -> pl.DataFrame:
    """Materialize one necessary selected-column partition using streaming execution."""
    return frame.filter(partition_filter(boundary)).collect(engine="streaming")


def training_category_mappings(train: pl.DataFrame) -> dict[str, dict[Any, int]]:
    """Fit deterministic categorical mappings from training values only."""
    mappings: dict[str, dict[Any, int]] = {}
    for feature in CATEGORICAL_FEATURES:
        values = train.get_column(feature).drop_nulls().unique().sort().to_list()
        mappings[feature] = {value: index for index, value in enumerate(values, start=1)}
    return mappings


def encode_categoricals(
    frame: pl.DataFrame, mappings: dict[str, dict[Any, int]]
) -> pl.DataFrame:
    """Encode categories using train-only mappings.

    Missing values receive -1, which LightGBM treats as missing for categorical
    features. Validation-only values receive the reserved valid category 0.
    """
    expressions = []
    for feature in CATEGORICAL_FEATURES:
        expressions.append(
            pl.when(pl.col(feature).is_null())
            .then(pl.lit(-1))
            .otherwise(pl.col(feature).replace_strict(mappings[feature], default=0))
            .cast(pl.Int32)
            .alias(feature)
        )
    return frame.with_columns(*expressions)


def matrix_and_target(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Create a float32 model matrix; numeric nulls are preserved as NaN."""
    matrix = frame.select(
        *[pl.col(feature).cast(pl.Float32).alias(feature) for feature in FEATURES]
    ).to_numpy()
    target = frame.get_column(LABEL_FIELD).cast(pl.Int8).to_numpy()
    return matrix, target


def predeclared_model_parameters(scale_pos_weight: float) -> dict[str, Any]:
    """Return the fixed Model A configuration before validation prediction."""
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": TRAINING_ITERATIONS,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "min_split_gain": 0.0,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "max_bin": 255,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_SEED,
        "n_jobs": NUM_THREADS,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def best_recall_at_fpr(
    target: np.ndarray, probability: np.ndarray, fpr_cap: float
) -> dict[str, float]:
    """Find max recall under an FPR cap, breaking ties with higher threshold."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(target, probability)
    eligible = np.flatnonzero(false_positive_rate <= fpr_cap)
    if eligible.size == 0:
        raise RuntimeError(f"No ROC threshold satisfies FPR <= {fpr_cap}")
    best_recall = float(true_positive_rate[eligible].max())
    best = eligible[np.isclose(true_positive_rate[eligible], best_recall)]
    selected_index = best[np.argmax(thresholds[best])]
    return {
        "fpr_cap": fpr_cap,
        "recall": float(true_positive_rate[selected_index]),
        "fpr": float(false_positive_rate[selected_index]),
        "threshold": float(thresholds[selected_index]),
    }


def validation_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    """Calculate every reported metric using validation labels and predictions only."""
    at_one_percent = best_recall_at_fpr(target, probability, 0.01)
    at_point_one_percent = best_recall_at_fpr(target, probability, 0.001)
    threshold = at_one_percent["threshold"]
    predicted = (probability >= threshold).astype(np.int8)
    matrix = confusion_matrix(target, predicted, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, predicted, average="binary", zero_division=0
    )
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    return {
        "evaluation_partition": "validation",
        "row_count": int(target.size),
        "fraud_count": int(target.sum()),
        "fraud_prevalence": float(target.mean()),
        "pr_auc_average_precision": float(average_precision_score(target, probability)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "threshold_selection": {
            "method": "Maximize validation recall subject to FPR <= 1%; ties use the higher threshold to reduce flags.",
            "selected_threshold": threshold,
            "selected_threshold_fpr": at_one_percent["fpr"],
            "selected_threshold_recall": at_one_percent["recall"],
        },
        "precision_at_selected_threshold": float(precision),
        "recall_at_selected_threshold": float(recall),
        "f1_at_selected_threshold": float(f1),
        "recall_at_fpr_le_1_percent": at_one_percent,
        "recall_at_fpr_le_point_1_percent": at_point_one_percent,
        "confusion_matrix": {
            "labels": [0, 1],
            "rows": [[int(value) for value in row] for row in matrix.tolist()],
            "format": "[[true_negative, false_positive], [false_negative, true_positive]]",
        },
        "validation_transactions_flagged": int(predicted.sum()),
        "false_positive_count": int(false_positive),
        "false_negative_count": int(false_negative),
        "true_positive_count": int(true_positive),
        "true_negative_count": int(true_negative),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_feature_importance(path: Path, model: lgb.LGBMClassifier) -> None:
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    rows = sorted(
        (
            {
                "feature": feature,
                "importance_gain": float(feature_gain),
                "importance_split": int(feature_split),
            }
            for feature, feature_gain, feature_split in zip(FEATURES, gain, split)
        ),
        key=lambda row: (-row["importance_gain"], row["feature"]),
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "importance_gain", "importance_split"])
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    metrics: dict[str, Any],
    parameters: dict[str, Any],
    train_row_count: int,
    validation_row_count: int,
    category_cardinalities: dict[str, int],
    memory: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = metrics["threshold_selection"]
    matrix = metrics["confusion_matrix"]["rows"]
    feature_list = ", ".join(f"`{feature}`" for feature in FEATURES)
    categorical_list = ", ".join(f"`{feature}`" for feature in CATEGORICAL_FEATURES)
    numeric_list = ", ".join(f"`{feature}`" for feature in NUMERIC_FEATURES)
    path.write_text(
        f"# Model A baseline\n\n"
        f"## Objective\n\n"
        f"Model A is the first reproducible, leakage-safe LightGBM fraud-risk benchmark for AbuseRing Sentinel. It deliberately uses event-time transaction and entity fields only, so later experiments can measure the incremental value of richer feature sets.\n\n"
        f"## Temporal split and leakage controls\n\n"
        f"The script imports the locked boundaries from `scripts/validation_boundaries.py`; it does not recreate them. Training uses `TransactionDT` 86,400–11,059,199 ({train_row_count:,} rows) and validation uses 11,059,200–13,391,999 ({validation_row_count:,} rows). There is no random split or shuffle. The experiment materializes and evaluates only those two partitions; no other partition is scored, counted, or used for metrics or tuning.\n\n"
        f"All category mappings are fitted from training values only. No target encoding, fraud/entity rate, historical/velocity/frequency aggregate, graph, identity, device, V-family, or other derived feature is used. `isFraud` is used only as the training target and validation label.\n\n"
        f"## Features\n\n"
        f"The 50 approved features are: {feature_list}.\n\n"
        f"Categorical features are: {categorical_list}. Numeric features are: {numeric_list}.\n\n"
        f"## Preprocessing and model\n\n"
        f"Numeric nulls remain `NaN` for LightGBM's native missing-value handling. Training categories are assigned deterministic positive integer codes; missing categorical values receive `-1` (LightGBM categorical missing), and validation-only categories receive the fixed reserved code `0`. This means numeric-coded card and address fields are handled categorically, not as ordered measurements. Training category cardinalities are recorded in the experiment metadata.\n\n"
        f"The predeclared model is LightGBM GBDT with 500 fixed boosting iterations, learning rate 0.05, 31 leaves, `min_child_samples=100`, no row/feature subsampling, deterministic column-wise execution, seed {RANDOM_SEED:,}, and `scale_pos_weight={parameters['scale_pos_weight']:.12f}` derived only from the train target prevalence. There is no early stopping or iterative validation tuning.\n\n"
        f"## Validation results\n\n"
        f"| Metric | Value |\n| --- | ---: |\n"
        f"| PR-AUC / average precision | {metrics['pr_auc_average_precision']:.12f} |\n"
        f"| ROC-AUC | {metrics['roc_auc']:.12f} |\n"
        f"| Validation fraud prevalence | {metrics['fraud_prevalence']:.12%} |\n"
        f"| Selected threshold | {selected['selected_threshold']:.12f} |\n"
        f"| Precision at threshold | {metrics['precision_at_selected_threshold']:.12f} |\n"
        f"| Recall at threshold | {metrics['recall_at_selected_threshold']:.12f} |\n"
        f"| F1 at threshold | {metrics['f1_at_selected_threshold']:.12f} |\n"
        f"| Recall at FPR ≤1% | {metrics['recall_at_fpr_le_1_percent']['recall']:.12f} |\n"
        f"| Recall at FPR ≤0.1% | {metrics['recall_at_fpr_le_point_1_percent']['recall']:.12f} |\n"
        f"| Validation transactions flagged | {metrics['validation_transactions_flagged']:,} |\n"
        f"| False positives | {metrics['false_positive_count']:,} |\n"
        f"| False negatives | {metrics['false_negative_count']:,} |\n\n"
        f"Confusion matrix (`[[TN, FP], [FN, TP]]`): `[[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]]`.\n\n"
        f"## Threshold selection\n\n"
        f"The operating threshold was selected exactly once from validation predictions: maximize recall under validation FPR ≤1%; ties use the higher threshold to reduce unnecessary flags. The chosen threshold has FPR {selected['selected_threshold_fpr']:.12f} and recall {selected['selected_threshold_recall']:.12f}. The 0.1% FPR result is reported as a separate constrained-recall diagnostic, not as a tuning loop.\n\n"
        f"## Memory behavior and limitations\n\n"
        f"Polars lazily scans the CSV and projects only approved fields plus the target. It materializes the necessary train and validation selected-column frames separately. Observed selected-frame/matrix sizes were {memory['train_frame_bytes']:,}/{memory['train_matrix_bytes']:,} bytes for train and {memory['validation_frame_bytes']:,}/{memory['validation_matrix_bytes']:,} bytes for validation; the full 590,540-row transaction table is never collected.\n\n"
        f"Model A contains no V-family, identity/device, behavioral temporal, graph, target-encoding, historical aggregate, or production-serving components. It is a deliberately bounded benchmark, not a final fraud model.\n"
    )


def main() -> None:
    args = parse_args()
    require_file(args.transaction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_boundary, validation_boundary = selected_boundaries()
    transactions = scan_selected_columns(args.transaction)

    train = collect_partition(transactions, train_boundary)
    validation = collect_partition(transactions, validation_boundary)
    if train.height == 0 or validation.height == 0:
        raise ValueError("Train and validation partitions must both contain rows")

    mappings = training_category_mappings(train)
    category_cardinalities = {feature: len(mapping) for feature, mapping in mappings.items()}
    train_frame_bytes = train.estimated_size()
    validation_frame_bytes = validation.estimated_size()
    train_encoded = encode_categoricals(train, mappings)
    validation_encoded = encode_categoricals(validation, mappings)
    train_matrix, train_target = matrix_and_target(train_encoded)
    validation_matrix, validation_target = matrix_and_target(validation_encoded)
    del train_encoded, validation_encoded, train, validation

    train_positive = int(train_target.sum())
    train_negative = int(train_target.size - train_positive)
    if train_positive == 0 or train_negative == 0:
        raise ValueError("Training target must contain both fraud classes")
    parameters = predeclared_model_parameters(train_negative / train_positive)
    model = lgb.LGBMClassifier(**parameters)
    model.fit(
        train_matrix,
        train_target,
        feature_name=list(FEATURES),
        categorical_feature=list(CATEGORICAL_FEATURES),
    )
    validation_probability = model.predict_proba(validation_matrix)[:, 1]
    metrics = validation_metrics(validation_target, validation_probability)

    memory = {
        "train_frame_bytes": train_frame_bytes,
        "train_matrix_bytes": train_matrix.nbytes,
        "validation_frame_bytes": validation_frame_bytes,
        "validation_matrix_bytes": validation_matrix.nbytes,
    }
    metadata = {
        "experiment": "model_a",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "transaction_source": str(args.transaction.resolve()),
        "chronological_field": CHRONOLOGICAL_FIELD,
        "train_boundary": {"min": train_boundary.start, "max": train_boundary.end},
        "validation_boundary": {"min": validation_boundary.start, "max": validation_boundary.end},
        "train_row_count": int(train_target.size),
        "validation_row_count": int(validation_target.size),
        "train_fraud_count": train_positive,
        "validation_fraud_count": int(validation_target.sum()),
        "random_seed": RANDOM_SEED,
        "model_parameters": parameters,
        "categorical_handling": {
            "features": list(CATEGORICAL_FEATURES),
            "training_code_range": "positive integers assigned from sorted non-null training categories",
            "missing_code": -1,
            "unknown_validation_code": 0,
            "category_cardinalities_from_train_only": category_cardinalities,
        },
        "numeric_missing_values": "Preserved as NaN for LightGBM native missing-value handling.",
        "memory": memory,
        "package_versions": {
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "execution": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "num_threads": NUM_THREADS,
            "training_iterations_fixed_before_validation": TRAINING_ITERATIONS,
        },
    }
    write_json(args.output_dir / "config_metadata.json", metadata)
    write_json(
        args.output_dir / "feature_list.json",
        {
            "features": list(FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "numeric_features": list(NUMERIC_FEATURES),
        },
    )
    write_json(args.output_dir / "validation_metrics.json", metrics)
    write_json(
        args.output_dir / "reproducibility.json",
        {
            "random_seed": RANDOM_SEED,
            "package_versions": metadata["package_versions"],
            "python": metadata["execution"]["python"],
            "platform": metadata["execution"]["platform"],
            "fixed_training_iterations": TRAINING_ITERATIONS,
            "no_random_split_or_shuffle": True,
            "no_row_or_feature_subsampling": True,
        },
    )
    model.booster_.save_model(str(args.output_dir / "model.txt"))
    write_feature_importance(args.output_dir / "feature_importance.csv", model)
    write_report(
        args.report,
        metrics=metrics,
        parameters=parameters,
        train_row_count=int(train_target.size),
        validation_row_count=int(validation_target.size),
        category_cardinalities=category_cardinalities,
        memory=memory,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"train_model_a.py: {error}", file=sys.stderr)
        raise
