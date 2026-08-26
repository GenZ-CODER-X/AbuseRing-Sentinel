#!/usr/bin/env python3
"""Deterministic, memory-conscious reconnaissance for IEEE train data only."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


ENTITY_COLUMNS = (
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
)
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/recon"))
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV does not exist or is not a file: {path}")


def derive_schema(path: Path) -> dict[str, pl.DataType]:
    """Infer once over the complete CSV, then reuse the exact schema everywhere."""
    return dict(
        pl.scan_csv(
            path,
            infer_schema=True,
            infer_schema_length=None,
            null_values=None,
            empty_string_is_null=True,
            ignore_errors=False,
            try_parse_dates=False,
        ).collect_schema()
    )


def scan(path: Path, schema: dict[str, pl.DataType]) -> pl.LazyFrame:
    """Return a scan with the pre-derived, deterministic parsing schema."""
    return pl.scan_csv(
        path,
        schema=schema,
        infer_schema=False,
        null_values=None,
        empty_string_is_null=True,
        ignore_errors=False,
        try_parse_dates=False,
    )


def scalar_row(frame: pl.LazyFrame) -> dict[str, Any]:
    return frame.collect(engine="streaming").row(0, named=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def missingness_rows(
    source: str, frame: pl.LazyFrame, columns: list[str]
) -> tuple[int, list[dict[str, Any]]]:
    result = scalar_row(
        frame.select(
            pl.len().alias("row_count"),
            *[pl.col(column).null_count().alias(column) for column in columns],
        )
    )
    row_count = int(result["row_count"])
    rows = []
    for column in columns:
        null_count = int(result[column])
        rows.append(
            {
                "source": source,
                "column": column,
                "null_count": null_count,
                "null_percentage": (100.0 * null_count / row_count) if row_count else 0.0,
            }
        )
    return row_count, rows


def entity_rows(transaction: pl.LazyFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in ENTITY_COLUMNS:
        scalar = scalar_row(
            transaction.select(
                pl.len().alias("row_count"),
                pl.col(column).null_count().alias("null_count"),
                pl.col(column)
                .filter(pl.col(column).is_not_null())
                .n_unique()
                .alias("unique_count"),
            )
        )
        row_count = int(scalar["row_count"])
        null_count = int(scalar["null_count"])
        non_null_count = row_count - null_count

        # This collects only one aggregated row per distinct non-null value, never
        # transaction rows. It is required for exact singleton and top-20 counts.
        frequencies = (
            transaction.select(column)
            .filter(pl.col(column).is_not_null())
            .group_by(column)
            .agg(pl.len().alias("frequency"))
            .collect(engine="streaming")
        )
        singleton_count = int(frequencies.filter(pl.col("frequency") == 1).height)
        top_values = frequencies.sort(
            by=["frequency", column], descending=[True, False]
        ).head(20)
        top_value_records = top_values.to_dicts()
        for record in top_value_records:
            record["value"] = record.pop(column)

        rows.append(
            {
                "column": column,
                "null_count": null_count,
                "null_percentage": (100.0 * null_count / row_count) if row_count else 0.0,
                "unique_count": int(scalar["unique_count"]),
                "uniqueness_ratio": (
                    float(scalar["unique_count"]) / non_null_count if non_null_count else 0.0
                ),
                "singleton_count": singleton_count,
                "singleton_percentage": (
                    100.0 * singleton_count / non_null_count if non_null_count else 0.0
                ),
                "top_20_values": top_value_records,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    require_file(args.transaction, "Transaction")
    require_file(args.identity, "Identity")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transaction_schema = derive_schema(args.transaction)
    identity_schema = derive_schema(args.identity)
    transaction = scan(args.transaction, transaction_schema)
    identity = scan(args.identity, identity_schema)
    transaction_columns = list(transaction_schema)
    identity_columns = list(identity_schema)

    required_transaction = {"TransactionID", "TransactionDT", "isFraud", *ENTITY_COLUMNS}
    missing = required_transaction.difference(transaction_columns)
    if missing:
        raise ValueError(f"Transaction CSV lacks required columns: {sorted(missing)}")
    if "TransactionID" not in identity_columns:
        raise ValueError("Identity CSV lacks required column: TransactionID")

    transaction_rows, transaction_missingness = missingness_rows(
        "train_transaction", transaction, transaction_columns
    )
    identity_rows, identity_missingness = missingness_rows(
        "train_identity", identity, identity_columns
    )

    transaction_id_stats = scalar_row(
        transaction.select(
            pl.col("TransactionID").null_count().alias("null_count"),
            pl.col("TransactionID").n_unique().alias("unique_count"),
        )
    )
    duplicate_transaction_ids = transaction_rows - int(transaction_id_stats["unique_count"])

    fraud = scalar_row(
        transaction.select(
            (pl.col("isFraud") == 0).sum().alias("isFraud_0_count"),
            (pl.col("isFraud") == 1).sum().alias("isFraud_1_count"),
        )
    )
    fraud_zero = int(fraud["isFraud_0_count"])
    fraud_one = int(fraud["isFraud_1_count"])

    temporal_exprs = [
        pl.col("TransactionDT").min().alias("min_TransactionDT"),
        pl.col("TransactionDT").max().alias("max_TransactionDT"),
        pl.col("TransactionDT")
        .filter(pl.col("TransactionDT").is_not_null())
        .n_unique()
        .alias("unique_TransactionDT_count"),
        pl.col("TransactionDT").null_count().alias("null_count"),
    ]
    temporal_exprs.extend(
        pl.col("TransactionDT")
        .quantile(quantile, interpolation="nearest")
        .alias(f"q{quantile:g}")
        for quantile in QUANTILES
    )
    temporal = scalar_row(transaction.select(*temporal_exprs))

    identity_id_stats = scalar_row(
        identity.select(
            pl.len().alias("row_count"),
            pl.col("TransactionID").n_unique().alias("unique_count"),
        )
    )
    identity_unique_ids = identity.select("TransactionID").unique()
    transaction_unique_ids = transaction.select("TransactionID").unique()
    matched_transaction_ids = scalar_row(
        transaction.select("TransactionID")
        .join(identity_unique_ids, on="TransactionID", how="semi")
        .select(pl.len().alias("count"))
    )["count"]
    identity_ids_not_in_transaction = scalar_row(
        identity_unique_ids.join(transaction_unique_ids, on="TransactionID", how="anti").select(
            pl.len().alias("count")
        )
    )["count"]

    missingness_path = args.output_dir / "missingness.csv"
    with missingness_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "column", "null_count", "null_percentage"],
        )
        writer.writeheader()
        writer.writerows(transaction_missingness + identity_missingness)

    cardinality_path = args.output_dir / "cardinality.csv"
    entities = entity_rows(transaction)
    with cardinality_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "column",
                "null_count",
                "null_percentage",
                "unique_count",
                "uniqueness_ratio",
                "singleton_count",
                "singleton_percentage",
            ],
        )
        writer.writeheader()
        for entity in entities:
            writer.writerow({key: value for key, value in entity.items() if key != "top_20_values"})
    write_json(args.output_dir / "entity_stats.json", {"entities": entities})

    write_json(
        args.output_dir / "schema.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "parsing_policy": {
                "schema_derivation": "Polars scans each complete CSV once with infer_schema_length=None; the resulting schema is reused by every later scan.",
                "empty_string_is_null": True,
                "null_values": None,
                "try_parse_dates": False,
                "ignore_errors": False,
            },
            "python_version": platform.python_version(),
            "polars_version": pl.__version__,
            "sources": {
                "train_transaction": {
                    "path": str(args.transaction.resolve()),
                    "row_count": transaction_rows,
                    "column_count": len(transaction_columns),
                    "columns": [
                        {"name": name, "physical_dtype": str(dtype)}
                        for name, dtype in transaction_schema.items()
                    ],
                    "TransactionID_null_count": int(transaction_id_stats["null_count"]),
                    "duplicate_TransactionID_rows_beyond_first": duplicate_transaction_ids,
                },
                "train_identity": {
                    "path": str(args.identity.resolve()),
                    "row_count": identity_rows,
                    "column_count": len(identity_columns),
                    "columns": [
                        {"name": name, "physical_dtype": str(dtype)}
                        for name, dtype in identity_schema.items()
                    ],
                },
            },
        },
    )
    write_json(
        args.output_dir / "fraud_stats.json",
        {
            "isFraud_0_count": fraud_zero,
            "isFraud_1_count": fraud_one,
            "fraud_rate": fraud_one / transaction_rows if transaction_rows else 0.0,
            "row_count": transaction_rows,
        },
    )
    write_json(args.output_dir / "temporal_stats.json", temporal)
    write_json(
        args.output_dir / "identity_coverage.json",
        {
            "train_identity_row_count": int(identity_id_stats["row_count"]),
            "train_identity_unique_TransactionID_count": int(identity_id_stats["unique_count"]),
            "duplicate_identity_TransactionID_rows_beyond_first": int(identity_id_stats["row_count"])
            - int(identity_id_stats["unique_count"]),
            "train_transaction_TransactionIDs_matching_identity_count": int(matched_transaction_ids),
            "train_transaction_identity_coverage_percentage": (
                100.0 * int(matched_transaction_ids) / transaction_rows if transaction_rows else 0.0
            ),
            "identity_TransactionIDs_not_present_in_train_transaction_count": int(
                identity_ids_not_in_transaction
            ),
        },
    )
    update_documentation(
        transaction_rows=transaction_rows,
        transaction_columns=len(transaction_columns),
        identity_rows=identity_rows,
        identity_columns=len(identity_columns),
        duplicate_transaction_ids=duplicate_transaction_ids,
        fraud_zero=fraud_zero,
        fraud_one=fraud_one,
        fraud_rate=fraud_one / transaction_rows if transaction_rows else 0.0,
        temporal=temporal,
        identity_unique_ids=int(identity_id_stats["unique_count"]),
        identity_duplicate_ids=int(identity_id_stats["row_count"])
        - int(identity_id_stats["unique_count"]),
        matched_transaction_ids=int(matched_transaction_ids),
        identity_ids_not_in_transaction=int(identity_ids_not_in_transaction),
    )


def update_documentation(
    *,
    transaction_rows: int,
    transaction_columns: int,
    identity_rows: int,
    identity_columns: int,
    duplicate_transaction_ids: int,
    fraud_zero: int,
    fraud_one: int,
    fraud_rate: float,
    temporal: dict[str, Any],
    identity_unique_ids: int,
    identity_duplicate_ids: int,
    matched_transaction_ids: int,
    identity_ids_not_in_transaction: int,
) -> None:
    docs_path = Path(__file__).resolve().parents[1] / "docs" / "DATASET_RECON.md"
    start = "<!-- GENERATED SUMMARY START -->"
    end = "<!-- GENERATED SUMMARY END -->"
    text = docs_path.read_text()
    summary = f"""{start}
## Generated results

| Measure | Result |
| --- | ---: |
| Train transaction rows / columns | {transaction_rows:,} / {transaction_columns:,} |
| Duplicate transaction-ID rows beyond first | {duplicate_transaction_ids:,} |
| Train identity rows / columns | {identity_rows:,} / {identity_columns:,} |
| `isFraud=0` / `isFraud=1` | {fraud_zero:,} / {fraud_one:,} |
| Fraud rate | {fraud_rate:.8%} |
| `TransactionDT` min / max | {temporal['min_TransactionDT']:,} / {temporal['max_TransactionDT']:,} |
| Unique non-null `TransactionDT` | {temporal['unique_TransactionDT_count']:,} |
| Unique identity `TransactionID` | {identity_unique_ids:,} |
| Duplicate identity-ID rows beyond first | {identity_duplicate_ids:,} |
| Transaction IDs matching identity | {matched_transaction_ids:,} ({(100.0 * matched_transaction_ids / transaction_rows) if transaction_rows else 0.0:.6f}%) |
| Identity IDs absent from transaction | {identity_ids_not_in_transaction:,} |
{end}"""
    if start not in text or end not in text:
        raise RuntimeError(f"Documentation markers are missing from {docs_path}")
    before, remainder = text.split(start, maxsplit=1)
    _, after = remainder.split(end, maxsplit=1)
    docs_path.write_text(before + summary + after)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"recon_dataset.py: {error}", file=sys.stderr)
        raise
