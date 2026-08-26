#!/usr/bin/env python3
"""Define and audit the locked chronological validation partitions.

The boundaries below are the sole source of truth for chronological splits in
future training code.  They are based only on ``TransactionDT``; neither
``TransactionID`` nor labels participate in assigning a partition.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


CHRONOLOGICAL_FIELD = "TransactionDT"
LABEL_FIELD = "isFraud"


@dataclass(frozen=True)
class PartitionBoundary:
    """An inclusive chronological interval used for one data partition."""

    name: str
    start: int
    end: int


# Locked, inclusive boundaries. Future training scripts should import these
# definitions and use ``partition_filter`` rather than recreate their own split.
PARTITION_BOUNDARIES = (
    PartitionBoundary("train", 86_400, 11_059_199),
    PartitionBoundary("validation", 11_059_200, 13_391_999),
    PartitionBoundary("final_test", 13_392_000, 15_811_131),
)


def partition_filter(boundary: PartitionBoundary) -> pl.Expr:
    """Return the inclusive TransactionDT predicate for ``boundary``."""
    return pl.col(CHRONOLOGICAL_FIELD).is_between(boundary.start, boundary.end)


def validate_boundaries(
    boundaries: tuple[PartitionBoundary, ...] = PARTITION_BOUNDARIES,
) -> None:
    """Fail when locked intervals overlap, are invalid, or leave a gap."""
    if not boundaries:
        raise ValueError("At least one partition boundary is required")

    for boundary in boundaries:
        if boundary.start > boundary.end:
            raise ValueError(f"{boundary.name} starts after it ends")

    for previous, current in zip(boundaries, boundaries[1:]):
        if current.start <= previous.end:
            raise ValueError(
                f"Overlapping partition boundaries: {previous.name} and {current.name}"
            )
        if current.start != previous.end + 1:
            raise ValueError(
                f"Non-contiguous partition boundaries: {previous.name} and {current.name}"
            )


def scan_transactions(path: Path) -> pl.LazyFrame:
    """Lazily scan only the fields needed for the boundary audit."""
    if not path.is_file():
        raise FileNotFoundError(f"Transaction CSV does not exist or is not a file: {path}")

    frame = pl.scan_csv(path).select(CHRONOLOGICAL_FIELD, LABEL_FIELD)
    columns = set(frame.collect_schema().names())
    missing = {CHRONOLOGICAL_FIELD, LABEL_FIELD}.difference(columns)
    if missing:
        raise ValueError(f"Transaction CSV lacks required columns: {sorted(missing)}")
    return frame


def scalar_row(frame: pl.LazyFrame) -> dict[str, Any]:
    """Collect only a single aggregate row via Polars' streaming engine."""
    return frame.collect(engine="streaming").row(0, named=True)


def audit_partitions(
    transactions: pl.LazyFrame,
    boundaries: tuple[PartitionBoundary, ...] = PARTITION_BOUNDARIES,
) -> dict[str, Any]:
    """Return partition counts without materializing transaction rows."""
    validate_boundaries(boundaries)
    predicates = {boundary.name: partition_filter(boundary) for boundary in boundaries}
    in_any_partition = pl.any_horizontal(*predicates.values())

    aggregates: list[pl.Expr] = [
        pl.len().alias("total_row_count"),
        pl.col(CHRONOLOGICAL_FIELD).min().alias("observed_min_TransactionDT"),
        pl.col(CHRONOLOGICAL_FIELD).max().alias("observed_max_TransactionDT"),
        pl.col(CHRONOLOGICAL_FIELD).null_count().alias("TransactionDT_null_count"),
        (~in_any_partition).sum().alias("outside_locked_boundaries_count"),
    ]
    for boundary in boundaries:
        predicate = predicates[boundary.name]
        aggregates.extend(
            (
                predicate.sum().alias(f"{boundary.name}_row_count"),
                ((pl.col(LABEL_FIELD) == 1) & predicate)
                .sum()
                .alias(f"{boundary.name}_fraud_count"),
            )
        )

    result = scalar_row(transactions.select(*aggregates))
    expected_min = boundaries[0].start
    expected_max = boundaries[-1].end
    observed_min = result["observed_min_TransactionDT"]
    observed_max = result["observed_max_TransactionDT"]
    if observed_min != expected_min or observed_max != expected_max:
        raise ValueError(
            "Expected TransactionDT range is not present: "
            f"expected {expected_min}..{expected_max}, observed {observed_min}..{observed_max}"
        )
    if result["TransactionDT_null_count"]:
        raise ValueError("TransactionDT contains null values")
    if result["outside_locked_boundaries_count"]:
        raise ValueError("Rows exist outside the locked TransactionDT boundaries")

    partitions: dict[str, dict[str, int | float]] = {}
    for boundary in boundaries:
        row_count = int(result[f"{boundary.name}_row_count"])
        fraud_count = int(result[f"{boundary.name}_fraud_count"])
        partitions[boundary.name] = {
            "TransactionDT_min": boundary.start,
            "TransactionDT_max": boundary.end,
            "row_count": row_count,
            "fraud_count": fraud_count,
            "fraud_rate": fraud_count / row_count if row_count else 0.0,
        }

    return {
        "chronological_ordering_field": CHRONOLOGICAL_FIELD,
        "boundary_assignment": "TransactionDT only; no shuffle, TransactionID, or label is used to assign partitions.",
        "boundaries": {
            boundary.name: {"TransactionDT_min": boundary.start, "TransactionDT_max": boundary.end}
            for boundary in boundaries
        },
        "expected_TransactionDT_range": {
            "min": expected_min,
            "max": expected_max,
        },
        "observed_TransactionDT_range": {"min": observed_min, "max": observed_max},
        "partitions": partitions,
        "checks": {
            "boundaries_non_overlapping": True,
            "boundaries_contiguous": True,
            "expected_TransactionDT_range_present": True,
            "rows_outside_locked_boundaries_count": int(
                result["outside_locked_boundaries_count"]
            ),
            "total_row_count": int(result["total_row_count"]),
        },
        "final_test_policy": "Final-test metrics are for one-time final evaluation only and must not be used for model tuning.",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/validation/partition_summary.json"),
        help="Small JSON summary written after successful validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_partitions(scan_transactions(args.transaction))
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"validation_boundaries.py: {error}", file=sys.stderr)
        raise
