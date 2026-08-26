"""Label-free, strict-prior event-time temporal address feature construction.

Computes five temporal aggregates over the addr1 field:
- tb_addr_prior_count
- tb_addr_amt_mean
- tb_addr_amt_std
- tb_addr_recency
- tb_addr_amt_zscore
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import polars as pl

ADDR_ENTITY_COLUMN = "addr1"
ADDR_INPUT_COLUMNS = ("TransactionID", "TransactionDT", "addr1", "TransactionAmt")
ADDR_FEATURES = (
    "tb_addr_prior_count",
    "tb_addr_amt_mean",
    "tb_addr_amt_std",
    "tb_addr_recency",
    "tb_addr_amt_zscore",
)

# Configuration for sliding windows (seconds) - same as TBGF for consistency
MAX_AGE = 30 * 86400  # 30 days horizon for timestamp pruning (optional, not used for addr1 recency)


def prepare_addr_events(events: pl.DataFrame) -> pl.DataFrame:
    """Validate and select address-related columns."""
    missing = set(ADDR_INPUT_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(f"Addr input lacks required columns: {sorted(missing)}")
    return events.select(*ADDR_INPUT_COLUMNS)


def _welford_update(n: float, mean: float, M2: float, x: float) -> Tuple[float, float, float]:
    """Update Welford's online algorithm for mean and variance."""
    n += 1
    delta = x - mean
    mean += delta / n
    M2 += delta * (x - mean)
    return n, mean, M2


def _welford_finalize(n: float, mean: float, M2: float) -> Tuple[float, float]:
    """Return mean and sample variance (unbiased). If n < 2, variance is 0.0."""
    if n < 2:
        return mean, 0.0
    variance = M2 / (n - 1)
    return mean, variance


def build_addr_features(events: pl.DataFrame) -> pl.DataFrame:
    """Compute ADDR features from events strictly earlier than each event's timestamp.

    Rows sharing a TransactionDT are all scored from the same pre-batch state,
    then the complete batch updates state. No label field is accepted or used.
    """
    required = {"TransactionID", "TransactionDT", ADDR_ENTITY_COLUMN, "TransactionAmt"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Prepared ADDR events lack required columns: {sorted(missing)}")
    if events.get_column("TransactionID").n_unique() != events.height:
        raise ValueError("TransactionID must be unique when attaching ADDR features")

    # State variables
    addr_prior_count: Dict[str, int] = {}          # addr -> prior transaction count
    addr_welford: Dict[str, Tuple[float, float, float]] = {}  # addr -> (n, mean, M2)
    addr_most_recent_ts: Dict[str, int] = {}       # addr -> most recent timestamp

    feature_rows: list[dict[str, Any]] = []

    ordered = events.sort(["TransactionDT", "TransactionID"])
    for batch in ordered.partition_by("TransactionDT", maintain_order=True):
        batch_rows = list(batch.iter_rows(named=True))
        batch_now = batch_rows[0]["TransactionDT"] if batch_rows else 0

        # Pre-compute features for each row using pre-batch state
        for row in batch_rows:
            addr = row[ADDR_ENTITY_COLUMN]
            now = batch_now
            amount = row["TransactionAmt"]

            # Initialize default feature values
            count = 0
            amt_mean = None
            amt_std = 0.0
            recency = None
            zscore = 0.0

            if addr is not None:
                # Prior count
                count = addr_prior_count.get(addr, 0)

                # Amount mean and std
                n, mean, M2 = addr_welford.get(addr, (0.0, 0.0, 0.0))
                if n == 0:
                    amt_mean = None
                    amt_std = 0.0
                else:
                    amt_mean = mean
                    if n >= 2:
                        _, variance = _welford_finalize(n, mean, M2)
                        amt_std = variance ** 0.5 if variance > 0 else 0.0
                    else:
                        amt_std = 0.0

                # Recency: seconds since most recent prior transaction
                most_recent = addr_most_recent_ts.get(addr)
                if most_recent is not None:
                    recency = now - most_recent
                else:
                    recency = None

                # Z-score
                if amt_std > 0:
                    zscore = (amount - amt_mean) / amt_std
                else:
                    zscore = 0.0
            # else: missing addr -> keep defaults (count=0, mean=None, std=0, recency=None, zscore=0)

            feature_rows.append({
                "TransactionID": row["TransactionID"],
                "tb_addr_prior_count": float(count),
                "tb_addr_amt_mean": amt_mean,
                "tb_addr_amt_std": float(amt_std),
                "tb_addr_recency": float(recency) if recency is not None else None,
                "tb_addr_amt_zscore": float(zscore),
            })

        # Second pass: update state with the entire batch
        for row in batch_rows:
            addr = row[ADDR_ENTITY_COLUMN]
            now = batch_now
            amount = row["TransactionAmt"]

            if addr is not None:
                # Update prior count
                addr_prior_count[addr] = addr_prior_count.get(addr, 0) + 1

                # Update Welford stats
                n, mean, M2 = addr_welford.get(addr, (0.0, 0.0, 0.0))
                n, mean, M2 = _welford_update(n, mean, M2, amount)
                addr_welford[addr] = (n, mean, M2)

                # Update most recent timestamp
                addr_most_recent_ts[addr] = now

    # Convert to DataFrame, preserving order
    return pl.DataFrame(feature_rows).select("TransactionID", *ADDR_FEATURES)


def addr_feature_specification() -> dict[str, Any]:
    """Return compact artifact-ready evidence of the ADDR contract."""
    return {
        "graph_entity_nodes": ["addr1"],
        "graph_input_columns": list(ADDR_INPUT_COLUMNS),
        "forbidden_inputs": ["isFraud", "target encoding", "fraud rates", "fraud counts", "future transactions"],
        "features": list(ADDR_FEATURES),
        "causality": "Features for a TransactionDT batch read state from strictly earlier TransactionDT values; all same-timestamp rows update state only after feature emission.",
    }


if __name__ == "__main__":
    # Simple self-test (can be expanded)
    pass
