"""Label-free, strict-prior event-time temporal UID feature construction.

Computes five temporal aggregates over the UID_G field (card1-6 + addr1):
- tb_uid_prior_count
- tb_uid_amt_mean
- tb_uid_amt_std
- tb_uid_recency
- tb_uid_amt_zscore
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import polars as pl

UID_ENTITY_COLUMN = "uid_g"
UID_INPUT_COLUMNS = (
    "TransactionID",
    "TransactionDT",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "TransactionAmt",
)
UID_FEATURES = (
    "tb_uid_prior_count",
    "tb_uid_amt_mean",
    "tb_uid_amt_std",
    "tb_uid_recency",
    "tb_uid_amt_zscore",
)

# Configuration for sliding windows (seconds) - same as TBGF for consistency
MAX_AGE = 30 * 86400  # 30 days horizon for timestamp pruning (optional, not used for UID recency)


def prepare_uid_events(events: pl.DataFrame) -> pl.DataFrame:
    """Validate and select UID-related columns."""
    missing = set(UID_INPUT_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(f"UID input lacks required columns: {sorted(missing)}")
    return events.select(*UID_INPUT_COLUMNS)


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


def build_uid_features(events: pl.DataFrame) -> pl.DataFrame:
    """Compute UID features from events strictly earlier than each event's timestamp.

    Rows sharing a TransactionDT are all scored from the same pre-batch state,
    then the complete batch updates state. No label field is accepted or used.
    """
    required = {"TransactionID", "TransactionDT", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "TransactionAmt"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Prepared UID events lack required columns: {sorted(missing)}")
    if events.get_column("TransactionID").n_unique() != events.height:
        raise ValueError("TransactionID must be unique when attaching UID features")

    # State variables
    uid_prior_count: Dict[str, int] = {}          # uid -> prior transaction count
    uid_welford: Dict[str, Tuple[float, float, float]] = {}  # uid -> (n, mean, M2)
    uid_most_recent_ts: Dict[str, int] = {}       # uid -> most recent timestamp

    feature_rows: list[dict[str, Any]] = []

    ordered = events.sort(["TransactionDT", "TransactionID"])
    for batch in ordered.partition_by("TransactionDT", maintain_order=True):
        batch_rows = list(batch.iter_rows(named=True))
        batch_now = batch_rows[0]["TransactionDT"] if batch_rows else 0

        # Pre-compute features for each row using pre-batch state
        for row in batch_rows:
            # Compute UID: only if all card1-6 and addr1 are not null
            c1 = row["card1"]
            c2 = row["card2"]
            c3 = row["card3"]
            c4 = row["card4"]
            c5 = row["card5"]
            c6 = row["card6"]
            addr = row["addr1"]
            if None in (c1, c2, c3, c4, c5, c6, addr):
                uid = None
            else:
                # Convert to string for consistency, but we can keep original types? We'll cast to string.
                uid = f"{c1}|{c2}|{c3}|{c4}|{c5}|{c6}|{addr}"
            now = batch_now
            amount = row["TransactionAmt"]

            # Initialize default feature values
            count = 0
            amt_mean = None
            amt_std = 0.0
            recency = None
            zscore = 0.0

            if uid is not None:
                # Prior count
                count = uid_prior_count.get(uid, 0)

                # Amount mean and std
                n, mean, M2 = uid_welford.get(uid, (0.0, 0.0, 0.0))
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
                most_recent = uid_most_recent_ts.get(uid)
                if most_recent is not None:
                    recency = now - most_recent
                else:
                    recency = None

                # Z-score
                if amt_std > 0:
                    zscore = (amount - amt_mean) / amt_std
                else:
                    zscore = 0.0
            # else: missing UID -> keep defaults (count=0, mean=None, std=0, recency=None, zscore=0)

            feature_rows.append({
                "TransactionID": row["TransactionID"],
                "tb_uid_prior_count": float(count),
                "tb_uid_amt_mean": amt_mean,
                "tb_uid_amt_std": float(amt_std),
                "tb_uid_recency": float(recency) if recency is not None else None,
                "tb_uid_amt_zscore": float(zscore),
            })

        # Second pass: update state with the entire batch
        for row in batch_rows:
            # Compute UID again
            c1 = row["card1"]
            c2 = row["card2"]
            c3 = row["card3"]
            c4 = row["card4"]
            c5 = row["card5"]
            c6 = row["card6"]
            addr = row["addr1"]
            if None in (c1, c2, c3, c4, c5, c6, addr):
                uid = None
            else:
                uid = f"{c1}|{c2}|{c3}|{c4}|{c5}|{c6}|{addr}"
            now = batch_now
            amount = row["TransactionAmt"]

            if uid is not None:
                # Update prior count
                uid_prior_count[uid] = uid_prior_count.get(uid, 0) + 1

                # Update Welford stats
                n, mean, M2 = uid_welford.get(uid, (0.0, 0.0, 0.0))
                n, mean, M2 = _welford_update(n, mean, M2, amount)
                uid_welford[uid] = (n, mean, M2)

                # Update most recent timestamp
                uid_most_recent_ts[uid] = now

    # Convert to DataFrame, preserving order
    return pl.DataFrame(feature_rows).select("TransactionID", *UID_FEATURES)


def uid_feature_specification() -> dict[str, Any]:
    """Return compact artifact-ready evidence of the UID contract."""
    return {
        "graph_entity_nodes": ["card1", "card2", "card3", "card4", "card5", "card6", "addr1"],
        "graph_input_columns": list(UID_INPUT_COLUMNS),
        "forbidden_inputs": ["isFraud", "target encoding", "fraud rates", "fraud counts", "future transactions"],
        "features": list(UID_FEATURES),
        "causality": "Features for a TransactionDT batch read state from strictly earlier TransactionDT values; all same-timestamp rows update state only after feature emission.",
    }


if __name__ == "__main__":
    # Simple self-test (can be expanded)
    pass
