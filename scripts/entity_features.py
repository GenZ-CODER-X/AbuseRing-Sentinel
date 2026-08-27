"""Label-free, strict-prior event-time entity-level feature construction.

Computes seven temporal aggregates over the UID_G field (card1-6 + addr1):
- tb_entity_prior_count
- tb_entity_first_seen_age
- tb_entity_recency
- tb_entity_amt_mean
- tb_entity_amt_std
- tb_entity_amt_zscore
- tb_entity_unique_product_count
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Set

import polars as pl

ENTITY_ENTITY_COLUMN = "entity_uid"
ENTITY_INPUT_COLUMNS = (
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
    "ProductCD",
)
ENTITY_FEATURES = (
    "tb_entity_prior_count",
    "tb_entity_first_seen_age",
    "tb_entity_recency",
    "tb_entity_amt_mean",
    "tb_entity_amt_std",
    "tb_entity_amt_zscore",
    "tb_entity_unique_product_count",
)

# Configuration for sliding windows (seconds) - same as TBGF/UID for consistency
MAX_AGE = 30 * 86400  # 30 days horizon for timestamp pruning (optional, not used for entity recency)


def prepare_entity_events(events: pl.DataFrame) -> pl.DataFrame:
    """Validate and select entity-related columns."""
    missing = set(ENTITY_INPUT_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(f"Entity input lacks required columns: {sorted(missing)}")
    return events.select(*ENTITY_INPUT_COLUMNS)


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


def build_entity_features(events: pl.DataFrame) -> pl.DataFrame:
    """Compute entity features from events strictly earlier than each event's timestamp.

    Rows sharing a TransactionDT are all scored from the same pre-batch state,
    then the complete batch updates state. No label field is accepted or used.
    """
    required = {
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
        "ProductCD",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Prepared entity events lack required columns: {sorted(missing)}")
    if events.get_column("TransactionID").n_unique() != events.height:
        raise ValueError("TransactionID must be unique when attaching entity features")

    # State variables
    entity_prior_count: Dict[str, int] = {}          # entity -> prior transaction count
    entity_welford: Dict[str, Tuple[float, float, float]] = {}  # entity -> (n, mean, M2)
    entity_most_recent_ts: Dict[str, int] = {}       # entity -> most recent timestamp
    entity_first_seen_ts: Dict[str, int] = {}        # entity -> first seen timestamp
    entity_unique_products: Dict[str, Set[str]] = {} # entity -> set of unique ProductCD values seen

    feature_rows: list[dict[str, Any]] = []

    ordered = events.sort(["TransactionDT", "TransactionID"])
    for batch in ordered.partition_by("TransactionDT", maintain_order=True):
        batch_rows = list(batch.iter_rows(named=True))
        batch_now = batch_rows[0]["TransactionDT"] if batch_rows else 0

        # Pre-compute features for each row using pre-batch state
        for row in batch_rows:
            # Compute entity UID: only if all card1-6 and addr1 are not null
            c1 = row["card1"]
            c2 = row["card2"]
            c3 = row["card3"]
            c4 = row["card4"]
            c5 = row["card5"]
            c6 = row["card6"]
            addr = row["addr1"]
            if None in (c1, c2, c3, c4, c5, c6, addr):
                entity_uid = None
            else:
                # Convert to string for consistency
                entity_uid = f"{c1}|{c2}|{c3}|{c4}|{c5}|{c6}|{addr}"
            now = batch_now
            amount = row["TransactionAmt"]
            product_cd = row["ProductCD"]

            # Initialize default feature values
            prior_count = 0
            first_seen_age = 0  # 0 for unseen entities
            recency = None
            amt_mean = None
            amt_std = 0.0
            zscore = 0.0
            unique_product_count = 0

            if entity_uid is not None:
                # Prior count
                prior_count = entity_prior_count.get(entity_uid, 0)

                # First seen age
                first_seen = entity_first_seen_ts.get(entity_uid)
                if first_seen is not None:
                    first_seen_age = now - first_seen
                else:
                    first_seen_age = 0  # Will be updated to now in second pass

                # Amount mean and std
                n, mean, M2 = entity_welford.get(entity_uid, (0.0, 0.0, 0.0))
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
                most_recent = entity_most_recent_ts.get(entity_uid)
                if most_recent is not None:
                    recency = now - most_recent
                else:
                    recency = None

                # Z-score
                if amt_std > 0:
                    zscore = (amount - amt_mean) / amt_std
                else:
                    zscore = 0.0

                # Unique product count
                unique_products = entity_unique_products.get(entity_uid, set())
                unique_product_count = len(unique_products)

            # else: missing entity UID -> keep defaults (all zero/null as initialized)

            feature_rows.append({
                "TransactionID": row["TransactionID"],
                "tb_entity_prior_count": float(prior_count),
                "tb_entity_first_seen_age": float(first_seen_age),
                "tb_entity_recency": float(recency) if recency is not None else None,
                "tb_entity_amt_mean": amt_mean,
                "tb_entity_amt_std": float(amt_std),
                "tb_entity_amt_zscore": float(zscore),
                "tb_entity_unique_product_count": float(unique_product_count),
            })

        # Second pass: update state with the entire batch
        for row in batch_rows:
            # Compute entity UID again
            c1 = row["card1"]
            c2 = row["card2"]
            c3 = row["card3"]
            c4 = row["card4"]
            c5 = row["card5"]
            c6 = row["card6"]
            addr = row["addr1"]
            if None in (c1, c2, c3, c4, c5, c6, addr):
                entity_uid = None
            else:
                entity_uid = f"{c1}|{c2}|{c3}|{c4}|{c5}|{c6}|{addr}"
            now = batch_now
            amount = row["TransactionAmt"]
            product_cd = row["ProductCD"]

            if entity_uid is not None:
                # Update prior count
                entity_prior_count[entity_uid] = entity_prior_count.get(entity_uid, 0) + 1

                # Update first seen timestamp
                if entity_uid not in entity_first_seen_ts:
                    entity_first_seen_ts[entity_uid] = now

                # Update Welford stats
                n, mean, M2 = entity_welford.get(entity_uid, (0.0, 0.0, 0.0))
                n, mean, M2 = _welford_update(n, mean, M2, amount)
                entity_welford[entity_uid] = (n, mean, M2)

                # Update most recent timestamp
                entity_most_recent_ts[entity_uid] = now

                # Update unique products set
                if entity_uid not in entity_unique_products:
                    entity_unique_products[entity_uid] = set()
                entity_unique_products[entity_uid].add(str(product_cd))  # Ensure string representation

    # Convert to DataFrame, preserving order
    return pl.DataFrame(feature_rows).select("TransactionID", *ENTITY_FEATURES)


def entity_feature_specification() -> dict[str, Any]:
    """Return compact artifact-ready evidence of the entity feature contract."""
    return {
        "graph_entity_nodes": ["card1", "card2", "card3", "card4", "card5", "card6", "addr1"],
        "graph_input_columns": list(ENTITY_INPUT_COLUMNS),
        "forbidden_inputs": ["isFraud", "target encoding", "fraud rates", "fraud counts", "future transactions"],
        "features": list(ENTITY_FEATURES),
        "causality": "Features for a TransactionDT batch read state from strictly earlier TransactionDT values; all same-timestamp rows update state only after feature emission.",
    }


if __name__ == "__main__":
    # Simple self-test (can be expanded)
    print("Entity feature contract:")
    for feature in ENTITY_FEATURES:
        print(f"  - {feature}")