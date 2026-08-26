"""Label-free, strict-prior event-time temporal behavioral graph feature construction for E4.

Adds three card-device relationship features:
- tb_graph_card_device_count
- tb_graph_card_device_recency
- tb_graph_card_device_strength
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import polars as pl

# Reuse the same card components and entity definition as TBGF
CARD_COMPONENTS = ("card1", "card2", "card3", "card4", "card5", "card6")
CARD_ENTITY_COLUMN = "graph_card_entity"
DEVICE_ENTITY_COLUMN = "DeviceInfo"
ADDRESS_ENTITY_COLUMN = "graph_address_entity"
E4_INPUT_COLUMNS = ("TransactionID", "TransactionDT", *CARD_COMPONENTS, "DeviceInfo", "addr1")

E4_FEATURES = (
    "tb_graph_card_device_count",
    "tb_graph_card_device_recency",
    "tb_graph_card_device_strength",
)

# Configuration for sliding windows (seconds) - same as TBGF for consistency
MAX_AGE = 30 * 86400  # 30 days horizon for timestamp pruning


def prepare_e4_events(events: pl.DataFrame) -> pl.DataFrame:
    """Derive graph identifiers from raw, non-label event fields only.

    Same as TBGF preparation.
    """
    missing = set(E4_INPUT_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(f"E4 input lacks required columns: {sorted(missing)}")
    complete_card = pl.col(CARD_COMPONENTS[0]).is_not_null()
    for column in CARD_COMPONENTS[1:]:
        complete_card = complete_card & pl.col(column).is_not_null()
    return events.select(*E4_INPUT_COLUMNS).with_columns(
        pl.when(complete_card)
        .then(pl.concat_str([pl.col(column).cast(pl.String) for column in CARD_COMPONENTS], separator="|"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(CARD_ENTITY_COLUMN),
        pl.col("addr1").cast(pl.String).alias(ADDRESS_ENTITY_COLUMN),
    ).select("TransactionID", "TransactionDT", CARD_ENTITY_COLUMN, DEVICE_ENTITY_COLUMN, ADDRESS_ENTITY_COLUMN)


def build_e4_features(events: pl.DataFrame) -> pl.DataFrame:
    """Compute E4 features from events strictly earlier than each event's timestamp.

    Rows sharing a TransactionDT are all scored from the same pre-batch state,
    then the complete batch updates state. No label field is accepted or used.
    """
    required = {"TransactionID", "TransactionDT", CARD_ENTITY_COLUMN, DEVICE_ENTITY_COLUMN, ADDRESS_ENTITY_COLUMN}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Prepared E4 events lack required columns: {sorted(missing)}")
    if events.get_column("TransactionID").n_unique() != events.height:
        raise ValueError("TransactionID must be unique when attaching E4 features")

    # State variables
    # For each card: prior transaction count (unbounded, like TBGF)
    card_prior_count: Dict[str, int] = {}
    # For each (card, device) pair: prior count and most recent timestamp
    card_device_prior_count: Dict[Tuple[str, str], int] = {}
    card_device_most_recent_ts: Dict[Tuple[str, str], int] = {}

    feature_rows: list[dict[str, Any]] = []

    ordered = events.sort(["TransactionDT", "TransactionID"])
    for batch in ordered.partition_by("TransactionDT", maintain_order=True):
        batch_rows = list(batch.iter_rows(named=True))
        batch_now = batch_rows[0]["TransactionDT"] if batch_rows else 0

        # Pre-compute features for each row using pre-batch state
        for row in batch_rows:
            card = row[CARD_ENTITY_COLUMN]
            device = row[DEVICE_ENTITY_COLUMN]
            now = batch_now

            # Initialize default feature values
            count = 0
            recency = None
            strength = 0.0

            # Process only if card is not None (complete card)
            if card is not None:
                # Get prior card count for strength denominator
                prior_card_cnt = card_prior_count.get(card, 0)

                if device is not None:
                    pair_key = (card, device)
                    # Prior pair count
                    prior_pair_cnt = card_device_prior_count.get(pair_key, 0)
                    count = prior_pair_cnt

                    # Recency: most recent timestamp within MAX_AGE
                    most_recent = card_device_most_recent_ts.get(pair_key)
                    if most_recent is not None and (now - most_recent) <= MAX_AGE:
                        recency = now - most_recent
                    else:
                        recency = None

                    # Strength: prior_pair_cnt / prior_card_cnt if prior_card_cnt > 0 else 0.0
                    if prior_card_cnt > 0:
                        strength = prior_pair_cnt / prior_card_cnt
                    else:
                        strength = 0.0
                else:
                    # missing device -> count=0, recency=None, strength=0.0
                    pass
            else:
                # missing card -> count=0, recency=None, strength=0.0
                pass

            feature_rows.append({
                "TransactionID": row["TransactionID"],
                "tb_graph_card_device_count": float(count),
                "tb_graph_card_device_recency": float(recency) if recency is not None else None,
                "tb_graph_card_device_strength": float(strength),
            })

        # Second pass: update state with the entire batch
        for row in batch_rows:
            card = row[CARD_ENTITY_COLUMN]
            device = row[DEVICE_ENTITY_COLUMN]
            now = batch_now

            # Update card prior count (if card present)
            if card is not None:
                card_prior_count[card] = card_prior_count.get(card, 0) + 1

            # Update pair state (if both card and device present)
            if card is not None and device is not None:
                pair_key = (card, device)
                # Update pair count
                card_device_prior_count[pair_key] = card_device_prior_count.get(pair_key, 0) + 1
                # Update most recent timestamp
                card_device_most_recent_ts[pair_key] = now

    # Convert to DataFrame, preserving order
    return pl.DataFrame(feature_rows).select("TransactionID", *E4_FEATURES)


def e4_feature_specification() -> dict[str, Any]:
    """Return compact artifact-ready evidence of the E4 contract."""
    return {
        "graph_entity_nodes": ["full_card_signature", "DeviceInfo", "addr1"],
        "full_card_signature": "Concatenation of card1-card6 only when all six raw fields are non-null; components are not independent graph nodes.",
        "graph_input_columns": list(E4_INPUT_COLUMNS),
        "forbidden_inputs": ["isFraud", "target encoding", "fraud rates", "fraud counts", "future transactions"],
        "features": list(E4_FEATURES),
        "causality": "Features for a TransactionDT batch read state from strictly earlier TransactionDT values; all same-timestamp rows update state only after feature emission.",
    }


if __name__ == "__main__":
    # Simple self-test (can be expanded)
    pass
