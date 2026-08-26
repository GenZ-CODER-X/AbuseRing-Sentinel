"""Label-free, strict-prior event-time temporal behavioral graph feature construction."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Set, Tuple

import polars as pl

# Entity construction (same as Model C) plus TransactionAmt for amount features
CARD_COMPONENTS = ("card1", "card2", "card3", "card4", "card5", "card6")
CARD_ENTITY_COLUMN = "graph_card_entity"
DEVICE_ENTITY_COLUMN = "DeviceInfo"
ADDRESS_ENTITY_COLUMN = "graph_address_entity"
TBGF_INPUT_COLUMNS = ("TransactionID", "TransactionDT", *CARD_COMPONENTS, "DeviceInfo", "addr1", "TransactionAmt")

TBGF_FEATURES = (
    "tb_graph_card_recency",
    "tb_graph_card_velocity",
    "tb_graph_is_novel_card_device",
    "tb_graph_card_amt_mean",
    "tb_graph_card_amt_zscore",
    "tb_graph_card_device_stability",
)

# Configuration for sliding windows (seconds)
RECENT_WINDOW = 600  # 10 minutes
PRIOR_WINDOW = 600   # 10 minutes prior to recent window
MAX_AGE = 30 * 86400  # 30 days horizon for timestamp pruning (we will prune to keep memory bounded)
MAX_NOVELTY_PAIRS = 10_000_000  # bounded novelty set size


def prepare_tbgf_events(events: pl.DataFrame) -> pl.DataFrame:
    """Derive graph identifiers from raw, non-label event fields only.

    A card entity exists only when every card component is present. Components
    are serialized together into one identifier and never become graph nodes on
    their own. ``addr1`` is serialized only for stable dictionary keys.
    """
    missing = set(TBGF_INPUT_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(f"TBGF input lacks required columns: {sorted(missing)}")
    complete_card = pl.col(CARD_COMPONENTS[0]).is_not_null()
    for column in CARD_COMPONENTS[1:]:
        complete_card = complete_card & pl.col(column).is_not_null()
    return events.select(*TBGF_INPUT_COLUMNS).with_columns(
        pl.when(complete_card)
        .then(pl.concat_str([pl.col(column).cast(pl.String) for column in CARD_COMPONENTS], separator="|"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(CARD_ENTITY_COLUMN),
        pl.col("addr1").cast(pl.String).alias(ADDRESS_ENTITY_COLUMN),
    ).select("TransactionID", "TransactionDT", CARD_ENTITY_COLUMN, DEVICE_ENTITY_COLUMN, ADDRESS_ENTITY_COLUMN, "TransactionAmt")


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


def build_tbgf_features(events: pl.DataFrame) -> pl.DataFrame:
    """Compute TBGF features from events strictly earlier than each event's timestamp.

    Rows sharing a TransactionDT are all scored from the same pre-batch state,
    then the complete batch updates state. No label field is accepted or used.
    """
    required = {"TransactionID", "TransactionDT", CARD_ENTITY_COLUMN, DEVICE_ENTITY_COLUMN, ADDRESS_ENTITY_COLUMN, "TransactionAmt"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Prepared TBGF events lack required columns: {sorted(missing)}")
    if events.get_column("TransactionID").n_unique() != events.height:
        raise ValueError("TransactionID must be unique when attaching TBGF features")

    # State variables
    card_timestamps: Dict[str, deque] = {}          # card -> deque of prior timestamps (pruned to MAX_AGE)
    card_amt_stats: Dict[str, Tuple[float, float, float]] = {}  # card -> (n, mean, M2)
    card_prior_count: Dict[str, int] = {}           # card -> prior transaction count (for stability denominator)
    # For novelty and pair count we need a FIFO bounded set
    novelty_deque: deque = deque()                  # FIFO order of (card, device) pairs
    novelty_set: Set[Tuple[str, str]] = set()       # for O(1) membership test
    card_device_prior_count: Dict[Tuple[str, str], int] = {}  # (card, device) -> prior pair count

    feature_rows: list[dict[str, Any]] = []

    ordered = events.sort(["TransactionDT", "TransactionID"])
    for batch in ordered.partition_by("TransactionDT", maintain_order=True):
        batch_rows = list(batch.iter_rows(named=True))
        # All rows in batch share the same TransactionDT
        batch_now = batch_rows[0]["TransactionDT"] if batch_rows else 0

        # Pre-compute velocity counts based on pre-batch state (timestamps from earlier batches only)
        recent_cutoff = batch_now - RECENT_WINDOW
        prior_cutoff = batch_now - 2 * RECENT_WINDOW
        recent_count = 0
        prior_count = 0
        # We'll compute per card later because each card has its own timestamps deque.
        # Instead we will compute per card inside the row loop but we can still reuse the cutoffs.
        # To avoid scanning each card's deque multiple times, we could compute counts per card and cache.
        # Given the number of distinct cards per batch is limited, we'll compute per card.

        # First pass: compute features for each row using pre-batch state
        for row in batch_rows:
            card = row[CARD_ENTITY_COLUMN]
            device = row[DEVICE_ENTITY_COLUMN]
            now = batch_now  # same for all rows in batch
            amount = row["TransactionAmt"]

            # Initialize default feature values
            recency = None
            velocity = 0.0
            is_novel = 1.0
            amt_mean = None
            amt_zscore = 0.0
            stability = 0.0

            # Process card-related state only if card is not None
            if card is not None:
                # Ensure state exists for this card
                if card not in card_timestamps:
                    card_timestamps[card] = deque()
                    card_amt_stats[card] = (0.0, 0.0, 0.0)  # n, mean, M2
                    card_prior_count[card] = 0

                timestamps_deque = card_timestamps[card]
                # Prune timestamps older than now - MAX_AGE (based on current now)
                while timestamps_deque and timestamps_deque[0] < now - MAX_AGE:
                    timestamps_deque.popleft()

                # Recency: seconds since last transaction, or None if never seen
                if timestamps_deque:
                    recency = now - timestamps_deque[-1]  # most recent timestamp is at the right end
                else:
                    recency = None

                # Velocity: (count in recent window) - (count in prior window)
                # Scan the deque for this card
                rc = 0
                pc = 0
                for ts in timestamps_deque:
                    if ts >= recent_cutoff:
                        rc += 1
                    if ts >= prior_cutoff and ts < recent_cutoff:
                        pc += 1
                velocity = rc - pc

                # Amount mean and z-score
                n, mean, M2 = card_amt_stats[card]
                if n == 0:
                    amt_mean = None
                    amt_zscore = 0.0
                else:
                    amt_mean = mean
                    if n >= 2:
                        _, variance = _welford_finalize(n, mean, M2)
                        if variance > 0:
                            amt_zscore = (amount - mean) / (variance ** 0.5)
                        else:
                            amt_zscore = 0.0
                    else:
                        amt_zscore = 0.0

                # Stability: prior_(card,device)_count / prior_card_count
                # Note: we need prior pair count and prior card count from state (pre-batch)
                prior_pair_count = card_device_prior_count.get((card, device), 0) if device is not None else 0
                prior_card_cnt = card_prior_count[card]
                if prior_card_cnt > 0:
                    stability = prior_pair_count / prior_card_cnt
                else:
                    stability = 0.0
            else:
                # card is None: card-related features remain default
                # For novelty and stability we still need to handle device None case
                pass

            # Novelty and pair count updates (depend on both card and device)
            # If either card or device is None, treat as unseen and do not update state
            if card is not None and device is not None:
                pair_key = (card, device)
                # Novelty: 1 if never seen before
                is_novel = 0.0 if pair_key in novelty_set else 1.0
                # For stability we already computed above using prior counts
            else:
                # missing card or device -> treat as novel
                is_novel = 1.0

            feature_rows.append({
                "TransactionID": row["TransactionID"],
                "tb_graph_card_recency": recency,
                "tb_graph_card_velocity": float(velocity),
                "tb_graph_is_novel_card_device": float(is_novel),
                "tb_graph_card_amt_mean": amt_mean,
                "tb_graph_card_amt_zscore": float(amt_zscore),
                "tb_graph_card_device_stability": float(stability),
            })

        # Second pass: update state with the entire batch
        for row in batch_rows:
            card = row[CARD_ENTITY_COLUMN]
            device = row[DEVICE_ENTITY_COLUMN]
            now = batch_now
            amount = row["TransactionAmt"]

            # Update card-related state only if card is not None
            if card is not None:
                # Ensure state exists
                if card not in card_timestamps:
                    card_timestamps[card] = deque()
                    card_amt_stats[card] = (0.0, 0.0, 0.0)
                    card_prior_count[card] = 0

                timestamps_deque = card_timestamps[card]
                # Prune old timestamps before appending new ones (based on now)
                while timestamps_deque and timestamps_deque[0] < now - MAX_AGE:
                    timestamps_deque.popleft()
                timestamps_deque.append(now)

                # Update Welford stats
                n, mean, M2 = card_amt_stats[card]
                n, mean, M2 = _welford_update(n, mean, M2, amount)
                card_amt_stats[card] = (n, mean, M2)

                # Update prior card count (will be incremented after processing the row)
                card_prior_count[card] += 1

            # Update pair count and novelty set only if both card and device are not None
            if card is not None and device is not None:
                pair_key = (card, device)
                # Update pair count
                prior_pair_count = card_device_prior_count.get(pair_key, 0)
                card_device_prior_count[pair_key] = prior_pair_count + 1
                # Update novelty set with FIFO bounding
                if pair_key not in novelty_set:
                    novelty_set.add(pair_key)
                    novelty_deque.append(pair_key)
                    # Enforce max size
                    if len(novelty_set) > MAX_NOVELTY_PAIRS:
                        oldest = novelty_deque.popleft()
                        novelty_set.discard(oldest)

    return pl.DataFrame(feature_rows).select("TransactionID", *TBGF_FEATURES)


def tbgf_feature_specification() -> dict[str, Any]:
    """Return compact artifact-ready evidence of the TBGF contract."""
    return {
        "graph_entity_nodes": ["full_card_signature", "DeviceInfo", "addr1"],
        "full_card_signature": "Concatenation of card1-card6 only when all six raw fields are non-null; components are not independent graph nodes.",
        "graph_input_columns": list(TBGF_INPUT_COLUMNS),
        "forbidden_inputs": ["isFraud", "target encoding", "fraud rates", "fraud counts", "future transactions"],
        "features": list(TBGF_FEATURES),
        "causality": "Features for a TransactionDT batch read state from strictly earlier TransactionDT values; all same-timestamp rows update state only after feature emission.",
    }


if __name__ == "__main__":
    # Simple self-test (can be expanded)
    pass
