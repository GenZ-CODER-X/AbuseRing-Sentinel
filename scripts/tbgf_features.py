"""Label-free, strict-prior event-time temporal behavioral graph features."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Set, Tuple

import polars as pl


# ---------------------------------------------------------------------------
# Graph entity construction
# ---------------------------------------------------------------------------

CARD_COMPONENTS = (
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
)

CARD_ENTITY_COLUMN = "graph_card_entity"
DEVICE_ENTITY_COLUMN = "DeviceInfo"
ADDRESS_ENTITY_COLUMN = "graph_address_entity"

TBGF_INPUT_COLUMNS = (
    "TransactionID",
    "TransactionDT",
    *CARD_COMPONENTS,
    "DeviceInfo",
    "addr1",
    "TransactionAmt",
)

# ---------------------------------------------------------------------------
# E7 feature contract
# ---------------------------------------------------------------------------

TBGF_FEATURES = (
    "tb_graph_card_recency",
    "tb_graph_card_velocity",
    "tb_graph_is_novel_card_device",
    "tb_graph_card_amt_mean",
    "tb_graph_card_amt_zscore",
    "tb_graph_card_device_stability",
    "tb_graph_pair_recency",
    "tb_graph_pair_velocity",
    "tb_graph_card_address_stability",
)

# ---------------------------------------------------------------------------
# Temporal configuration
# ---------------------------------------------------------------------------

RECENT_WINDOW = 600       # 10 minutes
PRIOR_WINDOW = 600        # preceding 10 minutes
MAX_AGE = 30 * 86400      # retain 30 days of temporal history
MAX_NOVELTY_PAIRS = 10_000_000


# ---------------------------------------------------------------------------
# Event preparation
# ---------------------------------------------------------------------------

def prepare_tbgf_events(events: pl.DataFrame) -> pl.DataFrame:
    """Prepare graph identifiers using only non-label transaction fields.

    A card entity is created only when all six card components are present.
    The six card components form one full-card signature; they are not treated
    as independent graph nodes.

    No label-derived information is accepted or constructed here.
    """
    missing = set(TBGF_INPUT_COLUMNS).difference(events.columns)

    if missing:
        raise ValueError(
            f"TBGF input lacks required columns: {sorted(missing)}"
        )

    complete_card = pl.col(CARD_COMPONENTS[0]).is_not_null()

    for column in CARD_COMPONENTS[1:]:
        complete_card = complete_card & pl.col(column).is_not_null()

    return (
        events
        .select(*TBGF_INPUT_COLUMNS)
        .with_columns(
            pl.when(complete_card)
            .then(
                pl.concat_str(
                    [
                        pl.col(column).cast(pl.String)
                        for column in CARD_COMPONENTS
                    ],
                    separator="|",
                )
            )
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias(CARD_ENTITY_COLUMN),

            pl.col("addr1")
            .cast(pl.String)
            .alias(ADDRESS_ENTITY_COLUMN),
        )
        .select(
            "TransactionID",
            "TransactionDT",
            CARD_ENTITY_COLUMN,
            DEVICE_ENTITY_COLUMN,
            ADDRESS_ENTITY_COLUMN,
            "TransactionAmt",
        )
    )


# ---------------------------------------------------------------------------
# Online amount statistics
# ---------------------------------------------------------------------------

def _welford_update(
    n: float,
    mean: float,
    M2: float,
    x: float,
) -> Tuple[float, float, float]:
    """Update online mean/variance statistics."""
    n += 1

    delta = x - mean
    mean += delta / n
    M2 += delta * (x - mean)

    return n, mean, M2


def _welford_finalize(
    n: float,
    mean: float,
    M2: float,
) -> Tuple[float, float]:
    """Return mean and sample variance."""
    if n < 2:
        return mean, 0.0

    return mean, M2 / (n - 1)


# ---------------------------------------------------------------------------
# Main temporal graph feature construction
# ---------------------------------------------------------------------------

def build_tbgf_features(events: pl.DataFrame) -> pl.DataFrame:
    """Build strict-prior temporal behavioral graph features.

    Leakage contract
    ----------------
    For every TransactionDT batch:

    1. Features are computed exclusively from state produced by
       strictly earlier TransactionDT values.
    2. All rows sharing the current TransactionDT see identical
       pre-batch state.
    3. The entire current batch is incorporated into state only after
       all its feature values have been emitted.
    4. No label, fraud count, fraud rate, target encoding, or future
       transaction information is used.
    """

    required = {
        "TransactionID",
        "TransactionDT",
        CARD_ENTITY_COLUMN,
        DEVICE_ENTITY_COLUMN,
        ADDRESS_ENTITY_COLUMN,
        "TransactionAmt",
    }

    missing = required.difference(events.columns)

    if missing:
        raise ValueError(
            f"Prepared TBGF events lack required columns: {sorted(missing)}"
        )

    if events.get_column("TransactionID").n_unique() != events.height:
        raise ValueError(
            "TransactionID must be unique when attaching TBGF features"
        )

    # -----------------------------------------------------------------------
    # Historical state
    # -----------------------------------------------------------------------

    # card -> prior timestamps
    card_timestamps: Dict[str, deque] = {}

    # card -> (n, mean, M2)
    card_amt_stats: Dict[str, Tuple[float, float, float]] = {}

    # card -> number of prior card transactions
    card_prior_count: Dict[str, int] = {}

    # (card, device) -> prior timestamps
    pair_timestamps: Dict[Tuple[str, str], deque] = {}

    # (card, device) -> number of prior transactions
    pair_prior_count: Dict[Tuple[str, str], int] = {}

    # (card, addr) -> prior transactions
    card_address_prior_count: Dict[Tuple[str, str], int] = {}

    # card -> number of prior transactions with an address
    card_address_total_count: Dict[str, int] = {}

    # Pair novelty state.
    novelty_deque: deque = deque()
    novelty_set: Set[Tuple[str, str]] = set()

    feature_rows: list[dict[str, Any]] = []

    # Deterministic event ordering.
    ordered = events.sort(
        ["TransactionDT", "TransactionID"],
    )

    # -----------------------------------------------------------------------
    # Process one TransactionDT batch at a time.
    # -----------------------------------------------------------------------

    for batch in ordered.partition_by(
        "TransactionDT",
        maintain_order=True,
    ):
        batch_rows = list(batch.iter_rows(named=True))

        if not batch_rows:
            continue

        batch_now = batch_rows[0]["TransactionDT"]

        recent_cutoff = batch_now - RECENT_WINDOW
        prior_cutoff = batch_now - RECENT_WINDOW - PRIOR_WINDOW

        # ================================================================
        # FIRST PASS
        #
        # Calculate features using ONLY state from earlier batches.
        # ================================================================

        for row in batch_rows:
            card = row[CARD_ENTITY_COLUMN]
            device = row[DEVICE_ENTITY_COLUMN]
            address = row[ADDRESS_ENTITY_COLUMN]
            amount = row["TransactionAmt"]

            # ------------------------------------------------------------
            # Defaults
            # ------------------------------------------------------------

            card_recency = None
            card_velocity = 0.0

            is_novel_card_device = 1.0

            card_amt_mean = None
            card_amt_zscore = 0.0

            card_device_stability = 0.0

            pair_recency = None
            pair_velocity = 0.0

            card_address_stability = 0.0

            # ------------------------------------------------------------
            # Card-level temporal features
            # ------------------------------------------------------------

            if card is not None:

                if card not in card_timestamps:
                    card_timestamps[card] = deque()

                if card not in card_amt_stats:
                    card_amt_stats[card] = (0.0, 0.0, 0.0)

                if card not in card_prior_count:
                    card_prior_count[card] = 0

                timestamps = card_timestamps[card]

                # Keep bounded temporal history.
                while timestamps and timestamps[0] < batch_now - MAX_AGE:
                    timestamps.popleft()

                # --------------------------------------------------------
                # Card recency
                # --------------------------------------------------------

                if timestamps:
                    card_recency = batch_now - timestamps[-1]

                # --------------------------------------------------------
                # Card velocity
                #
                # recent activity - preceding-window activity
                # --------------------------------------------------------

                recent_count = 0
                prior_count = 0

                for ts in timestamps:

                    if ts >= recent_cutoff:
                        recent_count += 1

                    elif ts >= prior_cutoff:
                        prior_count += 1

                card_velocity = float(
                    recent_count - prior_count
                )

                # --------------------------------------------------------
                # Historical amount statistics
                # --------------------------------------------------------

                n, mean, M2 = card_amt_stats[card]

                if n > 0:

                    card_amt_mean = mean

                    if n >= 2:

                        _, variance = _welford_finalize(
                            n,
                            mean,
                            M2,
                        )

                        if variance > 0:
                            card_amt_zscore = (
                                amount - mean
                            ) / (variance ** 0.5)

                # --------------------------------------------------------
                # Card-device stability
                #
                # prior(card,device)
                # -----------------
                # prior(card)
                # --------------------------------------------------------

                if device is not None:

                    pair_key = (card, device)

                    prior_pair_count = pair_prior_count.get(
                        pair_key,
                        0,
                    )

                    prior_card_count = card_prior_count.get(
                        card,
                        0,
                    )

                    if prior_card_count > 0:
                        card_device_stability = (
                            prior_pair_count
                            / prior_card_count
                        )

                # --------------------------------------------------------
                # Card-address stability
                #
                # prior(card,address)
                # ------------------
                # prior(card)
                # --------------------------------------------------------

                if address is not None:

                    card_address_key = (
                        card,
                        address,
                    )

                    prior_card_address = (
                        card_address_prior_count.get(
                            card_address_key,
                            0,
                        )
                    )

                    prior_card_address_total = (
                        card_address_total_count.get(
                            card,
                            0,
                        )
                    )

                    if prior_card_address_total > 0:
                        card_address_stability = (
                            prior_card_address
                            / prior_card_address_total
                        )

            # ------------------------------------------------------------
            # Card-device pair features
            # ------------------------------------------------------------

            if card is not None and device is not None:

                pair_key = (card, device)

                pair_history = pair_timestamps.get(
                    pair_key
                )

                if pair_history is not None:

                    # Bound pair history.
                    while (
                        pair_history
                        and pair_history[0]
                        < batch_now - MAX_AGE
                    ):
                        pair_history.popleft()

                    # ----------------------------------------------------
                    # Pair recency
                    # ----------------------------------------------------

                    if pair_history:
                        pair_recency = (
                            batch_now
                            - pair_history[-1]
                        )

                    # ----------------------------------------------------
                    # Pair velocity
                    # ----------------------------------------------------

                    recent_pair_count = 0
                    prior_pair_window_count = 0

                    for ts in pair_history:

                        if ts >= recent_cutoff:
                            recent_pair_count += 1

                        elif ts >= prior_cutoff:
                            prior_pair_window_count += 1

                    pair_velocity = float(
                        recent_pair_count
                        - prior_pair_window_count
                    )

                # --------------------------------------------------------
                # Pair novelty
                # --------------------------------------------------------

                is_novel_card_device = (
                    0.0
                    if pair_key in novelty_set
                    else 1.0
                )

            # ------------------------------------------------------------
            # Emit features.
            #
            # IMPORTANT:
            # No state is modified during this first pass.
            # ------------------------------------------------------------

            feature_rows.append(
                {
                    "TransactionID": row["TransactionID"],

                    "tb_graph_card_recency": (
                        card_recency
                    ),

                    "tb_graph_card_velocity": (
                        card_velocity
                    ),

                    "tb_graph_is_novel_card_device": (
                        float(is_novel_card_device)
                    ),

                    "tb_graph_card_amt_mean": (
                        card_amt_mean
                    ),

                    "tb_graph_card_amt_zscore": (
                        float(card_amt_zscore)
                    ),

                    "tb_graph_card_device_stability": (
                        float(card_device_stability)
                    ),

                    "tb_graph_pair_recency": (
                        pair_recency
                    ),

                    "tb_graph_pair_velocity": (
                        float(pair_velocity)
                    ),

                    "tb_graph_card_address_stability": (
                        float(card_address_stability)
                    ),
                }
            )

        # ================================================================
        # SECOND PASS
        #
        # Only after ALL rows in this TransactionDT batch have been scored
        # do we update historical state.
        # ================================================================

        for row in batch_rows:

            card = row[CARD_ENTITY_COLUMN]
            device = row[DEVICE_ENTITY_COLUMN]
            address = row[ADDRESS_ENTITY_COLUMN]
            amount = row["TransactionAmt"]

            # ------------------------------------------------------------
            # Card state
            # ------------------------------------------------------------

            if card is not None:

                if card not in card_timestamps:
                    card_timestamps[card] = deque()

                if card not in card_amt_stats:
                    card_amt_stats[card] = (
                        0.0,
                        0.0,
                        0.0,
                    )

                if card not in card_prior_count:
                    card_prior_count[card] = 0

                timestamps = card_timestamps[card]

                while (
                    timestamps
                    and timestamps[0]
                    < batch_now - MAX_AGE
                ):
                    timestamps.popleft()

                timestamps.append(batch_now)

                # Historical amount statistics.
                n, mean, M2 = card_amt_stats[card]

                n, mean, M2 = _welford_update(
                    n,
                    mean,
                    M2,
                    amount,
                )

                card_amt_stats[card] = (
                    n,
                    mean,
                    M2,
                )

                card_prior_count[card] += 1

            # ------------------------------------------------------------
            # Card-device pair state
            # ------------------------------------------------------------

            if card is not None and device is not None:

                pair_key = (card, device)

                if pair_key not in pair_timestamps:
                    pair_timestamps[pair_key] = deque()

                pair_history = pair_timestamps[pair_key]

                while (
                    pair_history
                    and pair_history[0]
                    < batch_now - MAX_AGE
                ):
                    pair_history.popleft()

                pair_history.append(batch_now)

                pair_prior_count[pair_key] = (
                    pair_prior_count.get(
                        pair_key,
                        0,
                    )
                    + 1
                )

                # Novelty membership.
                if pair_key not in novelty_set:

                    novelty_set.add(pair_key)
                    novelty_deque.append(pair_key)

                    if (
                        len(novelty_set)
                        > MAX_NOVELTY_PAIRS
                    ):
                        oldest = novelty_deque.popleft()
                        novelty_set.discard(oldest)

            # ------------------------------------------------------------
            # Card-address state
            # ------------------------------------------------------------

            if card is not None and address is not None:

                card_address_key = (
                    card,
                    address,
                )

                card_address_prior_count[
                    card_address_key
                ] = (
                    card_address_prior_count.get(
                        card_address_key,
                        0,
                    )
                    + 1
                )

                card_address_total_count[card] = (
                    card_address_total_count.get(
                        card,
                        0,
                    )
                    + 1
                )

    # -----------------------------------------------------------------------
    # Final feature frame
    # -----------------------------------------------------------------------

    result = pl.DataFrame(feature_rows)

    expected_columns = [
        "TransactionID",
        *TBGF_FEATURES,
    ]

    missing_output = set(expected_columns).difference(
        result.columns
    )

    if missing_output:
        raise RuntimeError(
            "TBGF implementation failed to emit required "
            f"features: {sorted(missing_output)}"
        )

    return result.select(*expected_columns)


# ---------------------------------------------------------------------------
# Artifact / audit specification
# ---------------------------------------------------------------------------

def tbgf_feature_specification() -> dict[str, Any]:
    """Return artifact-ready evidence of the E7 TBGF contract."""

    return {
        "graph_entity_nodes": [
            "full_card_signature",
            "DeviceInfo",
            "addr1",
        ],

        "full_card_signature": (
            "Concatenation of card1-card6 only when all six "
            "raw fields are non-null; components are not "
            "independent graph nodes."
        ),

        "graph_input_columns": list(
            TBGF_INPUT_COLUMNS
        ),

        "forbidden_inputs": [
            "isFraud",
            "target encoding",
            "fraud rates",
            "fraud counts",
            "future transactions",
        ],

        "features": list(TBGF_FEATURES),

        "feature_semantics": {
            "tb_graph_card_recency": (
                "Seconds since the previous transaction "
                "for the same full card signature."
            ),

            "tb_graph_card_velocity": (
                "Prior-card transaction count in the "
                "recent 10-minute window minus the "
                "preceding 10-minute window."
            ),

            "tb_graph_is_novel_card_device": (
                "1 when the card-device pair has not "
                "previously appeared; otherwise 0."
            ),

            "tb_graph_card_amt_mean": (
                "Historical mean TransactionAmt for the "
                "same card using strictly prior events."
            ),

            "tb_graph_card_amt_zscore": (
                "Current TransactionAmt standardized against "
                "historical card-level amount statistics."
            ),

            "tb_graph_card_device_stability": (
                "Historical frequency of the current device "
                "within the current card's prior transactions."
            ),

            "tb_graph_pair_recency": (
                "Seconds since the previous transaction "
                "for the same card-device pair."
            ),

            "tb_graph_pair_velocity": (
                "Prior card-device transaction count in the "
                "recent 10-minute window minus the preceding "
                "10-minute window."
            ),

            "tb_graph_card_address_stability": (
                "Historical frequency of the current addr1 "
                "within the current card's prior transactions."
            ),
        },

        "causality": (
            "Features for a TransactionDT batch read state "
            "from strictly earlier TransactionDT values. "
            "All same-timestamp rows are scored from identical "
            "pre-batch state and update state only afterward."
        ),
    }


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("TBGF feature contract:")
    for feature in TBGF_FEATURES:
        print(f"  - {feature}")
