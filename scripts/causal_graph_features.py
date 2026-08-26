"""Label-free, strict-prior event-time graph feature construction."""

from __future__ import annotations

from typing import Any

import polars as pl


CARD_COMPONENTS = ("card1", "card2", "card3", "card4", "card5", "card6")
CARD_ENTITY_COLUMN = "graph_card_entity"
DEVICE_ENTITY_COLUMN = "DeviceInfo"
ADDRESS_ENTITY_COLUMN = "graph_address_entity"
GRAPH_INPUT_COLUMNS = ("TransactionID", "TransactionDT", *CARD_COMPONENTS, "DeviceInfo", "addr1")
GRAPH_FEATURES = (
    "graph_prior_card_transaction_count",
    "graph_prior_device_transaction_count",
    "graph_prior_address_transaction_count",
    "graph_prior_card_distinct_devices",
    "graph_prior_device_distinct_cards",
    "graph_prior_card_distinct_addresses",
    "graph_prior_address_distinct_cards",
    "graph_prior_shared_entity_connectivity",
)


def prepare_graph_events(events: pl.DataFrame) -> pl.DataFrame:
    """Derive graph identifiers from raw, non-label event fields only.

    A card entity exists only when every card component is present. Components
    are serialized together into one identifier and never become graph nodes on
    their own. ``addr1`` is serialized only for stable dictionary keys.
    """
    missing = set(GRAPH_INPUT_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(f"Graph input lacks required columns: {sorted(missing)}")
    complete_card = pl.col(CARD_COMPONENTS[0]).is_not_null()
    for column in CARD_COMPONENTS[1:]:
        complete_card = complete_card & pl.col(column).is_not_null()
    return events.select(*GRAPH_INPUT_COLUMNS).with_columns(
        pl.when(complete_card)
        .then(pl.concat_str([pl.col(column).cast(pl.String) for column in CARD_COMPONENTS], separator="|"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(CARD_ENTITY_COLUMN),
        pl.col("addr1").cast(pl.String).alias(ADDRESS_ENTITY_COLUMN),
    ).select("TransactionID", "TransactionDT", CARD_ENTITY_COLUMN, DEVICE_ENTITY_COLUMN, ADDRESS_ENTITY_COLUMN)


def _increment(mapping: dict[str, int], key: str | None) -> None:
    if key is not None:
        mapping[key] = mapping.get(key, 0) + 1


def _add_relation(mapping: dict[str, set[str]], left: str | None, right: str | None) -> None:
    if left is not None and right is not None:
        mapping.setdefault(left, set()).add(right)


def build_causal_graph_features(events: pl.DataFrame) -> pl.DataFrame:
    """Compute features from events strictly earlier than each event's timestamp.

    Rows sharing a TransactionDT are all scored from the same pre-batch state,
    then the complete batch updates state. No label field is accepted or used.
    """
    required = {"TransactionID", "TransactionDT", CARD_ENTITY_COLUMN, DEVICE_ENTITY_COLUMN, ADDRESS_ENTITY_COLUMN}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Prepared graph events lack required columns: {sorted(missing)}")
    if events.get_column("TransactionID").n_unique() != events.height:
        raise ValueError("TransactionID must be unique when attaching graph features")

    card_transaction_counts: dict[str, int] = {}
    device_transaction_counts: dict[str, int] = {}
    address_transaction_counts: dict[str, int] = {}
    card_devices: dict[str, set[str]] = {}
    device_cards: dict[str, set[str]] = {}
    card_addresses: dict[str, set[str]] = {}
    address_cards: dict[str, set[str]] = {}
    card_device_edges: set[tuple[str, str]] = set()
    card_address_edges: set[tuple[str, str]] = set()
    device_address_edges: set[tuple[str, str]] = set()
    feature_rows: list[dict[str, Any]] = []

    ordered = events.sort(["TransactionDT", "TransactionID"])
    for batch in ordered.partition_by("TransactionDT", maintain_order=True):
        batch_rows = list(batch.iter_rows(named=True))
        for row in batch_rows:
            card = row[CARD_ENTITY_COLUMN]
            device = row[DEVICE_ENTITY_COLUMN]
            address = row[ADDRESS_ENTITY_COLUMN]
            connectivity = 0
            available_pairs = 0
            if card is not None and device is not None:
                available_pairs += 1
                connectivity += int((card, device) in card_device_edges)
            if card is not None and address is not None:
                available_pairs += 1
                connectivity += int((card, address) in card_address_edges)
            if device is not None and address is not None:
                available_pairs += 1
                connectivity += int((device, address) in device_address_edges)
            feature_rows.append(
                {
                    "TransactionID": row["TransactionID"],
                    "graph_prior_card_transaction_count": card_transaction_counts.get(card, 0) if card is not None else None,
                    "graph_prior_device_transaction_count": device_transaction_counts.get(device, 0) if device is not None else None,
                    "graph_prior_address_transaction_count": address_transaction_counts.get(address, 0) if address is not None else None,
                    "graph_prior_card_distinct_devices": len(card_devices.get(card, set())) if card is not None else None,
                    "graph_prior_device_distinct_cards": len(device_cards.get(device, set())) if device is not None else None,
                    "graph_prior_card_distinct_addresses": len(card_addresses.get(card, set())) if card is not None else None,
                    "graph_prior_address_distinct_cards": len(address_cards.get(address, set())) if address is not None else None,
                    "graph_prior_shared_entity_connectivity": connectivity if available_pairs else None,
                }
            )
        # State is updated only after every same-timestamp row was emitted.
        for row in batch_rows:
            card = row[CARD_ENTITY_COLUMN]
            device = row[DEVICE_ENTITY_COLUMN]
            address = row[ADDRESS_ENTITY_COLUMN]
            _increment(card_transaction_counts, card)
            _increment(device_transaction_counts, device)
            _increment(address_transaction_counts, address)
            _add_relation(card_devices, card, device)
            _add_relation(device_cards, device, card)
            _add_relation(card_addresses, card, address)
            _add_relation(address_cards, address, card)
            if card is not None and device is not None:
                card_device_edges.add((card, device))
            if card is not None and address is not None:
                card_address_edges.add((card, address))
            if device is not None and address is not None:
                device_address_edges.add((device, address))
    return pl.DataFrame(feature_rows).select("TransactionID", *GRAPH_FEATURES)


def feature_specification() -> dict[str, Any]:
    """Return compact artifact-ready evidence of the causal graph contract."""
    return {
        "graph_entity_nodes": ["full_card_signature", "DeviceInfo", "addr1"],
        "full_card_signature": "Concatenation of card1-card6 only when all six raw fields are non-null; components are not independent graph nodes.",
        "graph_input_columns": list(GRAPH_INPUT_COLUMNS),
        "forbidden_inputs": ["isFraud", "target encoding", "fraud rates", "fraud counts", "future transactions"],
        "features": list(GRAPH_FEATURES),
        "causality": "Features for a TransactionDT batch read state from strictly earlier TransactionDT values; all same-timestamp rows update state only after feature emission.",
    }
