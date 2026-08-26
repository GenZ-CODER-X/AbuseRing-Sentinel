"""Label-free, strict-prior event-time graph relationship feature construction.

Computes two relationship diversity features over historical associations:
- tb_card1_device_prior_unique_count: For card1, count of distinct DeviceInfo values seen with it in past transactions
- tb_addr1_card1_prior_unique_count: For addr1, count of distinct card1 values seen with it in past transactions
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import polars as pl

GRAPH_REL_INPUT_COLUMNS = (
    "TransactionID",
    "TransactionDT",
    "card1",
    "DeviceInfo",
    "addr1",
)

GRAPH_REL_FEATURES = (
    "tb_card1_device_prior_unique_count",
    "tb_addr1_card1_prior_unique_count",
)


def prepare_graph_rel_events(events: pl.DataFrame) -> pl.DataFrame:
    """Validate and select graph relationship-related columns."""
    missing = set(GRAPH_REL_INPUT_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(f"Graph relationship input lacks required columns: {sorted(missing)}")
    return events.select(*GRAPH_REL_INPUT_COLUMNS)


def build_graph_rel_features(events: pl.DataFrame) -> pl.DataFrame:
    """Compute graph relationship features from events strictly earlier than each event's timestamp.

    Rows sharing a TransactionDT are all scored from the same pre-batch state,
    then the complete batch updates state. No label field is accepted or used.
    """
    required = {"TransactionID", "TransactionDT", "card1", "DeviceInfo", "addr1"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Prepared graph relationship events lack required columns: {sorted(missing)}")
    if events.get_column("TransactionID").n_unique() != events.height:
        raise ValueError("TransactionID must be unique when attaching graph relationship features")

    # State variables
    card1_to_devices: Dict[str, set] = {}          # card1 -> set of DeviceInfo values seen with it
    addr1_to_card1s: Dict[str, set] = {}           # addr1 -> set of card1 values seen with it

    feature_rows: list[dict[str, Any]] = []

    ordered = events.sort(["TransactionDT", "TransactionID"])
    for batch in ordered.partition_by("TransactionDT", maintain_order=True):
        batch_rows = list(batch.iter_rows(named=True))
        batch_now = batch_rows[0]["TransactionDT"] if batch_rows else 0

        # Pre-compute features for each row using pre-batch state
        for row in batch_rows:
            # Initialize default feature values
            card1_device_count = 0
            addr1_card1_count = 0

            c1 = row["card1"]
            device = row["DeviceInfo"]
            addr = row["addr1"]

            # Compute tb_card1_device_prior_unique_count
            if c1 is not None and device is not None:
                card1_device_count = len(card1_to_devices.get(c1, set()))
            # else: missing card1 or DeviceInfo -> keep default (0)

            # Compute tb_addr1_card1_prior_unique_count
            if addr is not None and c1 is not None:
                addr1_card1_count = len(addr1_to_card1s.get(addr, set()))
            # else: missing addr1 or card1 -> keep default (0)

            feature_rows.append({
                "TransactionID": row["TransactionID"],
                "tb_card1_device_prior_unique_count": float(card1_device_count),
                "tb_addr1_card1_prior_unique_count": float(addr1_card1_count),
            })

        # Second pass: update state with the entire batch
        for row in batch_rows:
            c1 = row["card1"]
            device = row["DeviceInfo"]
            addr = row["addr1"]

            # Update card1->devices mapping
            if c1 is not None and device is not None:
                if c1 not in card1_to_devices:
                    card1_to_devices[c1] = set()
                card1_to_devices[c1].add(device)

            # Update addr1->card1s mapping
            if addr is not None and c1 is not None:
                if addr not in addr1_to_card1s:
                    addr1_to_card1s[addr] = set()
                addr1_to_card1s[addr].add(c1)

    # Convert to DataFrame, preserving order
    return pl.DataFrame(feature_rows).select("TransactionID", *GRAPH_REL_FEATURES)


def graph_rel_feature_specification() -> dict[str, Any]:
    """Return compact artifact-ready evidence of the graph relationship contract."""
    return {
        "graph_entity_nodes": ["card1", "DeviceInfo", "addr1"],
        "graph_input_columns": list(GRAPH_REL_INPUT_COLUMNS),
        "forbidden_inputs": ["isFraud", "target encoding", "fraud rates", "fraud counts", "future transactions"],
        "features": list(GRAPH_REL_FEATURES),
        "causality": "Features for a TransactionDT batch read state from strictly earlier TransactionDT values; all same-timestamp rows update state only after feature emission.",
    }


if __name__ == "__main__":
    # Simple self-test (can be expanded)
    pass