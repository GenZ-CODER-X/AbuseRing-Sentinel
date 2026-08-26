"""Direct unit tests for strict-prior causal graph state."""

from __future__ import annotations

import sys

import polars as pl

sys.path.insert(0, "scripts")
from causal_graph_features import build_causal_graph_features, prepare_graph_events


def event(transaction_id: int, timestamp: int, addr1: float) -> dict[str, object]:
    return {
        "TransactionID": transaction_id,
        "TransactionDT": timestamp,
        "card1": 100,
        "card2": 1.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 200.0,
        "card6": "credit",
        "DeviceInfo": "device-a",
        "addr1": addr1,
    }


def test_same_timestamp_rows_do_not_see_each_other() -> None:
    features = build_causal_graph_features(prepare_graph_events(pl.DataFrame([event(1, 100, 10.0), event(2, 100, 10.0), event(3, 101, 20.0)]))).sort("TransactionID")
    first, second, third = features.to_dicts()
    assert first["graph_prior_card_transaction_count"] == 0
    assert second["graph_prior_card_transaction_count"] == 0
    assert first["graph_prior_shared_entity_connectivity"] == 0
    assert second["graph_prior_shared_entity_connectivity"] == 0
    assert third["graph_prior_card_transaction_count"] == 2
    assert third["graph_prior_device_transaction_count"] == 2
    assert third["graph_prior_card_distinct_devices"] == 1
    assert third["graph_prior_card_distinct_addresses"] == 1
    assert third["graph_prior_shared_entity_connectivity"] == 1


def test_missing_entities_remain_missing() -> None:
    row = event(4, 102, 30.0)
    row["DeviceInfo"] = None
    result = build_causal_graph_features(prepare_graph_events(pl.DataFrame([row]))).row(0, named=True)
    assert result["graph_prior_device_transaction_count"] is None
    assert result["graph_prior_device_distinct_cards"] is None


if __name__ == "__main__":
    test_same_timestamp_rows_do_not_see_each_other()
    test_missing_entities_remain_missing()
    print("causal graph unit tests passed")
