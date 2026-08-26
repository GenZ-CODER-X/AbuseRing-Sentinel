"""Direct unit tests for strict-prior temporal behavioral graph feature construction for E4."""

from __future__ import annotations

import sys

import polars as pl

sys.path.insert(0, "scripts")
from e4_features import (
    build_e4_features,
    prepare_e4_events,
    E4_FEATURES,
    CARD_COMPONENTS,
)


def event(transaction_id: int, timestamp: int, addr1: float, amount: float,
          card1: int = 100, card2: float = 1.0, card3: float = 150.0,
          card4: str = "visa", card5: float = 200.0, card6: str = "credit",
          device: str = "device-a") -> dict[str, object]:
    """Create a transaction event with default values matching the card entity."""
    return {
        "TransactionID": transaction_id,
        "TransactionDT": timestamp,
        "card1": card1,
        "card2": card2,
        "card3": card3,
        "card4": card4,
        "card5": card5,
        "card6": card6,
        "DeviceInfo": device,
        "addr1": addr1,
        "TransactionAmt": amount,  # amount not used in E4 features but kept for compatibility
    }


def test_first_seen_card_device() -> None:
    """First appearance of a card-device pair: count 0, recency None, strength 0."""
    df = pl.DataFrame([event(1, 100, 10.0, 50.0)])
    prepared = prepare_e4_events(df)
    result = build_e4_features(prepared).row(0, named=True)
    assert result["tb_graph_card_device_count"] == 0.0
    assert result["tb_graph_card_device_recency"] is None
    assert result["tb_graph_card_device_strength"] == 0.0


def test_repeated_card_device() -> None:
    """After N prior occurrences of same pair, count = N, recency = time diff, strength = N / prior_card_count."""
    # Prior transactions:
    #   T1: card C1, device D1
    #   T2: card C1, device D2 (different device)
    #   T3: card C1, device D1 (second time with D1)
    # Then we evaluate a new transaction with card C1, device D1.
    # Prior pair count for (C1,D1) = 2
    # Prior card count for C1 = 3
    # Expected count = 2
    # Expected recency: now - timestamp of T3
    # Expected strength = 2/3
    df = pl.DataFrame([
        event(1, 100, 10.0, 10.0, card1=111, card2=111, card3=111, card4="c1", card5=111, card6=111, device="dev1"),
        event(2, 200, 10.0, 20.0, card1=111, card2=111, card3=111, card4="c1", card5=111, card6=111, device="dev2"),
        event(3, 300, 10.0, 15.0, card1=111, card2=111, card3=111, card4="c1", card5=111, card6=111, device="dev1"),
        # The row we are evaluating:
        event(4, 400, 10.0, 30.0, card1=111, card2=111, card3=111, card4="c1", card5=111, card6=111, device="dev1"),
    ])
    prepared = prepare_e4_events(df)
    features = build_e4_features(prepared).sort("TransactionID")
    f1, f2, f3, f4 = features.to_dicts()
    # Check f4
    assert f4["tb_graph_card_device_count"] == 2.0
    assert f4["tb_graph_card_device_recency"] == 400.0 - 300.0  # 100.0
    assert abs(f4["tb_graph_card_device_strength"] - (2.0 / 3.0)) < 1e-9


def test_same_timestamp_rows_do_not_see_each_other() -> None:
    """Two rows with identical TransactionDT must not see each other's contributions."""
    df = pl.DataFrame([
        event(1, 100, 10.0, 50.0),
        event(2, 100, 10.0, 55.0),
        event(3, 101, 10.0, 60.0),
    ])
    prepared = prepare_e4_events(df)
    features = build_e4_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()

    # First and second rows should see zero prior card-device transactions
    assert f1["tb_graph_card_device_count"] == 0.0
    assert f1["tb_graph_card_device_recency"] is None
    assert f1["tb_graph_card_device_strength"] == 0.0

    assert f2["tb_graph_card_device_count"] == 0.0
    assert f2["tb_graph_card_device_recency"] is None
    assert f2["tb_graph_card_device_strength"] == 0.0

    # Third row (time 101) should see both prior rows
    # For card-device pair: assuming same card and device across all rows (default)
    # Prior pair count = 2
    assert f3["tb_graph_card_device_count"] == 2.0
    # Recency: most recent prior is at time 100 (either row) -> 101-100 = 1.0
    assert f3["tb_graph_card_device_recency"] == 1.0
    # Prior card count = 2 (two prior transactions with same card)
    assert abs(f3["tb_graph_card_device_strength"] - (2.0 / 2.0)) < 1e-9  # strength = 1.0


def test_missing_entities_remain_missing() -> None:
    """If card or device is missing, appropriate features should be 0/None/0."""
    row = event(4, 102, 30.0, 70.0)
    row["DeviceInfo"] = None  # missing device
    df = pl.DataFrame([row])
    prepared = prepare_e4_events(df)
    result = build_e4_features(prepared).row(0, named=True)

    # With missing device, count=0, recency=None, strength=0
    assert result["tb_graph_card_device_count"] == 0.0
    assert result["tb_graph_card_device_recency"] is None
    assert result["tb_graph_card_device_strength"] == 0.0

    # Missing card (all card components None)
    row2 = event(5, 103, 30.0, 80.0)
    for c in CARD_COMPONENTS:
        row2[c] = None
    df2 = pl.DataFrame([row2])
    prepared2 = prepare_e4_events(df2)
    result2 = build_e4_features(prepared2).row(0, named=True)
    assert result2["tb_graph_card_device_count"] == 0.0
    assert result2["tb_graph_card_device_recency"] is None
    assert result2["tb_graph_card_device_strength"] == 0.0


def test_zero_one_prior_card_observation() -> None:
    """Edge cases for zero/one prior card observations."""
    # Zero prior card: already covered in test_first_seen_card_device (strength 0)
    # One prior card: strength should be 0 if device differs, or 1 if same device? Actually strength = prior_pair_count / prior_card_count.
    # If we have one prior transaction with same device, then prior_pair_count=1, prior_card_count=1 => strength=1.
    df = pl.DataFrame([
        event(1, 100, 10.0, 10.0, device="devA"),
        event(2, 200, 10.0, 20.0, device="devA"),  # same device
    ])
    prepared = prepare_e4_events(df)
    features = build_e4_features(prepared).sort("TransactionID")
    f1, f2 = features.to_dicts()
    assert f1["tb_graph_card_device_count"] == 0.0
    assert f1["tb_graph_card_device_recency"] is None
    assert f1["tb_graph_card_device_strength"] == 0.0
    assert f2["tb_graph_card_device_count"] == 1.0
    assert f2["tb_graph_card_device_recency"] == 200.0 - 100.0  # 100.0
    assert f2["tb_graph_card_device_strength"] == 1.0


def test_deterministic_execution() -> None:
    """Same input yields same output."""
    df = pl.DataFrame([
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),
        event(3, 300, 10.0, 30.0),
    ])
    prepared = prepare_e4_events(df)
    features1 = build_e4_features(prepared)
    features2 = build_e4_features(prepared)
    assert features1.equals(features2)


def test_no_label_access() -> None:
    """Builder should not use isFraud column if present."""
    df = pl.DataFrame([
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),
    ])
    # Add isFraud column
    df = df.with_columns([
        pl.Series("isFraud", [1, 0])
    ])
    prepared = prepare_e4_events(df)
    assert "isFraud" not in prepared.columns
    features = build_e4_features(prepared)
    assert features.shape[0] == 2


def test_chronological_train_validation_behavior() -> None:
    """Using the actual boundaries from validation_boundaries.py, verify train->validation flow."""
    from validation_boundaries import PARTITION_BOUNDARIES
    train_boundary = next(b for b in PARTITION_BOUNDARIES if b.name == "train")
    val_boundary = next(b for b in PARTITION_BOUNDARIES if b.name == "validation")
    train_time = train_boundary.end  # last train timestamp
    val_time = val_boundary.start    # first validation timestamp
    df = pl.DataFrame([
        event(1, train_time, 10.0, 10.0),  # train event
        event(2, val_time, 10.0, 20.0),    # validation event 1
        event(3, val_time, 20.0, 30.0),    # validation event 2 (same TransactionDT)
    ])
    prepared = prepare_e4_events(df)
    features = build_e4_features(prepared).sort("TransactionID")
    f_train, f_val1, f_val2 = features.to_dicts()
    # Train event should see nothing prior
    assert f_train["tb_graph_card_device_count"] == 0.0
    assert f_train["tb_graph_card_device_recency"] is None
    assert f_train["tb_graph_card_device_strength"] == 0.0
    # First validation event should see the train event
    assert f_val1["tb_graph_card_device_count"] == 1.0
    assert f_val1["tb_graph_card_device_recency"] == val_time - train_time
    assert f_val1["tb_graph_card_device_strength"] == 1.0 / 1.0  # prior card count =1
    # Second validation event (same timestamp as first) should NOT see the first validation event
    assert f_val2["tb_graph_card_device_count"] == 1.0  # only from train
    assert f_val2["tb_graph_card_device_recency"] == val_time - train_time  # only from train
    assert f_val2["tb_graph_card_device_strength"] == 1.0 / 1.0


def test_missing_card_isolation() -> None:
    """Two rows with incomplete card (missing some card components) must not share state."""
    # Row 1: missing card6
    row1 = event(1, 100, 10.0, 10.0, card6=None)
    # Row 2: missing card5 (different missing component)
    row2 = event(2, 200, 10.0, 20.0, card5=None)
    # Row 3: complete card (should see prior complete cards only)
    row3 = event(3, 300, 10.0, 30.0)
    df = pl.DataFrame([row1, row2, row3])
    prepared = prepare_e4_events(df)
    features = build_e4_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()
    # f1 and f2 should have card-related features as 0/None/0 because card is None
    assert f1["tb_graph_card_device_count"] == 0.0
    assert f1["tb_graph_card_device_recency"] is None
    assert f1["tb_graph_card_device_strength"] == 0.0
    assert f2["tb_graph_card_device_count"] == 0.0
    assert f2["tb_graph_card_device_recency"] is None
    assert f2["tb_graph_card_device_strength"] == 0.0
    # f3 should see zero prior card-device transactions because prior cards were None and thus not counted
    assert f3["tb_graph_card_device_count"] == 0.0
    assert f3["tb_graph_card_device_recency"] is None
    assert f3["tb_graph_card_device_strength"] == 0.0


def test_missing_device_isolation() -> None:
    """Two rows with missing device must not affect each other's count/recency/strength."""
    # Row 1: device None
    row1 = event(1, 100, 10.0, 10.0, device=None)
    # Row 2: device None again
    row2 = event(2, 200, 10.0, 20.0, device=None)
    # Row 3: device present
    row3 = event(3, 300, 10.0, 30.0, device="devX")
    df = pl.DataFrame([row1, row2, row3])
    prepared = prepare_e4_events(df)
    features = build_e4_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()
    # f1: missing device -> count 0, recency None, strength 0
    assert f1["tb_graph_card_device_count"] == 0.0
    assert f1["tb_graph_card_device_recency"] is None
    assert f1["tb_graph_card_device_strength"] == 0.0
    # f2: still missing device -> count 0 (should not have seen previous None device as a pair)
    assert f2["tb_graph_card_device_count"] == 0.0
    assert f2["tb_graph_card_device_recency"] is None
    assert f2["tb_graph_card_device_strength"] == 0.0
    # f3: device present, should see zero prior device transactions because prior device=None rows were not counted
    assert f3["tb_graph_card_device_count"] == 0.0
    assert f3["tb_graph_card_device_recency"] is None
    assert f3["tb_graph_card_device_strength"] == 0.0


def test_future_row_invariance() -> None:
    """Adding future rows should not affect past feature values."""
    df1 = pl.DataFrame([
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),
    ])
    prepared1 = prepare_e4_events(df1)
    features1 = build_e4_features(prepared1).sort("TransactionID")
    # Add a future row
    df2 = pl.DataFrame([
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),
        event(3, 300, 10.0, 30.0),  # future
    ])
    prepared2 = prepare_e4_events(df2)
    features2 = build_e4_features(prepared2).sort("TransactionID")
    # Features for rows 1 and 2 should be identical
    f1_orig = features1.row(0, named=True)
    f2_orig = features1.row(1, named=True)
    f1_new = features2.row(0, named=True)
    f2_new = features2.row(1, named=True)
    assert f1_orig == f1_new
    assert f2_orig == f2_new


if __name__ == "__main__":
    test_first_seen_card_device()
    test_repeated_card_device()
    test_same_timestamp_rows_do_not_see_each_other()
    test_missing_entities_remain_missing()
    test_zero_one_prior_card_observation()
    test_deterministic_execution()
    test_no_label_access()
    test_chronological_train_validation_behavior()
    test_missing_card_isolation()
    test_missing_device_isolation()
    test_future_row_invariance()
    print("All E4 unit tests passed")
