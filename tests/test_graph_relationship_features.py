"""Direct unit tests for strict-prior temporal graph relationship feature construction."""

from __future__ import annotations

import sys

import polars as pl

sys.path.insert(0, "scripts")
from graph_relationship_features import (
    build_graph_rel_features,
    prepare_graph_rel_events,
    GRAPH_REL_FEATURES,
)


def event(transaction_id: int, timestamp: int,
          card1: float, device: str, addr: float) -> dict[str, object]:
    """Create a transaction event with graph relationship components."""
    return {
        "TransactionID": transaction_id,
        "TransactionDT": timestamp,
        "card1": card1,
        "DeviceInfo": device,
        "addr1": addr,
    }


def test_first_seen_associations() -> None:
    """First appearance of associations: count 0 for both features."""
    df = pl.DataFrame([event(
        1, 100,
        1.0,  # card1
        "deviceA",  # DeviceInfo
        10.0,   # addr1
    )])
    prepared = prepare_graph_rel_events(df)
    result = build_graph_rel_features(prepared).row(0, named=True)
    assert result["tb_card1_device_prior_unique_count"] == 0.0
    assert result["tb_addr1_card1_prior_unique_count"] == 0.0


def test_card1_device_association_counting() -> None:
    """Test counting distinct DeviceInfo values for each card1."""
    # Prior transactions:
    #   T1: card1=1.0 with DeviceInfo=deviceA
    #   T2: card1=1.0 with DeviceInfo=deviceB (new device for this card)
    #   T3: card1=1.0 with DeviceInfo=deviceA (repeat device)
    #   T4: card1=2.0 with DeviceInfo=deviceA (different card)
    # Then we evaluate a new transaction with card1=1.0, DeviceInfo=deviceC.
    df = pl.DataFrame([
        event(1, 100, 1.0, "deviceA", 10.0),
        event(2, 200, 1.0, "deviceB", 20.0),
        event(3, 300, 1.0, "deviceA", 30.0),
        event(4, 400, 2.0, "deviceA", 40.0),
        # The row we are evaluating:
        event(5, 500, 1.0, "deviceC", 50.0),
    ])
    prepared = prepare_graph_rel_events(df)
    features = build_graph_rel_features(prepared).sort("TransactionID")
    f1, f2, f3, f4, f5 = features.to_dicts()

    # Check f5 (the new transaction)
    # card1=1.0 has seen: deviceA, deviceB, deviceA (distinct: deviceA, deviceB = 2)
    assert f5["tb_card1_device_prior_unique_count"] == 2.0
    # addr1=50.0 is new, so no prior card1 associations
    assert f5["tb_addr1_card1_prior_unique_count"] == 0.0

    # Also check earlier rows for correctness
    # T1: first occurrence of card1=1.0
    assert f1["tb_card1_device_prior_unique_count"] == 0.0
    assert f1["tb_addr1_card1_prior_unique_count"] == 0.0  # addr1=10.0 is new

    # T2: card1=1.0 now has seen deviceA (from T1)
    assert f2["tb_card1_device_prior_unique_count"] == 1.0
    assert f2["tb_addr1_card1_prior_unique_count"] == 0.0  # addr1=20.0 is new

    # T3: card1=1.0 has seen deviceA, deviceB (distinct count still 2)
    assert f3["tb_card1_device_prior_unique_count"] == 2.0
    assert f3["tb_addr1_card1_prior_unique_count"] == 0.0  # addr1=30.0 is new

    # T4: card1=2.0 is new, addr1=40.0 is new
    assert f4["tb_card1_device_prior_unique_count"] == 0.0
    assert f4["tb_addr1_card1_prior_unique_count"] == 0.0


def test_addr1_card1_association_counting() -> None:
    """Test counting distinct card1 values for each addr1."""
    # Prior transactions:
    #   T1: addr1=10.0 with card1=1.0
    #   T2: addr1=10.0 with card1=2.0 (new card for this addr)
    #   T3: addr1=10.0 with card1=1.0 (repeat card)
    #   T4: addr1=20.0 with card1=1.0 (different addr)
    # Then we evaluate a new transaction with addr1=10.0, card1=3.0.
    df = pl.DataFrame([
        event(1, 100, 1.0, "deviceA", 10.0),
        event(2, 200, 2.0, "deviceB", 10.0),
        event(3, 300, 1.0, "deviceC", 10.0),
        event(4, 400, 1.0, "deviceD", 20.0),
        # The row we are evaluating:
        event(5, 500, 3.0, "deviceE", 10.0),
    ])
    prepared = prepare_graph_rel_events(df)
    features = build_graph_rel_features(prepared).sort("TransactionID")
    f1, f2, f3, f4, f5 = features.to_dicts()

    # Check f5 (the new transaction)
    # addr1=10.0 has seen: card1=1.0, card1=2.0, card1=1.0 (distinct: card1=1.0, card1=2.0 = 2)
    assert f5["tb_addr1_card1_prior_unique_count"] == 2.0
    # card1=3.0 is new, so no prior DeviceInfo associations
    assert f5["tb_card1_device_prior_unique_count"] == 0.0

    # Also check earlier rows for correctness
    # T1: first occurrence of addr1=10.0
    assert f1["tb_addr1_card1_prior_unique_count"] == 0.0
    assert f1["tb_card1_device_prior_unique_count"] == 0.0  # card1=1.0 is new

    # T2: addr1=10.0 now has seen card1=1.0 (from T1)
    assert f2["tb_addr1_card1_prior_unique_count"] == 1.0
    assert f2["tb_card1_device_prior_unique_count"] == 0.0  # card1=2.0 is new

    # T3: addr1=10.0 has seen card1=1.0, card1=2.0 (distinct count still 2)
    assert f3["tb_addr1_card1_prior_unique_count"] == 2.0
    assert f3["tb_card1_device_prior_unique_count"] == 1.0  # card1=1.0 has seen deviceB (from T2)

    # T4: addr1=20.0 is new, card1=1.0 has seen deviceA, deviceC (from T1 and T3)
    assert f4["tb_addr1_card1_prior_unique_count"] == 0.0
    assert f4["tb_card1_device_prior_unique_count"] == 2.0


def test_repeated_association_does_not_increment() -> None:
    """Repeated association with same device/card should not increase unique count."""
    # Multiple transactions with same card1-device pair
    df = pl.DataFrame([
        event(1, 100, 1.0, "deviceA", 10.0),
        event(2, 200, 1.0, "deviceA", 20.0),  # same pair
        event(3, 300, 1.0, "deviceA", 30.0),  # same pair
        # The row we are evaluating:
        event(4, 400, 1.0, "deviceB", 40.0),  # new device
    ])
    prepared = prepare_graph_rel_events(df)
    features = build_graph_rel_features(prepared).sort("TransactionID")
    f1, f2, f3, f4 = features.to_dicts()

    # T1: first occurrence
    assert f1["tb_card1_device_prior_unique_count"] == 0.0
    assert f1["tb_addr1_card1_prior_unique_count"] == 0.0

    # T2: same card1-device pair as T1
    assert f2["tb_card1_device_prior_unique_count"] == 1.0  # deviceA
    assert f2["tb_addr1_card1_prior_unique_count"] == 0.0  # addr1=20.0 is new

    # T3: same card1-device pair as T1 and T2
    assert f3["tb_card1_device_prior_unique_count"] == 1.0  # still just deviceA
    assert f3["tb_addr1_card1_prior_unique_count"] == 0.0  # addr1=30.0 is new

    # T4: new device for card1=1.0
    assert f4["tb_card1_device_prior_unique_count"] == 1.0  # deviceA only
    assert f4["tb_addr1_card1_prior_unique_count"] == 0.0  # addr1=40.0 is new


def test_same_timestamp_rows_do_not_see_each_other() -> None:
    """Two rows with identical TransactionDT must not see each other's contributions."""
    df = pl.DataFrame([
        event(1, 100, 1.0, "deviceA", 10.0),
        event(2, 100, 1.0, "deviceB", 20.0),  # same timestamp, different device
        event(3, 101, 1.0, "deviceC", 30.0),
    ])
    prepared = prepare_graph_rel_events(df)
    features = build_graph_rel_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()

    # First and second rows should see zero prior device associations for card1=1.0
    assert f1["tb_card1_device_prior_unique_count"] == 0.0
    assert f1["tb_addr1_card1_prior_unique_count"] == 0.0  # addr1=10.0 is new

    assert f2["tb_card1_device_prior_unique_count"] == 0.0  # should NOT see T1's device
    assert f2["tb_addr1_card1_prior_unique_count"] == 0.0  # addr1=20.0 is new

    # Third row (time 101) should see both prior rows' contributions
    # card1=1.0 has seen deviceA and deviceB from times 100 (both rows)
    assert f3["tb_card1_device_prior_unique_count"] == 2.0
    # addr1=30.0 is new, so no prior card1 associations
    assert f3["tb_addr1_card1_prior_unique_count"] == 0.0


def test_missing_field_handling() -> None:
    """Missing any component should not create associations and should remain zero."""
    # Row with missing card1
    row1 = event(1, 100, None, "deviceA", 10.0)
    # Row with missing DeviceInfo
    row2 = event(2, 200, 1.0, None, 20.0)
    # Row with missing addr1
    row3 = event(3, 300, 1.0, "deviceB", None)
    # Row with all present
    row4 = event(4, 400, 2.0, "deviceC", 30.0)
    df = pl.DataFrame([row1, row2, row3, row4])
    prepared = prepare_graph_rel_events(df)
    features = build_graph_rel_features(prepared).sort("TransactionID")
    f1, f2, f3, f4 = features.to_dicts()

    # f1: missing card1 -> no associations possible
    assert f1["tb_card1_device_prior_unique_count"] == 0.0
    assert f1["tb_addr1_card1_prior_unique_count"] == 0.0  # addr1=10.0 is new

    # f2: missing DeviceInfo -> no associations possible
    assert f2["tb_card1_device_prior_unique_count"] == 0.0
    assert f2["tb_addr1_card1_prior_unique_count"] == 0.0  # addr1=20.0 is new

    # f3: missing addr1 -> no associations possible
    assert f3["tb_card1_device_prior_unique_count"] == 0.0  # card1=1.0 has no DeviceInfo to associate
    assert f3["tb_addr1_card1_prior_unique_count"] == 0.0

    # f4: all present, but prior missing rows didn't create associations
    assert f4["tb_card1_device_prior_unique_count"] == 0.0
    assert f4["tb_addr1_card1_prior_unique_count"] == 0.0


def test_future_row_invariance() -> None:
    """Adding future rows should not affect past feature values."""
    df1 = pl.DataFrame([
        event(1, 100, 1.0, "deviceA", 10.0),
        event(2, 200, 1.0, "deviceB", 20.0),
    ])
    prepared1 = prepare_graph_rel_events(df1)
    features1 = build_graph_rel_features(prepared1).sort("TransactionID")
    # Add a future row
    df2 = pl.DataFrame([
        event(1, 100, 1.0, "deviceA", 10.0),
        event(2, 200, 1.0, "deviceB", 20.0),
        event(3, 300, 1.0, "deviceC", 30.0),  # future
    ])
    prepared2 = prepare_graph_rel_events(df2)
    features2 = build_graph_rel_features(prepared2).sort("TransactionID")
    # Features for rows 1 and 2 should be identical
    f1_orig = features1.row(0, named=True)
    f2_orig = features1.row(1, named=True)
    f1_new = features2.row(0, named=True)
    f2_new = features2.row(1, named=True)
    assert f1_orig == f1_new
    assert f2_orig == f2_new


def test_no_label_access() -> None:
    """Builder should not use isFraud column if present."""
    df = pl.DataFrame([
        event(1, 100, 1.0, "deviceA", 10.0),
        event(2, 200, 1.0, "deviceB", 20.0),
    ])
    # Add isFraud column
    df = df.with_columns([
        pl.Series("isFraud", [1, 0])
    ])
    prepared = prepare_graph_rel_events(df)
    assert "isFraud" not in prepared.columns
    features = build_graph_rel_features(prepared)
    assert features.shape[0] == 2


def test_deterministic_execution() -> None:
    """Same input yields same output."""
    df = pl.DataFrame([
        event(1, 100, 1.0, "deviceA", 10.0),
        event(2, 200, 1.0, "deviceB", 20.0),
        event(3, 300, 1.0, "deviceC", 30.0),
    ])
    prepared = prepare_graph_rel_events(df)
    features1 = build_graph_rel_features(prepared)
    features2 = build_graph_rel_features(prepared)
    assert features1.equals(features2)


def test_chronological_train_validation_behavior() -> None:
    """Using the actual boundaries from validation_boundaries.py, verify train->validation flow."""
    from validation_boundaries import PARTITION_BOUNDARIES
    train_boundary = next(b for b in PARTITION_BOUNDARIES if b.name == "train")
    val_boundary = next(b for b in PARTITION_BOUNDARIES if b.name == "validation")
    train_time = train_boundary.end  # last train timestamp
    val_time = val_boundary.start    # first validation timestamp
    df = pl.DataFrame([
        event(1, train_time, 1.0, "deviceA", 10.0),  # train event
        event(2, val_time, 1.0, "deviceB", 20.0),    # validation event 1 (same card1, new device)
        event(3, val_time, 1.0, "deviceB", 30.0),    # validation event 2 (same card1, same device as above)
    ])
    prepared = prepare_graph_rel_events(df)
    features = build_graph_rel_features(prepared).sort("TransactionID")
    f_train, f_val1, f_val2 = features.to_dicts()

    # Train event should see nothing prior
    assert f_train["tb_card1_device_prior_unique_count"] == 0.0
    assert f_train["tb_addr1_card1_prior_unique_count"] == 0.0

    # First validation event should see the train event
    # card1=1.0 has seen deviceA from train
    assert f_val1["tb_card1_device_prior_unique_count"] == 1.0
    # addr1=20.0 is new, so no prior card1 associations
    assert f_val1["tb_addr1_card1_prior_unique_count"] == 0.0

    # Second validation event (same timestamp as first) should NOT see the first validation event
    assert f_val2["tb_card1_device_prior_unique_count"] == 1.0  # only from train (deviceA)
    assert f_val2["tb_addr1_card1_prior_unique_count"] == 0.0  # addr1=30.0 is new


def test_zero_one_prior_association() -> None:
    """Edge cases for zero/one prior associations."""
    # Zero prior: already tested in test_first_seen_associations
    # One prior: std should be 0 (but we're dealing with counts, so just test the logic)
    df = pl.DataFrame([
        event(1, 100, 1.0, "deviceA", 10.0),
        event(2, 200, 1.0, "deviceB", 20.0),  # new device for card1=1.0
    ])
    prepared = prepare_graph_rel_events(df)
    features = build_graph_rel_features(prepared).sort("TransactionID")
    f1, f2 = features.to_dicts()
    assert f1["tb_card1_device_prior_unique_count"] == 0.0
    assert f1["tb_addr1_card1_prior_unique_count"] == 0.0
    assert f2["tb_card1_device_prior_unique_count"] == 1.0  # deviceA
    assert f2["tb_addr1_card1_prior_unique_count"] == 0.0  # addr1=20.0 is new


if __name__ == "__main__":
    test_first_seen_associations()
    test_card1_device_association_counting()
    test_addr1_card1_association_counting()
    test_repeated_association_does_not_increment()
    test_same_timestamp_rows_do_not_see_each_other()
    test_missing_field_handling()
    test_future_row_invariance()
    test_no_label_access()
    test_deterministic_execution()
    test_chronological_train_validation_behavior()
    test_zero_one_prior_association()
    print("All graph relationship unit tests passed")