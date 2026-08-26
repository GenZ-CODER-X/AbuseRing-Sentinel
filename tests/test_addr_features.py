"""Direct unit tests for strict-prior temporal address feature construction."""

from __future__ import annotations

import sys

import polars as pl

sys.path.insert(0, "scripts")
from addr_features import (
    build_addr_features,
    prepare_addr_events,
    ADDR_FEATURES,
)


def event(transaction_id: int, timestamp: int, addr1: float, amount: float) -> dict[str, object]:
    """Create a transaction event with addr1 and amount."""
    return {
        "TransactionID": transaction_id,
        "TransactionDT": timestamp,
        "addr1": addr1,
        "TransactionAmt": amount,
    }


def test_first_seen_address() -> None:
    """First appearance of an address: count 0, mean None, std 0, recency None, zscore 0."""
    df = pl.DataFrame([event(1, 100, 10.0, 50.0)])
    prepared = prepare_addr_events(df)
    result = build_addr_features(prepared).row(0, named=True)
    assert result["tb_addr_prior_count"] == 0.0
    assert result["tb_addr_amt_mean"] is None
    assert result["tb_addr_amt_std"] == 0.0
    assert result["tb_addr_recency"] is None
    assert result["tb_addr_amt_zscore"] == 0.0


def test_repeated_address() -> None:
    """After N prior occurrences of same address, compute correct aggregates."""
    # Prior transactions:
    #   T1: addr A, amount 10
    #   T2: addr A, amount 20
    #   T3: addr A, amount 30
    # Then we evaluate a new transaction with addr A, amount 40.
    df = pl.DataFrame([
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),
        event(3, 300, 10.0, 30.0),
        # The row we are evaluating:
        event(4, 400, 10.0, 40.0),
    ])
    prepared = prepare_addr_events(df)
    features = build_addr_features(prepared).sort("TransactionID")
    f1, f2, f3, f4 = features.to_dicts()
    # Check f4 (the new transaction)
    # Prior count = 3
    assert f4["tb_addr_prior_count"] == 3.0
    # Prior mean = (10+20+30)/3 = 20.0
    assert f4["tb_addr_amt_mean"] == 20.0
    # Prior std: sample variance of [10,20,30] = ((10-20)^2+(20-20)^2+(30-20)^2)/(3-1) = (100+0+100)/2 = 100 => std = 10
    assert f4["tb_addr_amt_std"] == 10.0
    # Recency: most recent prior at time 300 => 400-300 = 100.0
    assert f4["tb_addr_recency"] == 100.0
    # Z-score: (40-20)/10 = 2.0
    assert f4["tb_addr_amt_zscore"] == 2.0

    # Also check earlier rows for correctness
    assert f1["tb_addr_prior_count"] == 0.0
    assert f1["tb_addr_amt_mean"] is None
    assert f1["tb_addr_amt_std"] == 0.0
    assert f1["tb_addr_recency"] is None
    assert f1["tb_addr_amt_zscore"] == 0.0

    assert f2["tb_addr_prior_count"] == 1.0
    assert f2["tb_addr_amt_mean"] == 10.0
    assert f2["tb_addr_amt_std"] == 0.0  # only one prior -> variance 0 -> std 0
    assert f2["tb_addr_recency"] == 200.0 - 100.0  # 100.0
    assert f2["tb_addr_amt_zscore"] == 0.0  # std 0 -> zscore 0

    assert f3["tb_addr_prior_count"] == 2.0
    assert f3["tb_addr_amt_mean"] == 15.0  # (10+20)/2
    # Sample std for [10,20]: variance = ((10-15)^2+(20-15)^2)/(2-1) = (25+25)/1 = 50 => std = sqrt(50)
    expected_std = ((10-15)**2 + (20-15)**2) ** 0.5  # sqrt(50) ≈ 7.0710678
    assert abs(f3["tb_addr_amt_std"] - expected_std) < 1e-9
    assert f3["tb_addr_recency"] == 300.0 - 200.0  # 100.0
    # Z-score: (30-15)/std = 15 / sqrt(50) ≈ 2.12132034
    expected_z = (30.0 - 15.0) / expected_std
    assert abs(f3["tb_addr_amt_zscore"] - expected_z) < 1e-9


def test_same_timestamp_rows_do_not_see_each_other() -> None:
    """Two rows with identical TransactionDT must not see each other's contributions."""
    df = pl.DataFrame([
        event(1, 100, 10.0, 50.0),
        event(2, 100, 10.0, 55.0),
        event(3, 101, 10.0, 60.0),
    ])
    prepared = prepare_addr_events(df)
    features = build_addr_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()

    # First and second rows should see zero prior address transactions
    assert f1["tb_addr_prior_count"] == 0.0
    assert f1["tb_addr_amt_mean"] is None
    assert f1["tb_addr_amt_std"] == 0.0
    assert f1["tb_addr_recency"] is None
    assert f1["tb_addr_amt_zscore"] == 0.0

    assert f2["tb_addr_prior_count"] == 0.0
    assert f2["tb_addr_amt_mean"] is None
    assert f2["tb_addr_amt_std"] == 0.0
    assert f2["tb_addr_recency"] is None
    assert f2["tb_addr_amt_zscore"] == 0.0

    # Third row (time 101) should see both prior rows
    # Prior count = 2
    assert f3["tb_addr_prior_count"] == 2.0
    # Prior mean = (50+55)/2 = 52.5
    assert f3["tb_addr_amt_mean"] == 52.5
    # Prior std: sample variance of [50,55] = ((50-52.5)^2+(55-52.5)^2)/(2-1) = (6.25+6.25)/1 = 12.5 => std = sqrt(12.5)
    expected_std = ((50-52.5)**2 + (55-52.5)**2) ** 0.5
    assert abs(f3["tb_addr_amt_std"] - expected_std) < 1e-9
    # Recency: most recent prior is at time 100 (either row) => 101-100 = 1.0
    assert f3["tb_addr_recency"] == 1.0
    # Z-score: (60 - 52.5) / std = 7.5 / sqrt(12.5)
    expected_z = (60.0 - 52.5) / expected_std
    assert abs(f3["tb_addr_amt_zscore"] - expected_z) < 1e-9


def test_missing_addr_isolation() -> None:
    """Missing addr1 must remain missing and not create shared state."""
    # Row with missing addr1
    row1 = event(1, 100, None, 10.0)
    # Another missing addr1
    row2 = event(2, 200, None, 20.0)
    # Row with addr present
    row3 = event(3, 300, 30.0, 30.0)
    df = pl.DataFrame([row1, row2, row3])
    prepared = prepare_addr_events(df)
    features = build_addr_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()
    # f1 and f2: missing addr -> defaults
    assert f1["tb_addr_prior_count"] == 0.0
    assert f1["tb_addr_amt_mean"] is None
    assert f1["tb_addr_amt_std"] == 0.0
    assert f1["tb_addr_recency"] is None
    assert f1["tb_addr_amt_zscore"] == 0.0
    assert f2["tb_addr_prior_count"] == 0.0
    assert f2["tb_addr_amt_mean"] is None
    assert f2["tb_addr_amt_std"] == 0.0
    assert f2["tb_addr_recency"] is None
    assert f2["tb_addr_amt_zscore"] == 0.0
    # f3: should see zero prior addr transactions because prior missing rows were not counted
    assert f3["tb_addr_prior_count"] == 0.0
    assert f3["tb_addr_amt_mean"] is None
    assert f3["tb_addr_amt_std"] == 0.0
    assert f3["tb_addr_recency"] is None
    assert f3["tb_addr_amt_zscore"] == 0.0


def test_future_row_invariance() -> None:
    """Adding future rows should not affect past feature values."""
    df1 = pl.DataFrame([
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),
    ])
    prepared1 = prepare_addr_events(df1)
    features1 = build_addr_features(prepared1).sort("TransactionID")
    # Add a future row
    df2 = pl.DataFrame([
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),
        event(3, 300, 10.0, 30.0),  # future
    ])
    prepared2 = prepare_addr_events(df2)
    features2 = build_addr_features(prepared2).sort("TransactionID")
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
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),
    ])
    # Add isFraud column
    df = df.with_columns([
        pl.Series("isFraud", [1, 0])
    ])
    prepared = prepare_addr_events(df)
    assert "isFraud" not in prepared.columns
    features = build_addr_features(prepared)
    assert features.shape[0] == 2


def test_deterministic_execution() -> None:
    """Same input yields same output."""
    df = pl.DataFrame([
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),
        event(3, 300, 10.0, 30.0),
    ])
    prepared = prepare_addr_events(df)
    features1 = build_addr_features(prepared)
    features2 = build_addr_features(prepared)
    assert features1.equals(features2)


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
        event(3, val_time, 10.0, 30.0),    # validation event 2 (same addr, same TransactionDT)
    ])
    prepared = prepare_addr_events(df)
    features = build_addr_features(prepared).sort("TransactionID")
    f_train, f_val1, f_val2 = features.to_dicts()
    # Train event should see nothing prior
    assert f_train["tb_addr_prior_count"] == 0.0
    assert f_train["tb_addr_amt_mean"] is None
    assert f_train["tb_addr_amt_std"] == 0.0
    assert f_train["tb_addr_recency"] is None
    assert f_train["tb_addr_amt_zscore"] == 0.0
    # First validation event should see the train event
    assert f_val1["tb_addr_prior_count"] == 1.0
    assert f_val1["tb_addr_amt_mean"] == 10.0
    assert f_val1["tb_addr_amt_std"] == 0.0  # only one prior -> std 0
    assert f_val1["tb_addr_recency"] == val_time - train_time
    assert f_val1["tb_addr_amt_zscore"] == 0.0
    # Second validation event (same timestamp as first) should NOT see the first validation event
    assert f_val2["tb_addr_prior_count"] == 1.0  # only from train
    assert f_val2["tb_addr_amt_mean"] == 10.0
    assert f_val2["tb_addr_amt_std"] == 0.0
    assert f_val2["tb_addr_recency"] == val_time - train_time  # only from train
    assert f_val2["tb_addr_amt_zscore"] == 0.0


def test_zero_one_prior_observation() -> None:
    """Edge cases for zero/one prior observations."""
    # Zero prior: already tested in test_first_seen_address
    # One prior: std should be 0, zscore 0
    df = pl.DataFrame([
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),  # same address
    ])
    prepared = prepare_addr_events(df)
    features = build_addr_features(prepared).sort("TransactionID")
    f1, f2 = features.to_dicts()
    assert f1["tb_addr_prior_count"] == 0.0
    assert f1["tb_addr_amt_mean"] is None
    assert f1["tb_addr_amt_std"] == 0.0
    assert f1["tb_addr_recency"] is None
    assert f1["tb_addr_amt_zscore"] == 0.0
    assert f2["tb_addr_prior_count"] == 1.0
    assert f2["tb_addr_amt_mean"] == 10.0
    assert f2["tb_addr_amt_std"] == 0.0
    assert f2["tb_addr_recency"] == 200.0 - 100.0  # 100.0
    assert f2["tb_addr_amt_zscore"] == 0.0


if __name__ == "__main__":
    test_first_seen_address()
    test_repeated_address()
    test_same_timestamp_rows_do_not_see_each_other()
    test_missing_addr_isolation()
    test_future_row_invariance()
    test_no_label_access()
    test_deterministic_execution()
    test_chronological_train_validation_behavior()
    test_zero_one_prior_observation()
    print("All ADDR unit tests passed")
