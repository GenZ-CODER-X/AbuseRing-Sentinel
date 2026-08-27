"""Direct unit tests for strict-prior temporal entity feature construction."""

from __future__ import annotations

import sys

import polars as pl

sys.path.insert(0, "scripts")
from entity_features import (
    build_entity_features,
    prepare_entity_events,
    ENTITY_FEATURES,
)


def event(transaction_id: int, timestamp: int,
          card1: float, card2: float, card3: float, card4: float, card5: float, card6: float,
          addr: float, amount: float, product: str) -> dict[str, object]:
    """Create a transaction event with entity components."""
    return {
        "TransactionID": transaction_id,
        "TransactionDT": timestamp,
        "card1": card1,
        "card2": card2,
        "card3": card3,
        "card4": card4,
        "card5": card5,
        "card6": card6,
        "addr1": addr,
        "TransactionAmt": amount,
        "ProductCD": product,
    }


def test_first_seen_entity() -> None:
    """First appearance of entity: count=0, first_seen_age=0, recency=null, mean=null, std=0, zscore=0, unique_products=0."""
    df = pl.DataFrame([event(
        1, 100,
        1.0,  # card1
        2.0,  # card2
        3.0,  # card3
        4.0,  # card4
        5.0,  # card5
        6.0,  # card6
        10.0, # addr1
        50.0, # TransactionAmt
        "H"   # ProductCD
    )])
    prepared = prepare_entity_events(df)
    result = build_entity_features(prepared).row(0, named=True)
    assert result["tb_entity_prior_count"] == 0.0
    assert result["tb_entity_first_seen_age"] == 0.0  # First seen, age = 0
    assert result["tb_entity_recency"] is None
    assert result["tb_entity_amt_mean"] is None
    assert result["tb_entity_amt_std"] == 0.0
    assert result["tb_entity_amt_zscore"] == 0.0
    assert result["tb_entity_unique_product_count"] == 0.0


def test_repeated_entity() -> None:
    """Test repeated entity with correct aggregates."""
    # Prior transactions:
    #   T1: amount=50.0, product=H
    #   T2: amount=100.0, product=J (new product)
    # Then we evaluate a new transaction with amount=75.0, product=H
    df = pl.DataFrame([
        event(1, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        event(2, 200, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 100.0, "J"),
        # The row we are evaluating:
        event(3, 300, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 75.0, "H"),
    ])
    prepared = prepare_entity_events(df)
    features = build_entity_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()

    # T1: first occurrence
    assert f1["tb_entity_prior_count"] == 0.0
    assert f1["tb_entity_first_seen_age"] == 0.0
    assert f1["tb_entity_recency"] is None
    assert f1["tb_entity_amt_mean"] is None
    assert f1["tb_entity_amt_std"] == 0.0
    assert f1["tb_entity_amt_zscore"] == 0.0
    assert f1["tb_entity_unique_product_count"] == 0.0

    # T2: second transaction
    assert f2["tb_entity_prior_count"] == 1.0  # One prior
    assert f2["tb_entity_first_seen_age"] == 100.0  # Seen 100 seconds ago
    assert f2["tb_entity_recency"] == 100.0  # 100 seconds since prior
    assert f2["tb_entity_amt_mean"] == 50.0  # Mean of prior amount
    assert f2["tb_entity_amt_std"] == 0.0  # Std of single prior is 0
    assert f2["tb_entity_amt_zscore"] == 0.0  # Std=0 so zscore=0
    assert f2["tb_entity_unique_product_count"] == 1.0  # One unique product (H) seen prior

    # T3: third transaction
    assert f3["tb_entity_prior_count"] == 2.0  # Two priors
    assert f3["tb_entity_first_seen_age"] == 200.0  # First seen 200 seconds ago
    assert f3["tb_entity_recency"] == 100.0  # 100 seconds since prior (T2)
    assert f3["tb_entity_amt_mean"] == 75.0  # Mean of [50.0, 100.0]
    # Std of [50.0, 100.0] = sqrt(((50-75)^2 + (100-75)^2)/(2-1)) = sqrt(1250) ≈ 35.355
    assert abs(f3["tb_entity_amt_std"] - 35.355) < 0.001
    # Z-score = (75.0 - 75.0) / 35.355 = 0.0
    assert f3["tb_entity_amt_zscore"] == 0.0
    assert f3["tb_entity_unique_product_count"] == 2.0  # Two unique products (H, J) seen prior


def test_same_timestamp_rows_do_not_see_each_other() -> None:
    """Two rows with identical TransactionDT must not see each other's contributions."""
    df = pl.DataFrame([
        event(1, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        event(2, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 100.0, "J"),  # same timestamp, different amount/product
        event(3, 200, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 75.0, "H"),
    ])
    prepared = prepare_entity_events(df)
    features = build_entity_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()

    # First and second rows should see zero prior transactions for this entity
    assert f1["tb_entity_prior_count"] == 0.0
    assert f1["tb_entity_first_seen_age"] == 0.0
    assert f1["tb_entity_recency"] is None
    assert f1["tb_entity_amt_mean"] is None
    assert f1["tb_entity_amt_std"] == 0.0
    assert f1["tb_entity_amt_zscore"] == 0.0
    assert f1["tb_entity_unique_product_count"] == 0.0

    assert f2["tb_entity_prior_count"] == 0.0  # Should NOT see T1's transaction
    assert f2["tb_entity_first_seen_age"] == 0.0
    assert f2["tb_entity_recency"] is None
    assert f2["tb_entity_amt_mean"] is None
    assert f2["tb_entity_amt_std"] == 0.0
    assert f2["tb_entity_amt_zscore"] == 0.0
    assert f2["tb_entity_unique_product_count"] == 0.0

    # Third row (time 200) should see both prior rows' contributions
    assert f3["tb_entity_prior_count"] == 2.0
    assert f3["tb_entity_first_seen_age"] == 100.0  # First seen at time 100
    assert f3["tb_entity_recency"] == 100.0  # 100 seconds since most recent prior
    assert f3["tb_entity_amt_mean"] == 75.0  # Mean of [50.0, 100.0]
    assert abs(f3["tb_entity_amt_std"] - 35.355) < 0.001  # Std of [50.0, 100.0] ≈ 35.355
    assert f3["tb_entity_amt_zscore"] == 0.0  # (75.0 - 75.0) / 35.355 = 0.0
    assert f3["tb_entity_unique_product_count"] == 2.0  # Two unique products (H, J)


def test_missing_field_handling() -> None:
    """Missing any component should not create entities and should remain at defaults."""
    # Row with missing card1
    row1 = event(1, 100, None, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H")
    # Row with missing addr1
    row2 = event(2, 200, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, None, 100.0, "J")
    # Row with all present
    row3 = event(3, 300, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 75.0, "H")
    df = pl.DataFrame([row1, row2, row3])
    prepared = prepare_entity_events(df)
    features = build_entity_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()

    # f1: missing card1 -> no entity possible
    assert f1["tb_entity_prior_count"] == 0.0
    assert f1["tb_entity_first_seen_age"] == 0.0
    assert f1["tb_entity_recency"] is None
    assert f1["tb_entity_amt_mean"] is None
    assert f1["tb_entity_amt_std"] == 0.0
    assert f1["tb_entity_amt_zscore"] == 0.0
    assert f1["tb_entity_unique_product_count"] == 0.0

    # f2: missing addr1 -> no entity possible
    assert f2["tb_entity_prior_count"] == 0.0
    assert f2["tb_entity_first_seen_age"] == 0.0
    assert f2["tb_entity_recency"] is None
    assert f2["tb_entity_amt_mean"] is None
    assert f2["tb_entity_amt_std"] == 0.0
    assert f2["tb_entity_amt_zscore"] == 0.0
    assert f2["tb_entity_unique_product_count"] == 0.0

    # f3: all present, but prior missing rows didn't create entities
    assert f3["tb_entity_prior_count"] == 0.0
    assert f3["tb_entity_first_seen_age"] == 0.0
    assert f3["tb_entity_recency"] is None
    assert f3["tb_entity_amt_mean"] is None
    assert f3["tb_entity_amt_std"] == 0.0
    assert f3["tb_entity_amt_zscore"] == 0.0
    assert f3["tb_entity_unique_product_count"] == 0.0


def test_future_row_invariance() -> None:
    """Adding future rows should not affect past feature values."""
    df1 = pl.DataFrame([
        event(1, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        event(2, 200, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 100.0, "J"),
    ])
    prepared1 = prepare_entity_events(df1)
    features1 = build_entity_features(prepared1).sort("TransactionID")
    # Add a future row
    df2 = pl.DataFrame([
        event(1, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        event(2, 200, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 100.0, "J"),
        event(3, 300, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 75.0, "H"),  # future
    ])
    prepared2 = prepare_entity_events(df2)
    features2 = build_entity_features(prepared2).sort("TransactionID")
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
        event(1, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        event(2, 200, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 100.0, "J"),
    ])
    # Add isFraud column
    df = df.with_columns([
        pl.Series("isFraud", [1, 0])
    ])
    prepared = prepare_entity_events(df)
    assert "isFraud" not in prepared.columns
    features = build_entity_features(prepared)
    assert features.shape[0] == 2


def test_deterministic_execution() -> None:
    """Same input yields same output."""
    df = pl.DataFrame([
        event(1, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        event(2, 200, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 100.0, "J"),
        event(3, 300, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 75.0, "H"),
    ])
    prepared = prepare_entity_events(df)
    features1 = build_entity_features(prepared)
    features2 = build_entity_features(prepared)
    assert features1.equals(features2)


def test_chronological_train_validation_behavior() -> None:
    """Using the actual boundaries from validation_boundaries.py, verify train->validation flow."""
    from validation_boundaries import PARTITION_BOUNDARIES
    train_boundary = next(b for b in PARTITION_BOUNDARIES if b.name == "train")
    val_boundary = next(b for b in PARTITION_BOUNDARIES if b.name == "validation")
    train_time = train_boundary.end  # last train timestamp
    val_time = val_boundary.start    # first validation timestamp
    df = pl.DataFrame([
        event(1, train_time, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),  # train event
        event(2, val_time, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 100.0, "J"),    # validation event 1 (same entity, new product)
        event(3, val_time, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 75.0, "H"),    # validation event 2 (same entity, same product as T1)
    ])
    prepared = prepare_entity_events(df)
    features = build_entity_features(prepared).sort("TransactionID")
    f_train, f_val1, f_val2 = features.to_dicts()

    # Train event should see nothing prior
    assert f_train["tb_entity_prior_count"] == 0.0
    assert f_train["tb_entity_first_seen_age"] == 0.0
    assert f_train["tb_entity_recency"] is None
    assert f_train["tb_entity_amt_mean"] is None
    assert f_train["tb_entity_amt_std"] == 0.0
    assert f_train["tb_entity_amt_zscore"] == 0.0
    assert f_train["tb_entity_unique_product_count"] == 0.0

    # First validation event should see the train event
    assert f_val1["tb_entity_prior_count"] == 1.0  # One prior transaction
    assert f_val1["tb_entity_first_seen_age"] == (val_time - train_time)  # Age since first seen
    assert f_val1["tb_entity_recency"] == (val_time - train_time)  # Time since prior
    assert f_val1["tb_entity_amt_mean"] == 50.0  # Mean of prior amount
    assert f_val1["tb_entity_amt_std"] == 0.0  # Std of single prior is 0
    assert f_val1["tb_entity_amt_zscore"] == 0.0  # Std=0 so zscore=0
    assert f_val1["tb_entity_unique_product_count"] == 1.0  # One unique product (H) seen prior

    # Second validation event (same timestamp as first) should NOT see the first validation event
    assert f_val2["tb_entity_prior_count"] == 1.0  # Only from train (not from same-timestamp validation)
    assert f_val2["tb_entity_first_seen_age"] == (val_time - train_time)  # Age since first seen
    assert f_val2["tb_entity_recency"] == (val_time - train_time)  # Time since prior (still only train)
    assert f_val2["tb_entity_amt_mean"] == 50.0  # Mean of prior amount (still only train)
    assert f_val2["tb_entity_amt_std"] == 0.0  # Std of single prior
    assert f_val2["tb_entity_amt_zscore"] == 0.0  # Std=0 so zscore=0
    assert f_val2["tb_entity_unique_product_count"] == 1.0  # One unique product (H) from train only


def test_zero_one_prior_association() -> None:
    """Edge cases for zero/one prior associations."""
    # Zero prior: already tested in test_first_seen_entity
    # One prior: std should be 0 (but we're dealing with counts, so just test the logic)
    df = pl.DataFrame([
        event(1, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        event(2, 200, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 100.0, "J"),  # new product for entity
    ])
    prepared = prepare_entity_events(df)
    features = build_entity_features(prepared).sort("TransactionID")
    f1, f2 = features.to_dicts()
    assert f1["tb_entity_prior_count"] == 0.0
    assert f1["tb_entity_first_seen_age"] == 0.0
    assert f1["tb_entity_recency"] is None
    assert f1["tb_entity_amt_mean"] is None
    assert f1["tb_entity_amt_std"] == 0.0
    assert f1["tb_entity_amt_zscore"] == 0.0
    assert f1["tb_entity_unique_product_count"] == 0.0
    assert f2["tb_entity_prior_count"] == 1.0  # One prior
    assert f2["tb_entity_first_seen_age"] == 100.0  # Age
    assert f2["tb_entity_recency"] == 100.0  # Recency
    assert f2["tb_entity_amt_mean"] == 50.0  # Mean
    assert f2["tb_entity_amt_std"] == 0.0  # Std of single value
    assert f2["tb_entity_amt_zscore"] == 0.0  # Z-score with zero std
    assert f2["tb_entity_unique_product_count"] == 1.0  # One unique product seen prior


def test_insufficient_history_for_std() -> None:
    """Test behavior when there's insufficient history for std calculation (n=1)."""
    df = pl.DataFrame([
        event(1, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        # Second transaction - should have n=1 prior, so std=0
        event(2, 200, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 75.0, "H"),
    ])
    prepared = prepare_entity_events(df)
    features = build_entity_features(prepared).sort("TransactionID")
    f1, f2 = features.to_dicts()

    # First transaction: no priors
    assert f1["tb_entity_prior_count"] == 0.0
    assert f1["tb_entity_amt_std"] == 0.0
    assert f1["tb_entity_amt_zscore"] == 0.0

    # Second transaction: one prior
    assert f2["tb_entity_prior_count"] == 1.0
    assert f2["tb_entity_amt_mean"] == 50.0
    assert f2["tb_entity_amt_std"] == 0.0  # Should be 0 for n=1
    # Z-score should be 0 when std=0
    assert f2["tb_entity_amt_zscore"] == 0.0


if __name__ == "__main__":
    test_first_seen_entity()
    test_repeated_entity()
    test_same_timestamp_rows_do_not_see_each_other()
    test_missing_field_handling()
    test_future_row_invariance()
    test_no_label_access()
    test_deterministic_execution()
    test_chronological_train_validation_behavior()
    test_zero_one_prior_association()
    test_insufficient_history_for_std()
    print("All entity feature unit tests passed")