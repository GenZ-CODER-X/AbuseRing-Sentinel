"""Direct unit tests for strict-prior temporal entity feature construction (Model E final)."""

from __future__ import annotations

import sys
import math

import polars as pl

sys.path.insert(0, "scripts")
from entity_features_final import (
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
    """First appearance of entity: count=0, mean=null, std=0, entropy=0, velocity=0, first_seen=0, unique_products=0, trend=0."""
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
    assert result["tb_entity_amt_mean"] is None
    assert result["tb_entity_amt_std"] == 0.0
    assert result["tb_entity_amt_entropy"] == 0.0
    assert result["tb_entity_txn_velocity"] == 0.0
    assert result["tb_entity_amt_velocity"] == 0.0
    assert result["tb_entity_first_seen"] == 0.0
    assert result["tb_entity_unique_product_count"] == 0.0
    assert result["tb_entity_amt_trend"] == 0.0


def test_repeated_entity() -> None:
    """Test repeated entity with correct aggregates."""
    # Prior transactions:
    #   T1: amount=50.0, product=H, time=100
    #   T2: amount=100.0, product=J (new product), time=200
    # Then we evaluate a new transaction with amount=75.0, product=H, time=300
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
    assert f1["tb_entity_amt_mean"] is None
    assert f1["tb_entity_amt_std"] == 0.0
    assert f1["tb_entity_amt_entropy"] == 0.0
    assert f1["tb_entity_txn_velocity"] == 0.0
    assert f1["tb_entity_amt_velocity"] == 0.0
    assert f1["tb_entity_first_seen"] == 0.0
    assert f1["tb_entity_unique_product_count"] == 0.0
    assert f1["tb_entity_amt_trend"] == 0.0

    # T2: second transaction
    assert f2["tb_entity_prior_count"] == 1.0  # One prior
    assert f2["tb_entity_amt_mean"] == 50.0  # Mean of prior amount
    assert f2["tb_entity_amt_std"] == 0.0  # Std of single prior is 0
    # Entropy: one amount in one bin -> entropy 0
    assert f2["tb_entity_amt_entropy"] == 0.0
    # Velocity: prior_count=1, elapsed time since first seen = 100 seconds -> 100/86400 days ~0.001157 -> velocity=1/0.001157=864 approx? Wait compute: velocity = prior_count / elapsed_days.
    # elapsed_seconds = 100, elapsed_days = 100/86400 ≈ 0.001157, velocity = 1 / 0.001157 ≈ 864.0
    expected_velocity = 1.0 / (100.0 / 86400.0)
    assert abs(f2["tb_entity_txn_velocity"] - expected_velocity) < 1e-6
    # Amount velocity: sum_amt=50.0, elapsed_days same -> 50.0 / 0.001157 ≈ 43200
    expected_amt_velocity = 50.0 / (100.0 / 86400.0)
    assert abs(f2["tb_entity_amt_velocity"] - expected_amt_velocity) < 1e-6
    assert f2["tb_entity_first_seen"] == 100.0  # First seen 100 seconds ago
    assert f2["tb_entity_unique_product_count"] == 1.0  # One unique product (H) seen prior
    # Trend: only one prior point -> slope undefined -> 0
    assert f2["tb_entity_amt_trend"] == 0.0

    # T3: third transaction
    assert f3["tb_entity_prior_count"] == 2.0  # Two priors
    assert f3["tb_entity_amt_mean"] == 75.0  # Mean of [50.0, 100.0]
    # Std of [50.0, 100.0] = sqrt(((50-75)^2 + (100-75)^2)/(2-1)) = sqrt(1250) ≈ 35.355
    assert abs(f3["tb_entity_amt_std"] - 35.355) < 0.001
    # Entropy: two distinct amounts likely in same bin? 50 and 100 both in bin 0 (0-200) -> counts [2,0,0,...] -> entropy 0
    assert f3["tb_entity_amt_entropy"] == 0.0
    # Velocity: prior_count=2, elapsed time since first seen = 200 seconds -> elapsed_days = 200/86400 ≈ 0.0023148 -> velocity = 2 / 0.0023148 ≈ 864.0 (same as before? Actually 2/(200/86400)= 2*86400/200=864)
    expected_velocity = 2.0 / (200.0 / 86400.0)
    assert abs(f3["tb_entity_txn_velocity"] - expected_velocity) < 1e-6
    # Amount velocity: sum_amt=150.0, elapsed_days=200/86400 -> 150 / (200/86400)=150*86400/200=64800
    expected_amt_velocity = 150.0 / (200.0 / 86400.0)
    assert abs(f3["tb_entity_amt_velocity"] - expected_amt_velocity) < 1e-6
    assert f3["tb_entity_first_seen"] == 200.0  # First seen 200 seconds ago
    assert f3["tb_entity_unique_product_count"] == 2.0  # Two unique products (H, J) seen prior
    # Trend: slope of line through points (100,50) and (200,100) => (100-50)/(200-100)=50/100=0.5
    # Using formula: slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
    # n=2, sum_x=300, sum_y=150, sum_xy=100*50+200*100=5000+20000=25000, sum_x2=100^2+200^2=10000+40000=50000
    # numerator = 2*25000 - 300*150 = 50000 - 45000 = 5000
    # denominator = 2*50000 - 300^2 = 100000 - 90000 = 10000
    # slope = 5000/10000 = 0.5
    assert abs(f3["tb_entity_amt_trend"] - 0.5) < 1e-6


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
    assert f1["tb_entity_amt_mean"] is None
    assert f1["tb_entity_amt_std"] == 0.0
    assert f1["tb_entity_amt_entropy"] == 0.0
    assert f1["tb_entity_txn_velocity"] == 0.0
    assert f1["tb_entity_amt_velocity"] == 0.0
    assert f1["tb_entity_first_seen"] == 0.0
    assert f1["tb_entity_unique_product_count"] == 0.0
    assert f1["tb_entity_amt_trend"] == 0.0

    assert f2["tb_entity_prior_count"] == 0.0  # Should NOT see T1's transaction
    assert f2["tb_entity_amt_mean"] is None
    assert f2["tb_entity_amt_std"] == 0.0
    assert f2["tb_entity_amt_entropy"] == 0.0
    assert f2["tb_entity_txn_velocity"] == 0.0
    assert f2["tb_entity_amt_velocity"] == 0.0
    assert f2["tb_entity_first_seen"] == 0.0
    assert f2["tb_entity_unique_product_count"] == 0.0
    assert f2["tb_entity_amt_trend"] == 0.0

    # Third row (time 200) should see both prior rows' contributions
    assert f3["tb_entity_prior_count"] == 2.0
    assert f3["tb_entity_amt_mean"] == 75.0  # Mean of [50.0, 100.0]
    assert abs(f3["tb_entity_amt_std"] - 35.355) < 0.001
    assert f3["tb_entity_amt_entropy"] == 0.0  # both amounts in bin 0
    # Velocity: prior_count=2, elapsed time since first seen = 100 seconds? Wait first seen at time 100, now at 200 -> elapsed=100 seconds
    expected_velocity = 2.0 / (100.0 / 86400.0)
    assert abs(f3["tb_entity_txn_velocity"] - expected_velocity) < 1e-6
    # Amount velocity: sum_amt=150.0, elapsed_seconds=100 -> 150 / (100/86400)=150*86400/100=129600
    expected_amt_velocity = 150.0 / (100.0 / 86400.0)
    assert abs(f3["tb_entity_amt_velocity"] - expected_amt_velocity) < 1e-6
    assert f3["tb_entity_first_seen"] == 100.0  # First seen at time 100
    assert f3["tb_entity_unique_product_count"] == 2.0  # H and J
    # Trend: both priors have same timestamp (100), so slope undefined -> 0
    assert f3["tb_entity_amt_trend"] == 0.0


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
    assert f1["tb_entity_amt_mean"] is None
    assert f1["tb_entity_amt_std"] == 0.0
    assert f1["tb_entity_amt_entropy"] == 0.0
    assert f1["tb_entity_txn_velocity"] == 0.0
    assert f1["tb_entity_amt_velocity"] == 0.0
    assert f1["tb_entity_first_seen"] == 0.0
    assert f1["tb_entity_unique_product_count"] == 0.0
    assert f1["tb_entity_amt_trend"] == 0.0

    # f2: missing addr1 -> no entity possible
    assert f2["tb_entity_prior_count"] == 0.0
    assert f2["tb_entity_amt_mean"] is None
    assert f2["tb_entity_amt_std"] == 0.0
    assert f2["tb_entity_amt_entropy"] == 0.0
    assert f2["tb_entity_txn_velocity"] == 0.0
    assert f2["tb_entity_amt_velocity"] == 0.0
    assert f2["tb_entity_first_seen"] == 0.0
    assert f2["tb_entity_unique_product_count"] == 0.0
    assert f2["tb_entity_amt_trend"] == 0.0

    # f3: all present, but prior missing rows didn't create entities
    assert f3["tb_entity_prior_count"] == 0.0
    assert f3["tb_entity_amt_mean"] is None
    assert f3["tb_entity_amt_std"] == 0.0
    assert f3["tb_entity_amt_entropy"] == 0.0
    assert f3["tb_entity_txn_velocity"] == 0.0
    assert f3["tb_entity_amt_velocity"] == 0.0
    assert f3["tb_entity_first_seen"] == 0.0
    assert f3["tb_entity_unique_product_count"] == 0.0
    assert f3["tb_entity_amt_trend"] == 0.0


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
    assert f_train["tb_entity_amt_mean"] is None
    assert f_train["tb_entity_amt_std"] == 0.0
    assert f_train["tb_entity_amt_entropy"] == 0.0
    assert f_train["tb_entity_txn_velocity"] == 0.0
    assert f_train["tb_entity_amt_velocity"] == 0.0
    assert f_train["tb_entity_first_seen"] == 0.0
    assert f_train["tb_entity_unique_product_count"] == 0.0
    assert f_train["tb_entity_amt_trend"] == 0.0

    # First validation event should see the train event
    assert f_val1["tb_entity_prior_count"] == 1.0  # One prior transaction
    assert f_val1["tb_entity_amt_mean"] == 50.0  # Mean of prior amount
    assert f_val1["tb_entity_amt_std"] == 0.0  # Std of single prior is 0
    assert f_val1["tb_entity_amt_entropy"] == 0.0
    # Velocity: prior_count=1, elapsed time since first seen = (val_time - train_time)
    elapsed = val_time - train_time
    expected_velocity = 1.0 / (elapsed / 86400.0)
    assert abs(f_val1["tb_entity_txn_velocity"] - expected_velocity) < 1e-6
    # Amount velocity: sum_amt=50.0, same elapsed
    expected_amt_velocity = 50.0 / (elapsed / 86400.0)
    assert abs(f_val1["tb_entity_amt_velocity"] - expected_amt_velocity) < 1e-6
    assert f_val1["tb_entity_first_seen"] == elapsed  # Age since first seen
    assert f_val1["tb_entity_unique_product_count"] == 1.0  # One unique product (H) seen prior
    # Trend: only one prior -> 0
    assert f_val1["tb_entity_amt_trend"] == 0.0

    # Second validation event (same timestamp as first) should NOT see the first validation event
    assert f_val2["tb_entity_prior_count"] == 1.0  # Only from train (not from same-timestamp validation)
    assert f_val2["tb_entity_amt_mean"] == 50.0  # Mean of prior amount (still only train)
    assert f_val2["tb_entity_amt_std"] == 0.0  # Std of single prior
    assert f_val2["tb_entity_amt_entropy"] == 0.0
    # Velocity: prior_count=1, elapsed time since first seen = (val_time - train_time) (same as above)
    assert abs(f_val2["tb_entity_txn_velocity"] - expected_velocity) < 1e-6
    assert abs(f_val2["tb_entity_amt_velocity"] - expected_amt_velocity) < 1e-6
    assert f_val2["tb_entity_first_seen"] == elapsed  # Age since first seen (still only train)
    assert f_val2["tb_entity_unique_product_count"] == 1.0  # One unique product (H) from train only
    # Trend: still only one prior -> 0
    assert f_val2["tb_entity_amt_trend"] == 0.0


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
    assert f1["tb_entity_amt_mean"] is None
    assert f1["tb_entity_amt_std"] == 0.0
    assert f1["tb_entity_amt_entropy"] == 0.0
    assert f1["tb_entity_txn_velocity"] == 0.0
    assert f1["tb_entity_amt_velocity"] == 0.0
    assert f1["tb_entity_first_seen"] == 0.0
    assert f1["tb_entity_unique_product_count"] == 0.0
    assert f1["tb_entity_amt_trend"] == 0.0
    assert f2["tb_entity_prior_count"] == 1.0  # One prior
    assert f2["tb_entity_amt_mean"] == 50.0  # Mean
    assert f2["tb_entity_amt_std"] == 0.0  # Std of single value
    assert f2["tb_entity_amt_entropy"] == 0.0  # One value -> entropy 0
    # Velocity: prior_count=1, elapsed time since first seen = 100 seconds
    expected_velocity = 1.0 / (100.0 / 86400.0)
    assert abs(f2["tb_entity_txn_velocity"] - expected_velocity) < 1e-6
    expected_amt_velocity = 50.0 / (100.0 / 86400.0)
    assert abs(f2["tb_entity_amt_velocity"] - expected_amt_velocity) < 1e-6
    assert f2["tb_entity_first_seen"] == 100.0
    assert f2["tb_entity_unique_product_count"] == 1.0  # One unique product seen prior
    # Trend: only one prior -> 0
    assert f2["tb_entity_amt_trend"] == 0.0


def test_entropy_different_bins() -> None:
    """Test entropy when amounts fall into different bins."""
    # Amounts: 50 (bin0), 250 (bin1), 450 (bin2) -> three different bins each count=1 -> uniform distribution over 3 bins -> entropy = log2(3) ≈ 1.585
    df = pl.DataFrame([
        event(1, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        event(2, 200, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 250.0, "H"),
        event(3, 300, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 450.0, "H"),
        # Row to evaluate: we want to see the entropy of the three priors
        event(4, 400, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 0.0, "H"),  # amount irrelevant
    ])
    prepared = prepare_entity_events(df)
    features = build_entity_features(prepared).sort("TransactionID")
    f1, f2, f3, f4 = features.to_dicts()
    # We only care about f4 (the row with amount 0)
    # Prior count = 3
    assert f4["tb_entity_prior_count"] == 3.0
    # Mean of [50,250,450] = 250.0
    assert f4["tb_entity_amt_mean"] == 250.0
    # Std: compute quickly: variance = ((50-250)^2+(250-250)^2+(450-250)^2)/2 = (40000+0+40000)/2=40000 -> std=200.0
    assert abs(f4["tb_entity_amt_std"] - 200.0) < 0.001
    # Entropy: three equally likely bins -> entropy = log2(3) ≈ 1.58496
    expected_entropy = math.log2(3)
    assert abs(f4["tb_entity_amt_entropy"] - expected_entropy) < 1e-6
    # Other fields not critical for this test


def test_velocity_zero_elapsed() -> None:
    """When elapsed time is zero (same timestamp as first seen), velocity should be 0."""
    # Two transactions at same timestamp: first sets first_seen, second sees prior count=1 but elapsed=0 -> velocity=0
    df = pl.DataFrame([
        event(1, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        event(2, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 100.0, "H"),  # same timestamp
    ])
    prepared = prepare_entity_events(df)
    features = build_entity_features(prepared).sort("TransactionID")
    f1, f2 = features.to_dicts()
    # First transaction: prior count 0 -> velocity 0
    assert f1["tb_entity_txn_velocity"] == 0.0
    assert f1["tb_entity_amt_velocity"] == 0.0
    # Second transaction: prior count=1, elapsed time since first seen = 0 (same timestamp) -> velocity should be 0 (to avoid division by zero)
    assert f2["tb_entity_txn_velocity"] == 0.0
    assert f2["tb_entity_amt_velocity"] == 0.0
    # First seen age should be 0 for second? Actually first seen at time 100, now at 100 -> age=0
    assert f2["tb_entity_first_seen"] == 0.0


def test_trend_constant_amount() -> None:
    """When amount is constant, trend should be 0."""
    df = pl.DataFrame([
        event(1, 100, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        event(2, 200, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        event(3, 300, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 50.0, "H"),
        event(4, 400, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 0.0, "H"),  # evaluate
    ])
    prepared = prepare_entity_events(df)
    features = build_entity_features(prepared).sort("TransactionID")
    f1, f2, f3, f4 = features.to_dicts()
    # For f4, priors amounts = [50,50,50] -> mean=50, std=0, trend should be 0
    assert f4["tb_entity_amt_mean"] == 50.0
    assert f4["tb_entity_amt_std"] == 0.0
    assert f4["tb_entity_amt_trend"] == 0.0


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
    test_entropy_different_bins()
    test_velocity_zero_elapsed()
    test_trend_constant_amount()
    print("All entity feature final unit tests passed")