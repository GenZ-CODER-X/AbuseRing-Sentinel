"""Direct unit tests for strict-prior temporal behavioral graph feature construction."""

from __future__ import annotations

import sys

import polars as pl

sys.path.insert(0, "scripts")
from tbgf_features import (
    build_tbgf_features,
    prepare_tbgf_events,
    TBGF_FEATURES,
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
        "TransactionAmt": amount,
    }


def test_same_timestamp_rows_do_not_see_each_other() -> None:
    """Two rows with identical TransactionDT must not see each other's contributions."""
    # First row at time 100, second row also at time 100 (same batch), third at time 101
    df = pl.DataFrame([
        event(1, 100, 10.0, 50.0),
        event(2, 100, 10.0, 55.0),
        event(3, 101, 10.0, 60.0),
    ])
    prepared = prepare_tbgf_events(df)
    features = build_tbgf_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()

    # First and second rows should see zero prior card transactions (recency None, velocity 0, etc.)
    assert f1["tb_graph_card_recency"] is None
    assert f1["tb_graph_card_velocity"] == 0.0
    assert f1["tb_graph_is_novel_card_device"] == 1.0  # (card,device) never seen
    assert f1["tb_graph_card_amt_mean"] is None
    assert f1["tb_graph_card_amt_zscore"] == 0.0
    assert f1["tb_graph_card_device_stability"] == 0.0

    assert f2["tb_graph_card_recency"] is None
    assert f2["tb_graph_card_velocity"] == 0.0
    assert f2["tb_graph_is_novel_card_device"] == 1.0  # still novel because first row not yet added to state
    assert f2["tb_graph_card_amt_mean"] is None
    assert f2["tb_graph_card_amt_zscore"] == 0.0
    assert f2["tb_graph_card_device_stability"] == 0.0

    # Third row (time 101) should see both prior rows
    assert f3["tb_graph_card_recency"] == 101 - 100  # 1.0 seconds
    # Velocity: recent window (101-600,101) -> none because prior timestamps are at 100 which is < 101-600? Actually 101-600 = -499, so recent window includes 100.
    # prior window (101-1200,101-600) = [-1099, -499) -> none.
    # So recent_count = 2, prior_count = 0 => velocity = 2.0
    assert f3["tb_graph_card_velocity"] == 2.0
    # Novelty: (card,device) already seen -> 0
    assert f3["tb_graph_is_novel_card_device"] == 0.0
    # Amount mean: average of 50 and 55 = 52.5
    assert f3["tb_graph_card_amt_mean"] == 52.5
    # Amount stddev: sqrt(((50-52.5)^2 + (55-52.5)^2)/1) = sqrt( (6.25+6.25)/1 ) = sqrt(12.5) ≈ 3.5355
    # zscore = (60 - 52.5) / 3.5355 ≈ 7.5 / 3.5355 ≈ 2.121
    expected_mean = 52.5
    expected_variance = ((50 - 52.5) ** 2 + (55 - 52.5) ** 2) / 1
    expected_std = expected_variance ** 0.5
    expected_z = (60 - expected_mean) / expected_std
    assert abs(f3["tb_graph_card_amt_zscore"] - expected_z) < 1e-9
    # Stability: prior_(card,device)_count = 2, prior_card_count = 2 => 1.0
    assert f3["tb_graph_card_device_stability"] == 1.0


def test_missing_entities_remain_missing() -> None:
    """If card or device is missing, appropriate features should be None/0."""
    row = event(4, 102, 30.0, 70.0)
    row["DeviceInfo"] = None  # missing device
    df = pl.DataFrame([row])
    prepared = prepare_tbgf_events(df)
    result = build_tbgf_features(prepared).row(0, named=True)

    # With missing device, novelty should be 1 (treated as unseen), stability 0
    assert result["tb_graph_is_novel_card_device"] == 1.0
    assert result["tb_graph_card_device_stability"] == 0.0
    # Recency, velocity, amount mean/zscore should be based on card only
    assert result["tb_graph_card_recency"] is None
    assert result["tb_graph_card_velocity"] == 0.0
    assert result["tb_graph_card_amt_mean"] is None
    assert result["tb_graph_card_amt_zscore"] == 0.0


def test_first_seen_card() -> None:
    """First appearance of a card: recency None, velocity 0, mean None, zscore 0, stability 0, novelty 1."""
    df = pl.DataFrame([event(1, 100, 10.0, 50.0)])
    prepared = prepare_tbgf_events(df)
    result = build_tbgf_features(prepared).row(0, named=True)
    assert result["tb_graph_card_recency"] is None
    assert result["tb_graph_card_velocity"] == 0.0
    assert result["tb_graph_is_novel_card_device"] == 1.0
    assert result["tb_graph_card_amt_mean"] is None
    assert result["tb_graph_card_amt_zscore"] == 0.0
    assert result["tb_graph_card_device_stability"] == 0.0


def test_first_seen_card_device() -> None:
    """First occurrence of a (card,device) pair: novelty 1, stability 0."""
    # First transaction with card A and device X
    df = pl.DataFrame([
        event(1, 100, 10.0, 50.0, card1=1, card2=2, card3=3, card4="a", card5=4, card6="b", device="devX")
    ])
    prepared = prepare_tbgf_events(df)
    result = build_tbgf_features(prepared).row(0, named=True)
    assert result["tb_graph_is_novel_card_device"] == 1.0
    assert result["tb_graph_card_device_stability"] == 0.0


def test_repeated_card_device() -> None:
    """After N prior occurrences of same pair, stability = N / (N + M) where M is other card transactions."""
    # Let's create a scenario:
    # Prior transactions:
    #   T1: card C1, device D1 (amount 10)
    #   T2: card C1, device D2 (amount 20)  # different device
    #   T3: card C1, device D1 (amount 15)  # second time with D1
    # Then we evaluate a new transaction with card C1, device D1.
    # Prior pair count for (C1,D1) = 2
    # Prior card count for C1 = 3
    # Expected stability = 2/3 ≈ 0.666...
    df = pl.DataFrame([
        event(1, 100, 10.0, 10.0, card1=111, card2=111, card3=111, card4="c1", card5=111, card6=111, device="dev1"),
        event(2, 200, 10.0, 20.0, card1=111, card2=111, card3=111, card4="c1", card5=111, card6=111, device="dev2"),
        event(3, 300, 10.0, 15.0, card1=111, card2=111, card3=111, card4="c1", card5=111, card6=111, device="dev1"),
        # The row we are evaluating:
        event(4, 400, 10.0, 30.0, card1=111, card2=111, card3=111, card4="c1", card5=111, card6=111, device="dev1"),
    ])
    prepared = prepare_tbgf_events(df)
    features = build_tbgf_features(prepared).sort("TransactionID")
    f1, f2, f3, f4 = features.to_dicts()
    # Check f4 stability
    assert abs(f4["tb_graph_card_device_stability"] - (2.0 / 3.0)) < 1e-9
    # Novelty should be 0
    assert f4["tb_graph_is_novel_card_device"] == 0.0


def test_amount_mean() -> None:
    """Running mean correctness."""
    df = pl.DataFrame([
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),
        event(3, 300, 10.0, 30.0),
    ])
    prepared = prepare_tbgf_events(df)
    features = build_tbgf_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()
    assert f1["tb_graph_card_amt_mean"] is None
    assert f2["tb_graph_card_amt_mean"] == 10.0
    assert f3["tb_graph_card_amt_mean"] == 15.0  # (10+20)/2


def test_amount_variance_zscore() -> None:
    """Welford variance and z-score (sample variance)."""
    df = pl.DataFrame([
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),
        event(3, 300, 10.0, 30.0),
    ])
    prepared = prepare_tbgf_events(df)
    features = build_tbgf_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()
    assert f1["tb_graph_card_amt_zscore"] == 0.0
    assert f2["tb_graph_card_amt_zscore"] == 0.0  # only one prior -> variance 0 -> zscore 0
    # For third row: prior amounts 10,20 => mean=15, variance=50, stddev≈7.0710678
    # zscore = (30-15)/sqrt(50) ≈ 15/7.0710678 ≈ 2.12132034
    expected_z = (30.0 - 15.0) / ((50.0) ** 0.5)
    assert abs(f3["tb_graph_card_amt_zscore"] - expected_z) < 1e-9


def test_zero_one_prior_observation() -> None:
    """Edge cases for zero/one prior observations."""
    # Zero prior: already tested in test_first_seen_card
    # One prior: stability should be 0 if device differs, or 1 if same device? Actually stability = prior_pair_count / prior_card_count.
    # If we have one prior transaction with same device, then prior_pair_count=1, prior_card_count=1 => stability=1.
    # Let's test.
    df = pl.DataFrame([
        event(1, 100, 10.0, 10.0, device="devA"),
        event(2, 200, 10.0, 20.0, device="devA"),  # same device
    ])
    prepared = prepare_tbgf_events(df)
    features = build_tbgf_features(prepared).sort("TransactionID")
    f1, f2 = features.to_dicts()
    assert f1["tb_graph_card_device_stability"] == 0.0
    assert f2["tb_graph_card_device_stability"] == 1.0
    # Novelty for second row: pair seen before => 0
    assert f2["tb_graph_is_novel_card_device"] == 0.0


def test_deterministic_execution() -> None:
    """Same input yields same output."""
    df = pl.DataFrame([
        event(1, 100, 10.0, 10.0),
        event(2, 200, 10.0, 20.0),
        event(3, 300, 10.0, 30.0),
    ])
    prepared = prepare_tbgf_events(df)
    features1 = build_tbgf_features(prepared)
    features2 = build_tbgf_features(prepared)
    # Compare the two DataFrames
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
    # The builder only expects the columns in TBGF_INPUT_COLUMNS; extra columns should be ignored.
    # However, our prepare_tbgf_events selects only TBGF_INPUT_COLUMNS, so isFraud is dropped.
    # That's fine.
    prepared = prepare_tbgf_events(df)
    # Ensure isFraud is not in the prepared DataFrame
    assert "isFraud" not in prepared.columns
    features = build_tbgf_features(prepared)
    # Should run without error
    assert features.shape[0] == 2
def test_chronological_train_validation_behavior() -> None:
    """Using the actual boundaries from validation_boundaries.py, verify train->validation flow."""
    from validation_boundaries import PARTITION_BOUNDARIES
    train_boundary = next(b for b in PARTITION_BOUNDARIES if b.name == "train")
    val_boundary = next(b for b in PARTITION_BOUNDARIES if b.name == "validation")
    # Create a small dataset that straddles the boundary
    # We'll put a train event just before validation start, and a validation event at validation start.
    # Another validation event at same TransactionDT to test same-timestamp isolation.
    train_time = train_boundary.end  # last train timestamp
    val_time = val_boundary.start    # first validation timestamp
    df = pl.DataFrame([
        event(1, train_time, 10.0, 10.0),  # train event
        event(2, val_time, 10.0, 20.0),    # validation event 1
        event(3, val_time, 20.0, 30.0),    # validation event 2 (same TransactionDT)
    ])
    prepared = prepare_tbgf_events(df)
    features = build_tbgf_features(prepared).sort("TransactionID")
    f_train, f_val1, f_val2 = features.to_dicts()
    # Train event should see nothing prior
    assert f_train["tb_graph_card_recency"] is None
    # First validation event should see the train event
    assert f_val1["tb_graph_card_recency"] == val_time - train_time
    # Second validation event (same timestamp as first) should NOT see the first validation event
    assert f_val2["tb_graph_card_recency"] == val_time - train_time  # only from train, not from other validation
    # Also, velocity etc. should reflect only train event.
    # We'll just check recency for simplicity.


def test_missing_card_isolation() -> None:
    """Two rows with incomplete card (missing some card components) must not share state."""
    # Row 1: missing card6
    row1 = event(1, 100, 10.0, 10.0, card6=None)
    # Row 2: missing card5 (different missing component)
    row2 = event(2, 200, 10.0, 20.0, card5=None)
    # Row 3: complete card (should see prior complete cards only)
    row3 = event(3, 300, 10.0, 30.0)
    df = pl.DataFrame([row1, row2, row3])
    prepared = prepare_tbgf_events(df)
    features = build_tbgf_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()
    # f1 and f2 should have card-related features as None/0/default because card is None
    assert f1["tb_graph_card_recency"] is None
    assert f1["tb_graph_card_velocity"] == 0.0
    assert f1["tb_graph_card_amt_mean"] is None
    assert f1["tb_graph_card_amt_zscore"] == 0.0
    assert f1["tb_graph_card_device_stability"] == 0.0
    # novelty should be 1.0 because card is None (treated as unseen)
    assert f1["tb_graph_is_novel_card_device"] == 1.0
    # same for f2
    assert f2["tb_graph_card_recency"] is None
    assert f2["tb_graph_card_velocity"] == 0.0
    assert f2["tb_graph_card_amt_mean"] is None
    assert f2["tb_graph_card_amt_zscore"] == 0.0
    assert f2["tb_graph_card_device_stability"] == 0.0
    assert f2["tb_graph_is_novel_card_device"] == 1.0
    # f3 should see zero prior card transactions because prior cards were None and thus not counted
    assert f3["tb_graph_card_recency"] is None
    assert f3["tb_graph_card_velocity"] == 0.0
    assert f3["tb_graph_card_amt_mean"] is None
    assert f3["tb_graph_card_amt_zscore"] == 0.0
    assert f3["tb_graph_card_device_stability"] == 0.0
    assert f3["tb_graph_is_novel_card_device"] == 1.0  # (card,device) never seen because prior card states were not updated


def test_missing_device_isolation() -> None:
    """Two rows with missing device must not affect each other's novelty/stability."""
    # Row 1: device None
    row1 = event(1, 100, 10.0, 10.0, device=None)
    # Row 2: device None again
    row2 = event(2, 200, 10.0, 20.0, device=None)
    # Row 3: device present
    row3 = event(3, 300, 10.0, 30.0, device="devX")
    df = pl.DataFrame([row1, row2, row3])
    prepared = prepare_tbgf_events(df)
    features = build_tbgf_features(prepared).sort("TransactionID")
    f1, f2, f3 = features.to_dicts()
    # f1: missing device -> novelty 1, stability 0
    assert f1["tb_graph_is_novel_card_device"] == 1.0
    assert f1["tb_graph_card_device_stability"] == 0.0
    # f2: still missing device -> novelty 1 (should not have seen previous None device as a pair)
    assert f2["tb_graph_is_novel_card_device"] == 1.0
    assert f2["tb_graph_card_device_stability"] == 0.0
    # f3: device present, should see zero prior device transactions because prior device=None rows were not counted
    assert f3["tb_graph_is_novel_card_device"] == 1.0
    assert f3["tb_graph_card_device_stability"] == 0.0


def test_velocity_boundaries() -> None:
    """Test exact boundaries for recent and prior windows."""
    # We'll set up timestamps such that we can test inclusion/exclusion at edges.
    # Use a fixed card to simplify.
    base_time = 10000
    recent_window = 600
    prior_window = 600
    # We'll create transactions at:
    # t0 = base_time - 2*recent_window  # 8800  -> should be in prior window for now = base_time
    # t1 = base_time - recent_window     # 9400  -> boundary of prior/recent? Actually t1 = now - recent_window -> should be included in recent window? recent window = [now-600, now) includes now-600.
    # t2 = base_time - 1                 # 9999  -> inside recent window
    # t3 = base_time                     # 10000 -> current now (not yet in state)
    # We'll evaluate a row at now = base_time.
    # Prior transactions: t0, t1, t2
    # Expected recent count: t1 and t2 (since t1 = now-600 included, t2 = now-1 included) => 2
    # Expected prior count: t0 only (t0 = now-1200 included, t1 = now-600 excluded from prior) => 1
    # Velocity = recent - prior = 2 - 1 = 1
    df = pl.DataFrame([
        event(1, base_time - 2*recent_window, 10.0, 10.0, card1=1, card2=2, card3=3, card4="c", card5=5, card6=6),
        event(2, base_time - recent_window, 10.0, 10.0, card1=1, card2=2, card3=3, card4="c", card5=5, card6=6),
        event(3, base_time - 1, 10.0, 10.0, card1=1, card2=2, card3=3, card4="c", card5=5, card6=6),
        # current row:
        event(4, base_time, 10.0, 10.0, card1=1, card2=2, card3=3, card4="c", card5=5, card6=6),
    ])
    prepared = prepare_tbgf_events(df)
    features = build_tbgf_features(prepared).sort("TransactionID")
    f1, f2, f3, f4 = features.to_dicts()
    assert f4["tb_graph_card_velocity"] == 1.0
    # Also test a timestamp just outside recent window (now-600 - epsilon) should not be counted in recent.
    # We'll do another set.
    epsilon = 1e-9
    df2 = pl.DataFrame([
        event(1, base_time - recent_window - epsilon, 10.0, 10.0, card1=1, card2=2, card3=3, card4="c", card5=5, card6=6),
        event(2, base_time, 10.0, 10.0, card1=1, card2=2, card3=3, card4="c", card5=5, card6=6),
    ])
    prepared2 = prepare_tbgf_events(df2)
    features2 = build_tbgf_features(prepared2).sort("TransactionID")
    g1, g2 = features2.to_dicts()
    # The prior timestamp is just outside recent window, so recent count = 0, prior count = ? 
    # prior window = [now-1200, now-600). The timestamp is now-600-epsilon, which is < now-600, so it falls in prior window if >= now-1200.
    # Assuming epsilon small, it is in prior window.
    # So prior count = 1, recent count = 0 => velocity = -1
    assert g2["tb_graph_card_velocity"] == -1.0


if __name__ == "__main__":
    test_same_timestamp_rows_do_not_see_each_other()
    test_missing_entities_remain_missing()
    test_first_seen_card()
    test_first_seen_card_device()
    test_repeated_card_device()
    test_amount_mean()
    test_amount_variance_zscore()
    test_zero_one_prior_observation()
    test_deterministic_execution()
    test_no_label_access()
    test_chronological_train_validation_behavior()
    test_missing_card_isolation()
    test_missing_device_isolation()
    test_velocity_boundaries()
    print("All TBGF unit tests passed")
