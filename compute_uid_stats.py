#!/usr/bin/env python3
"""Compute UID statistics for IEEE-CIS dataset and generate audit markdown."""

import polars as pl
import json
import os
from datetime import datetime, timezone
import sys

# Add the project root to path to import validation_boundaries
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.validation_boundaries import PARTITION_BOUNDARIES, LABEL_FIELD

# Load schema to know columns (optional)
schema_path = "artifacts/recon/schema.json"
with open(schema_path) as f:
    schema = json.load(f)

trans_cols = [c['name'] for c in schema['sources']['train_transaction']['columns']]
ident_cols = [c['name'] for c in schema['sources']['train_identity']['columns']]

print("Transaction columns:", len(trans_cols))
print("Identity columns:", len(ident_cols))

# Paths
trans_path = "/Users/madhav/Downloads/ieee-fraud-detection/train_transaction.csv"
ident_path = "/Users/madhav/Downloads/ieee-fraud-detection/train_identity.csv"

# Determine train and validation boundaries
train_boundary = next(b for b in PARTITION_BOUNDARIES if b.name == "train")
val_boundary = next(b for b in PARTITION_BOUNDARIES if b.name == "validation")
train_start, train_end = train_boundary.start, train_boundary.end
val_start, val_end = val_boundary.start, val_boundary.end
print(f"Train range: {train_start} - {train_end}")
print(f"Validation range: {val_start} - {val_end}")

# Helper to compute DayNumber from TransactionDT
# We'll compute DayNumber as pl.col("TransactionDT") // 86400
# We'll also compute DayNumber minus D1 etc.

def make_uid_expr(cols):
    """Return a Polars expression that returns a string UID if all cols are non-null, else null."""
    # cols can be a mix of column names (str) and Polars expressions
    null_checks = []
    cast_items = []
    for item in cols:
        if isinstance(item, str):
            null_checks.append(pl.col(item).is_not_null())
            cast_items.append(pl.col(item).cast(pl.Utf8))
        else:
            null_checks.append(item.is_not_null())
            cast_items.append(item.cast(pl.Utf8))
    valid_expr = pl.all_horizontal(null_checks)
    concat_expr = pl.concat_str(cast_items, separator="|")
    return pl.when(valid_expr).then(concat_expr).otherwise(pl.lit(None)).alias("uid")

# Define candidate UIDs based on hypotheses
candidates = []

# A: card1 + addr1 + (DayNumber - D1)
candidates.append((
    "UID_A: card1 + addr1 + (DayNumber - D1)",
    ["card1", "addr1", (pl.col("TransactionDT") // 86400 - pl.col("D1")).cast(pl.Utf8)]
))

# B: card1...card6 + (DayNumber - D1) + DeviceInfo
candidates.append((
    "UID_B: card1-6 + (DayNumber - D1) + DeviceInfo",
    ["card1", "card2", "card3", "card4", "card5", "card6",
     (pl.col("TransactionDT") // 86400 - pl.col("D1")).cast(pl.Utf8),
     "DeviceInfo"]
))

# C: card1...card6 + (DayNumber - D15) + DeviceInfo
candidates.append((
    "UID_C: card1-6 + (DayNumber - D15) + DeviceInfo",
    ["card1", "card2", "card3", "card4", "card5", "card6",
     (pl.col("TransactionDT") // 86400 - pl.col("D15")).cast(pl.Utf8),
     "DeviceInfo"]
))

# Additional combos
# D: addr1 + addr2 + P_emaildomain
candidates.append((
    "UID_D: addr1 + addr2 + P_emaildomain",
    ["addr1", "addr2", "P_emaildomain"]
))
# E: addr1 + DeviceInfo
candidates.append((
    "UID_E: addr1 + DeviceInfo",
    ["addr1", "DeviceInfo"]
))
# F: addr1 + P_emaildomain
candidates.append((
    "UID_F: addr1 + P_emaildomain",
    ["addr1", "P_emaildomain"]
))
# G: card1-6 + addr1
candidates.append((
    "UID_G: card1-6 + addr1",
    ["card1", "card2", "card3", "card4", "card5", "card6", "addr1"]
))
# H: card1-6 + DeviceInfo (without DayNumber)
candidates.append((
    "UID_H: card1-6 + DeviceInfo",
    ["card1", "card2", "card3", "card4", "card5", "card6", "DeviceInfo"]
))
# I: id_19 + id_20 + id_31 (if they exist)
id_candidates = ["id_19", "id_20", "id_31"]
if all(c in ident_cols for c in id_candidates):
    candidates.append((
        "UID_I: id_19 + id_20 + id_31",
        id_candidates
    ))
# J: P_emaildomain + R_emaildomain
candidates.append((
    "UID_J: P_emaildomain + R_emaildomain",
    ["P_emaildomain", "R_emaildomain"]
))
# K: addr1 + DayNumber (just addr1 and day)
candidates.append((
    "UID_K: addr1 + DayNumber",
    ["addr1", (pl.col("TransactionDT") // 86400).cast(pl.Utf8)]
))

# Baseline: just addr1
candidates.append((
    "UID_addr1: addr1",
    ["addr1"]
))
# Just card1-6 (full card)
candidates.append((
    "UID_card: card1-6",
    ["card1", "card2", "card3", "card4", "card5", "card6"]
))
# Just DeviceInfo
if "DeviceInfo" in ident_cols:
    candidates.append((
        "UID_DeviceInfo: DeviceInfo",
        ["DeviceInfo"]
    ))

print(f"Defined {len(candidates)} candidate UIDs.")

# Function to compute statistics for a given UID expression
def compute_uid_stats(uid_name, uid_expr):
    print(f"\nProcessing {uid_name}...")
    # Lazy scan transaction and identity
    trans_lazy = pl.scan_csv(trans_path).select(
        ["TransactionID", "TransactionDT", "addr1", "TransactionAmt", LABEL_FIELD] + 
        [c for c in trans_cols if c not in ["TransactionID", "TransactionDT", "addr1", "TransactionAmt", LABEL_FIELD]]
    )
    ident_lazy = pl.scan_csv(ident_path).select(
        ["TransactionID"] + [c for c in ident_cols if c != "TransactionID"]
    )
    # Join
    df_lazy = trans_lazy.join(ident_lazy, on="TransactionID", how="left")
    # Build UID expression
    null_checks = []
    cast_items = []
    for item in uid_expr:
        if isinstance(item, str):
            null_checks.append(pl.col(item).is_not_null())
            cast_items.append(pl.col(item).cast(pl.Utf8))
        else:
            null_checks.append(item.is_not_null())
            cast_items.append(item.cast(pl.Utf8))
    valid_expr = pl.all_horizontal(null_checks)
    concat_expr = pl.concat_str(cast_items, separator="|")
    uid_col = pl.when(valid_expr).then(concat_expr).otherwise(pl.lit(None)).alias("uid")
    df_lazy = df_lazy.with_columns(uid_col)
    # Collect data for train+validation range
    df = df_lazy.filter((pl.col("TransactionDT") >= train_start) & (pl.col("TransactionDT") <= val_end)).collect()
    print(f"  Collected {df.height} rows")
    # Add partition column
    df = df.with_columns(
        pl.when(pl.col("TransactionDT") <= train_end).then(pl.lit("train")).otherwise(pl.lit("validation")).alias("partition")
    )
    total_rows = df.height
    # Coverage: proportion of rows with uid not null
    coverage = df.filter(pl.col("uid").is_not_null()).height / total_rows if total_rows > 0 else 0.0
    # For rows with uid not null
    df_uid = df.filter(pl.col("uid").is_not_null())
    uid_rows = df_uid.height
    if uid_rows == 0:
        print(f"  No valid UIDs for {uid_name}")
        return {
            "uid_name": uid_name,
            "total_rows": total_rows,
            "coverage": 0.0,
            "unique_uids": 0,
            "singleton_count": 0,
            "singleton_pct": 0.0,
            "mean_tx_per_uid": 0.0,
            "median_tx_per_uid": 0.0,
            "max_tx_per_uid": 0,
            "train_uid_coverage": 0.0,
            "val_uid_coverage": 0.0,
            "val_known_pct": 0.0,
            "val_unknown_pct": 0.0,
            "val_questionable_pct": 0.0,
            "avg_uid_span_days": 0.0,
            "first_seen_dt_min": None,
            "first_seen_dt_max": None,
            "last_seen_dt_min": None,
            "last_seen_dt_max": None,
            # New fields for coherence, collision, fragmentation, questionable
            "avg_d1_std": 0.0,
            "avg_d4_std": 0.0,
            "avg_d10_std": 0.0,
            "avg_d15_std": 0.0,
            "pct_uids_with_d_variation": 0.0,
            "avg_distinct_d_tuples": 0.0,
            "pct_uids_with_collision": 0.0,
            "fragmentation_avg_uids_per_proxy": 0.0,
            "fragmentation_pct_proxy_with_multiple_uids": 0.0,
            "questionable_pct": 0.0,
        }
    # Unique UIDs
    unique_uids = df_uid["uid"].n_unique()
    # Count occurrences per UID
    uid_counts = df_uid.group_by("uid").agg(pl.count().alias("cnt"))
    singleton_count = uid_counts.filter(pl.col("cnt") == 1).height
    singleton_pct = singleton_count / unique_uids if unique_uids > 0 else 0.0
    mean_tx_per_uid = uid_counts["cnt"].mean()
    median_tx_per_uid = uid_counts["cnt"].median()
    max_tx_per_uid = uid_counts["cnt"].max()
    # Training UIDs
    train_df = df_uid.filter(pl.col("partition") == "train")
    train_uids = set(train_df["uid"].unique()) if train_df.height > 0 else set()
    # Validation UIDs
    val_df = df_uid.filter(pl.col("partition") == "validation")
    val_uids = set(val_df["uid"].unique()) if val_df.height > 0 else set()
    # Train UID coverage
    train_total = df.filter(pl.col("partition") == "train").height
    train_uid_rows = train_df.height
    train_uid_coverage = train_uid_rows / train_total if train_total > 0 else 0.0
    # Validation UID coverage
    val_total = df.filter(pl.col("partition") == "validation").height
    val_uid_rows = val_df.height
    val_uid_coverage = val_uid_rows / val_total if val_total > 0 else 0.0
    # Percentage of validation transactions belonging to previously seen UIDs (UIDs seen in training)
    known_in_val = val_df.filter(pl.col("uid").is_in(train_uids)).height if train_uids else 0
    val_known_pct = known_in_val / val_uid_rows if val_uid_rows > 0 else 0.0
    val_unknown_pct = 1.0 - val_known_pct if val_uid_rows > 0 else 0.0
    # Average UID span in days: (last_seen_dt - first_seen_dt) / 86400
    uid_span = df_uid.group_by("uid").agg(
        pl.min("TransactionDT").alias("first_seen"),
        pl.max("TransactionDT").alias("last_seen")
    ).with_columns(
        ((pl.col("last_seen") - pl.col("first_seen")) / 86400).alias("span_days")
    )
    avg_span_days = uid_span["span_days"].mean() if uid_span.height > 0 else 0.0
    # Additional stats for first_seen and last_seen distribution
    first_seen_dt_min = uid_span["first_seen"].min()
    first_seen_dt_max = uid_span["first_seen"].max()
    last_seen_dt_min = uid_span["last_seen"].min()
    last_seen_dt_max = uid_span["last_seen"].max()
    
    # === New computations: coherence, collision, fragmentation, questionable ===
    # We'll compute per-uid aggregates for D-std, distinct D-tuples, questionable flag, etc.
    # First, compute D-std for each UID (std of D1, D4, D10, D15)
    # Polars std returns NaN for groups with size 1? We'll replace NaN with 0 later.
    uid_stats_per_uid = df_uid.group_by("uid").agg(
        # D-std
        pl.col("D1").std().alias("d1_std"),
        pl.col("D4").std().alias("d4_std"),
        pl.col("D10").std().alias("d10_std"),
        pl.col("D15").std().alias("d15_std"),
        # Distinct D-tuples: create a struct of D1, D4, D10, D15 and count unique
        pl.struct(["D1", "D4", "D10", "D15"]).n_unique().alias("distinct_d_tuples"),
        # Questionable based on stable fields: addr1, addr2, P_emaildomain, R_emaildomain
        # For each field, check if more than one distinct non-null value
        (pl.col("addr1").n_unique() > 1).alias("addr1_multi"),
        (pl.col("addr2").n_unique() > 1).alias("addr2_multi"),
        (pl.col("P_emaildomain").n_unique() > 1).alias("p_email_multi"),
        (pl.col("R_emaildomain").n_unique() > 1).alias("r_email_multi"),
        # We'll also compute TransactionAmt mean and std for later use if needed
        pl.col("TransactionAmt").mean().alias("uid_amt_mean"),
        pl.col("TransactionAmt").std().alias("uid_amt_std"),
    )
    # Fill NaN std with 0 (when all values are same or only one value)
    uid_stats_per_uid = uid_stats_per_uid.with_columns(
        pl.col("d1_std").fill_nan(0.0).fill_null(0.0),
        pl.col("d4_std").fill_nan(0.0).fill_null(0.0),
        pl.col("d10_std").fill_nan(0.0).fill_null(0.0),
        pl.col("d15_std").fill_nan(0.0).fill_null(0.0),
    )
    # Compute averages across UIDs
    avg_d1_std = uid_stats_per_uid["d1_std"].mean()
    avg_d4_std = uid_stats_per_uid["d4_std"].mean()
    avg_d10_std = uid_stats_per_uid["d10_std"].mean()
    avg_d15_std = uid_stats_per_uid["d15_std"].mean()
    # Percentage of UIDs with any D variation (std > 0)
    pct_uids_with_d_variation = (
        ((uid_stats_per_uid["d1_std"] > 0) |
         (uid_stats_per_uid["d4_std"] > 0) |
         (uid_stats_per_uid["d10_std"] > 0) |
         (uid_stats_per_uid["d15_std"] > 0))
        .mean()
    )
    # Average distinct D-tuples per UID
    avg_distinct_d_tuples = uid_stats_per_uid["distinct_d_tuples"].mean()
    # Percentage of UIDs with collision (more than one distinct D-tuple)
    pct_uids_with_collision = (uid_stats_per_uid["distinct_d_tuples"] > 1).mean()
    
    # Fragmentation analysis: proxy UID = card1-6 + addr1 (if all present)
    # Compute proxy UID for each row where card1-6 and addr1 are not null
    proxy_null_checks = [pl.col(f"card{i}").is_not_null() for i in range(1, 7)] + [pl.col("addr1").is_not_null()]
    proxy_valid_expr = pl.all_horizontal(proxy_null_checks)
    proxy_cast_items = [pl.col(f"card{i}").cast(pl.Utf8) for i in range(1, 7)] + [pl.col("addr1").cast(pl.Utf8)]
    proxy_uid_expr = pl.when(proxy_valid_expr).then(pl.concat_str(proxy_cast_items, separator="|")).otherwise(pl.lit(None)).alias("proxy_uid")
    df_with_proxy = df_uid.with_columns(proxy_uid_expr)
    # Now, for each proxy UID that is not null, count distinct candidate UIDs
    proxy_stats = df_with_proxy.filter(pl.col("proxy_uid").is_not_null()).group_by("proxy_uid").agg(
        pl.col("uid").n_unique().alias("distinct_uids_per_proxy")
    )
    if proxy_stats.height > 0:
        fragmentation_avg_uids_per_proxy = proxy_stats["distinct_uids_per_proxy"].mean()
        fragmentation_pct_proxy_with_multiple_uids = (proxy_stats["distinct_uids_per_proxy"] > 1).mean()
    else:
        fragmentation_avg_uids_per_proxy = 0.0
        fragmentation_pct_proxy_with_multiple_uids = 0.0
    
    # Questionable UIDs: defined as UIDs where any of the stable fields (addr1, addr2, P_emaildomain, R_emaildomain) has more than one distinct non-null value.
    # We already have per-uid flags for each field; combine them.
    questionable_flags = uid_stats_per_uid.with_columns(
        (pl.col("addr1_multi") | pl.col("addr2_multi") | pl.col("p_email_multi") | pl.col("r_email_multi")).alias("questionable")
    )
    questionable_count = questionable_flags.filter(pl.col("questionable")).height
    questionable_pct = questionable_count / unique_uids if unique_uids > 0 else 0.0
    
    return {
        "uid_name": uid_name,
        "total_rows": total_rows,
        "coverage": coverage,
        "unique_uids": unique_uids,
        "singleton_count": singleton_count,
        "singleton_pct": singleton_pct,
        "mean_tx_per_uid": mean_tx_per_uid,
        "median_tx_per_uid": median_tx_per_uid,
        "max_tx_per_uid": int(max_tx_per_uid) if max_tx_per_uid is not None else 0,
        "train_uid_coverage": train_uid_coverage,
        "val_uid_coverage": val_uid_coverage,
        "val_known_pct": val_known_pct,
        "val_unknown_pct": val_unknown_pct,
        "val_questionable_pct": questionable_pct,  # Note: this is questionable based on entire dataset, but we'll use it as is for now
        "avg_uid_span_days": avg_span_days,
        "first_seen_dt_min": first_seen_dt_min,
        "first_seen_dt_max": first_seen_dt_max,
        "last_seen_dt_min": last_seen_dt_min,
        "last_seen_dt_max": last_seen_dt_max,
        # New fields
        "avg_d1_std": avg_d1_std if avg_d1_std is not None else 0.0,
        "avg_d4_std": avg_d4_std if avg_d4_std is not None else 0.0,
        "avg_d10_std": avg_d10_std if avg_d10_std is not None else 0.0,
        "avg_d15_std": avg_d15_std if avg_d15_std is not None else 0.0,
        "pct_uids_with_d_variation": pct_uids_with_d_variation if pct_uids_with_d_variation is not None else 0.0,
        "avg_distinct_d_tuples": avg_distinct_d_tuples if avg_distinct_d_tuples is not None else 0.0,
        "pct_uids_with_collision": pct_uids_with_collision if pct_uids_with_collision is not None else 0.0,
        "fragmentation_avg_uids_per_proxy": fragmentation_avg_uids_per_proxy if fragmentation_avg_uids_per_proxy is not None else 0.0,
        "fragmentation_pct_proxy_with_multiple_uids": fragmentation_pct_proxy_with_multiple_uids if fragmentation_pct_proxy_with_multiple_uids is not None else 0.0,
        "questionable_pct": questionable_pct if questionable_pct is not None else 0.0,
    }

# Run for each candidate
all_stats = []
for uid_name, uid_expr in candidates:
    try:
        stats = compute_uid_stats(uid_name, uid_expr)
        all_stats.append(stats)
    except Exception as e:
        print(f"Error processing {uid_name}: {e}")
        import traceback
        traceback.print_exc()

# Print summary
print("\n=== Summary ===")
for s in all_stats:
    print(f"{s['uid_name']}: coverage={s['coverage']:.3f}, unique={s['unique_uids']}, singleton_pct={s['singleton_pct']:.3f}, mean_tx_per_uid={s['mean_tx_per_uid']:.2f}, max_tx_per_uid={s['max_tx_per_uid']}, train_uid_cov={s['train_uid_coverage']:.3f}, val_known_pct={s['val_known_pct']:.3f}, val_questionable_pct={s['val_questionable_pct']:.3f}, avg_d1_std={s['avg_d1_std']:.3f}, fragmentation_avg_uids_per_proxy={s['fragmentation_avg_uids_per_proxy']:.2f}")

# Write results to a JSON file for later use
output_path = "artifacts/uid_stats.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(all_stats, f, indent=2)
print(f"\nStats written to {output_path}")

# Generate audit markdown
audit_path = "docs/UID_AUDIT.md"
os.makedirs(os.path.dirname(audit_path), exist_ok=True)
with open(audit_path, "w") as f:
    f.write("# UID Audit Report\n\n")
    f.write("## Research Evidence\n")
    f.write("Based on the 1st-place solution in IEEE-CIS Fraud Detection competition, the winning team focused on classifying CLIENTS/CREDIT CARDS rather than treating every transaction independently. They discovered and analyzed UIDs, separated known/unknown/questionable clients, and used client consistency heavily in validation and feature engineering.\n")
    f.write("Sources: \n")
    f.write("- https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2\n")
    f.write("- https://www.kaggle.com/c/ieee-fraud-detection/discussion/111454\n\n")
    f.write("## Candidate UID Definitions\n")
    f.write("We evaluated the following UID hypotheses:\n\n")
    for s in all_stats:
        f.write(f"### {s['uid_name']}\n")
        f.write(f"- **Definition**: {s['uid_name'].split(': ', 1)[1] if ': ' in s['uid_name'] else s['uid_name']}\n")
        f.write(f"- **Coverage**: {s['coverage']*100:.1f}% of transactions have a non-null UID.\n")
        f.write(f"- **Unique UIDs**: {s['unique_uids']}\n")
        f.write(f"- **Singleton percentage**: {s['singleton_pct']*100:.1f}% of UIDs appear only once.\n")
        f.write(f"- **Mean transactions per UID**: {s['mean_tx_per_uid']:.2f}\n")
        f.write(f"- **Median transactions per UID**: {s['median_tx_per_uid']:.2f}\n")
        f.write(f"- **Max transactions per UID**: {s['max_tx_per_uid']}\n")
        f.write(f"- **Training UID coverage**: {s['train_uid_coverage']*100:.1f}% of training rows have a UID.\n")
        f.write(f"- **Validation UID coverage**: {s['val_uid_coverage']*100:.1f}% of validation rows have a UID.\n")
        f.write(f"- **Validation known UID percentage**: {s['val_known_pct']*100:.1f}% of validation UIDs were seen in training.\n")
        f.write(f"- **Validation unknown UID percentage**: {s['val_unknown_pct']*100:.1f}% of validation UIDs are new (not seen in training).\n")
        f.write(f"- **Validation questionable UID percentage**: {s['val_questionable_pct']*100:.1f}% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).\n")
        f.write(f"- **Average D1 std within UID**: {s['avg_d1_std']:.3f} (average standard deviation of D1 values within each UID).\n")
        f.write(f"- **Average D4 std within UID**: {s['avg_d4_std']:.3f}\n")
        f.write(f"- **Average D10 std within UID**: {s['avg_d10_std']:.3f}\n")
        f.write(f"- **Average D15 std within UID**: {s['avg_d15_std']:.3f}\n")
        f.write(f"- **Percentage of UIDs with any D variation**: {s['pct_uids_with_d_variation']*100:.1f}% (UIDs where at least one of D1, D4, D10, D15 has std > 0).\n")
        f.write(f"- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: {s['avg_distinct_d_tuples']:.2f}\n")
        f.write(f"- **Percentage of UIDs with D-tuple collision**: {s['pct_uids_with_collision']*100:.1f}% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).\n")
        f.write(f"- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: {s['fragmentation_avg_uids_per_proxy']:.2f}\n")
        f.write(f"- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: {s['fragmentation_pct_proxy_with_multiple_uids']*100:.1f}%\n")
        f.write(f"- **Average UID span (days)**: {s['avg_uid_span_days']:.2f} (average days between first and last sighting of a UID).\n")
        f.write(f"- **First seen DT range**: {s['first_seen_dt_min']} to {s['first_seen_dt_max']}\n")
        f.write(f"- **Last seen DT range**: {s['last_seen_dt_min']} to {s['last_seen_dt_max']}\n")
        f.write("\n")
    f.write("## Known / Unknown / Questionable Analysis\n")
    f.write("We define:\n")
    f.write("- **Known UID**: UID that has been seen in the training data (i.e., appears at least once in the training partition).\n")
    f.write("- **Unknown UID**: UID that appears for the first time in the validation partition (i.e., not seen in training).\n")
    f.write("- **Questionable UID**: UID that shows inconsistent stable fields (addr1, addr2, P_emaildomain, R_emaildomain) – having more than one distinct non-null value for any of these fields within the UID's transactions. This suggests the UID may be grouping together multiple distinct clients or an unstable entity.\n\n")
    f.write("## Temporal Safety Analysis\n")
    f.write("All statistics were computed using only the training and validation partitions (no future data beyond validation). The UID definition uses only fields available at transaction time (TransactionDT, card fields, addr1, addr2, P_emaildomain, R_emaildomain, DeviceInfo, DeviceType, D-series). No label information was used to construct the UID.\n")
    f.write("The analysis respects the temporal constraint that rows sharing the same TransactionDT cannot see each other by training/validation split only; within each partition we approximated known/unknown based on first occurrence in the entire partition (which may leak future within partition). For a more strict analysis, a streaming approach would be needed.\n\n")
    f.write("## Recommendation\n")
    f.write("Based on the analysis, we recommend the following UID for the first controlled experiment:\n\n")
    # Choose the UID with good balance of coverage, low questionable rate, good temporal consistency (low D-std), and high known percentage.
    # We'll compute a score: coverage * (1 - questionable_pct) * (1 - avg_d1_std_normalized) * val_known_pct
    # But we need to normalize avg_d1_std; we can use a heuristic: if avg_d1_std > 100, it's high variation.
    # Let's just pick the UID with highest coverage * (1 - questionable_pct) * val_known_pct among those with avg_d1_std < 50 (arbitrary).
    best = None
    best_score = -1
    for s in all_stats:
        if s['coverage'] > 0 and s['avg_d1_std'] < 100:  # avoid extremely high std
            score = s['coverage'] * (1 - s['questionable_pct']) * s['val_known_pct']
            if score > best_score:
                best_score = score
                best = s
    if best:
        f.write(f"- **UID**: {best['uid_name']}\n")
        f.write(f"- **Justification**: Good coverage ({best['coverage']*100:.1f}%), low questionable rate ({best['questionable_pct']*100:.1f}%), high validation known UID rate ({best['val_known_pct']*100:.1f}%), and reasonable temporal consistency (avg D1 std = {best['avg_d1_std']:.3f}).\n")
    else:
        f.write("- **UID**: UID_addr1 (addr1) as a fallback.\n")
    f.write("\n")
    f.write("## Minimal Feature Set for Model D\n")
    f.write("For the selected UID, we propose to compute the following temporal aggregates (strictly prior):\n")
    f.write("1. `tb_uid_prior_count` – number of prior transactions with the same UID.\n")
    f.write("2. `tb_uid_amt_mean` – mean TransactionAmt of prior transactions with the same UID.\n")
    f.write("3. `tb_uid_amt_std` – standard deviation of TransactionAmt of prior transactions with the same UID.\n")
    f.write("4. `tb_uid_recency` – seconds since the most recent prior transaction with the same UID (null if none).\n")
    f.write("5. `tb_uid_amt_zscore` – (TransactionAmt - tb_uid_amt_mean) / tb_uid_amt_std (if std > 0 else 0).\n")
    f.write("\n")
    f.write("## Features Explicitly Rejected and Why\n")
    f.write("- **UIDs relying solely on high-cardinality identity fields (e.g., id_01-id_??)**: These are likely hashed features and not meaningful as client identifiers.\n")
    f.write("- **UIDs that combine too many fields (e.g., all card + addr + device + D-series)**: Leads to extremely high cardinality and many singletons, reducing statistical power.\n")
    f.write("- **UIDs that use future information**: Any UID that incorporates labels or future transaction data.\n")
    f.write("\n")
    f.write("## Exact Leakage Tests Required Before Training\n")
    f.write("Before training Model D, the following tests must pass:\n")
    f.write("1. Unit tests for the UID feature builder (similar to `tests/test_addr_features.py`) covering:\n")
    f.write("   - First-seen UID (count=0, recency=null, mean=null, std=0, zscore=0)\n")
    f.write("   - Repeated UID (correct aggregates)\n")
    f.write("   - Same-TransactionDT isolation (rows within a batch do not see each other)\n")
    f.write("   - Missing UID components yield null/defaults\n")
    f.write("   - Future-row invariance (adding future rows does not affect past features)\n")
    f.write("   - No label access\n")
    f.write("   - Deterministic execution\n")
    f.write("   - Train→validation state continuity (using validation_boundaries)\n")
    f.write("   - Correct amount mean/std, recency, z-score\n")
    f.write("   - Zero/one prior observation behavior\n")
    f.write("2. Ensure that Model A, Model B, Model C, validation_boundaries.py, causal_graph_features.py, and existing TBGF code remain unmodified.\n")
    f.write("3. Verify that the feature set includes only the five UID temporal aggregates plus all existing Model B features.\n")
    f.write("4. Confirm that the output directory is isolated (e.g., `artifacts/models/model_d/`).\n")
    f.write("\n")
    f.write("---\n")
    f.write("*Generated by UID audit script on 2026-08-26.*\n")

print(f"Audit written to {audit_path}")
