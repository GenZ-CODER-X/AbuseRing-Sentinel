import csv

# Define the feature names in order as they should appear in the CSV
# This order must match the order used in train_model_b.py for transaction_columns
# plus the two identity fields later.
# From train_model_b.py: transaction_columns = [feature for feature in model_b.FEATURES if feature not in {"DeviceInfo", "DeviceType"}]
# We need to know model_b.FEATURES order. Let's infer from train_model_a.py and the added fields.
# model_a.FEATURES order: CORE_FEATURES + ENTITY_FEATURES + C_FEATURES + D_FEATURES + M_FEATURES
# CORE_FEATURES: TransactionAmt, ProductCD, CHRONOLOGICAL_FIELD, dist1, dist2
# ENTITY_FEATURES: card1, card2, card3, card5, addr1, addr2, P_emaildomain, R_emaildomain
# C_FEATURES: C1,C2,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14 (skip C3)
# D_FEATURES: D1..D15
# M_FEATURES: M1..M9
# Then added raw categorical features: DeviceInfo, DeviceType, card4, card6 (but these are omitted from transaction CSV)
# So transaction_columns order is:
# TransactionAmt, ProductCD, CHRONOLOGICAL_FIELD, dist1, dist2,
# card1, card2, card3, card5, addr1, addr2, P_emaildomain, R_emaildomain,
# C1,C2,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,
# D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,
# M1,M2,M3,M4,M5,M6,M7,M8,M9,
# card4, card6   # because they are in model_b.FEATURES but not in the excluded set? Wait, they are NOT in the excluded set (excluded are DeviceInfo, DeviceType). So card4 and card6 are included.
# However, note that in model_b.FEATURES, the added features are appended at the end? Let's check train_model_b.py:
# ADDED_RAW_CATEGORICAL_FEATURES = ("DeviceInfo", "DeviceType", "card4", "card6")
# FEATURES = model_a.FEATURES + ADDED_RAW_CATEGORICAL_FEATURES
# So indeed, card4 and card6 are at the end after DeviceInfo and DeviceType.
# But we exclude DeviceInfo and DeviceType, so the transaction_columns will have model_a.FEATURES followed by card4, card6.
# So the order is as above.

feature_names = [
    "TransactionAmt",
    "ProductCD",
    "TransactionDT",  # CHRONOLOGICAL_FIELD
    "dist1",
    "dist2",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "C1","C2","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14",
    "D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15",
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
    "card4",
    "card6"
]

# Verify we have 52 feature names
print(f"Number of feature names: {len(feature_names)}")
assert len(feature_names) == 52

# Now create rows
rows = []
# Row 1: train
row1 = [
    1,                          # TransactionID
    100000,                     # TransactionDT
    0,                          # isFraud
    100.0,                      # TransactionAmt
    "H",                        # ProductCD
    100000,                     # dist1
    200000,                     # dist2
    1.0, 2.0, 3.0, 5.0,         # card1, card2, card3, card5
    10.0, 20.0,                 # addr1, addr2
    "example@gmail.com",        # P_emaildomain
    "example@yahoo.com",        # R_emaildomain
    # C1..C14 except C3
    1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,  # C1,C2,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14
    # D1..D15
    1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
    # M1..M9
    1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
    # card4, card6
    4.0, 6.0
]
# Row 2: validation
row2 = [
    2,
    11060000,
    1,
    150.0,
    "H",
    101000,
    201000,
    2.0, 3.0, 4.0, 6.0,
    11.0, 21.0,
    "example@gmail.com",
    "example@yahoo.com",
    # C's
    2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    # D's
    2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    # M's
    2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,
    # card4, card6
    4.0, 6.0
]

rows.append(row1)
rows.append(row2)

# Write to file
with open('smoke_test_data/transaction_test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # Header: TransactionID, TransactionDT, isFraud, then feature_names
    header = ["TransactionID", "TransactionDT", "isFraud"] + feature_names
    writer.writerow(header)
    writer.writerows(rows)

print("Written transaction_test.csv")