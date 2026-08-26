import csv
import json
import random

# Load the feature list from model_b to get the order
with open('artifacts/models/model_b/feature_list.json', 'r') as f:
    data = json.load(f)
all_features = data['features']  # list of 54 feature names
exclude = {"DeviceInfo", "DeviceType"}
transaction_features = [f for f in all_features if f not in exclude]
categorical_set = set(data['categorical_features'])
numeric_set = set(data['numeric_features'])

def make_value(feature_name, row_index):
    """Return a value for a feature, possibly depending on row index to introduce variation."""
    if feature_name in categorical_set:
        if feature_name == "ProductCD":
            # Cycle through a few product codes
            return random.choice(["H", "C", "R", "S"])
        elif feature_name in ["DeviceInfo", "DeviceType"]:
            # excluded, but just in case
            return f"device{row_index % 5}"
        elif feature_name in ["P_emaildomain", "R_emaildomain"]:
            return random.choice(["example@gmail.com", "example@yahoo.com", "test@hotmail.com"])
        else:
            # For card fields, addr, M features: return a small integer or float
            # We'll make them vary slowly so that UID groups appear.
            # For card1-6, we want some repetition to create UID groups.
            # Let's base on row_index modulo a small number.
            if feature_name.startswith("card"):
                # card1..card6: we'll make them integers 1-5, but with some repetition
                return float((row_index % 5) + 1)
            elif feature_name.startswith("addr"):
                return float((row_index % 3) + 10)
            elif feature_name.startswith("M"):
                return float((row_index % 2) + 1)
            else:
                # C, D features: treat as numeric but categorical in model? Actually they are numeric.
                # But they are in categorical_set? Let's check: from the JSON, C1..C14, D1..D15, M1..M9 are in categorical_features?
                # Looking at the JSON we saw earlier, categorical_features includes M1..M9 but NOT C or D.
                # So we need to check: we have categorical_set from JSON. We'll just return a float.
                return float(row_index % 10)
    else:
        # Numeric feature: return a float with some variation
        if feature_name == "TransactionAmt":
            return random.uniform(10, 500)
        elif feature_name == "TransactionDT":
            # This will be set later
            return 0.0
        elif feature_name in ["dist1", "dist2"]:
            return random.uniform(100, 10000)
        else:
            return random.uniform(0, 10)

# We'll generate 200 rows
num_rows = 200
# Split: first 120 rows train (TransactionDT 86400 to 11059199), next 80 validation (11059200 to 13391999)
# We'll assign TransactionDT linearly increasing.
train_start = 86400
train_end = 11059199
val_start = 11059200
val_end = 13391999

rows = []
for i in range(num_rows):
    if i < 120:
        # train
        dt = train_start + (train_end - train_start) * i // 119  # integer division
        is_fraud = 1 if random.random() < 0.03 else 0  # 3% fraud
    else:
        # validation
        dt = val_start + (val_end - val_start) * (i - 120) // (80 - 1)
        is_fraud = 1 if random.random() < 0.03 else 0

    row = [i+1]  # TransactionID
    # Features in order of transaction_features
    for feat in transaction_features:
        if feat == "TransactionDT":
            val = float(dt)
        else:
            val = make_value(feat, i)
        row.append(val)
    row.append(is_fraud)
    rows.append(row)

# Write transaction CSV
with open('smoke_test_data/transaction_large.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    header = ["TransactionID"] + transaction_features + ["isFraud"]
    writer.writerow(header)
    writer.writerows(rows)

print(f"Written transaction_large.csv with {num_rows} rows")

# Create identity CSV: one unique device per transaction? Actually DeviceInfo and DeviceType are per transaction.
# We'll make them vary per row as well.
identity_rows = []
for i in range(num_rows):
    identity_rows.append([i+1, f"device{i % 10}", f"type{i % 2}"])

with open('smoke_test_data/identity_large.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["TransactionID", "DeviceInfo", "DeviceType"])
    writer.writerows(identity_rows)

print(f"Written identity_large.csv with {num_rows} rows")