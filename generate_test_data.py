import csv
import json

# Load the feature list from model_b to get the order
with open('artifacts/models/model_b/feature_list.json', 'r') as f:
    data = json.load(f)
all_features = data['features']  # list of 54 feature names
print(f"All features ({len(all_features)}): {all_features}")

# Remove DeviceInfo and DeviceType to get transaction feature columns
exclude = {"DeviceInfo", "DeviceType"}
transaction_features = [f for f in all_features if f not in exclude]
print(f"Transaction features ({len(transaction_features)}): {transaction_features}")

# The CSV columns: TransactionID, then transaction_features, then isFraud
csv_header = ["TransactionID"] + transaction_features + ["isFraud"]
print(f"CSV header length: {len(csv_header)}")
print(f"First few: {csv_header[:5]}")
print(f"Last few: {csv_header[-5:]}")

# Create two rows: one for train, one for validation
# We'll fill with dummy values, but ensure correct data types where needed.
# For simplicity, we'll set most numeric values to 1.0, categorical strings to appropriate values.
# We need to know which features are categorical vs numeric. We'll use the categorical_features list from the same JSON.
categorical_set = set(data['categorical_features'])
numeric_set = set(data['numeric_features'])
# Note: Some features may be in both? Actually, the JSON separates them, but a feature is either categorical or numeric.
# We'll trust that.

def make_value(feature_name):
    """Return a dummy value for a feature."""
    if feature_name in categorical_set:
        # For simplicity, return a string that looks like a category index as string? But the CSV expects raw values.
        # In the original data, categorical fields like ProductCD are strings like 'H', card1 etc are floats but treated as categorical.
        # We'll return a float for card fields, string for ProductCD, etc.
        if feature_name == "ProductCD":
            return "H"
        elif feature_name in ["DeviceInfo", "DeviceType"]:
            # These are excluded, but just in case
            return "device1"
        elif feature_name in ["P_emaildomain", "R_emaildomain"]:
            return "example@gmail.com"
        else:
            # For card1, card2, etc, and addr1, addr2, and the M features, they are numeric in the raw data but treated as categorical.
            # We'll return a float.
            return 1.0
    else:
        # Numeric feature
        return 1.0

# Build a row of feature values in the order of transaction_features
def make_feature_row():
    return [make_value(f) for f in transaction_features]

# Row 1: train (TransactionDT within train range: 86400 to 11059199)
row1 = [1]  # TransactionID
row1 += make_feature_row()
# We need to set TransactionDT specifically to a value within train range.
# Find the index of "TransactionDT" in transaction_features
dt_index = transaction_features.index("TransactionDT")
row1[1 + dt_index] = 100000.0  # overwrite the dummy value
# Set isFraud
row1.append(0)

# Row 2: validation (TransactionDT within validation range: 11059200 to 13391999)
row2 = [2]  # TransactionID
row2 += make_feature_row()
dt_index2 = transaction_features.index("TransactionDT")
row2[1 + dt_index2] = 11060000.0
row2.append(1)

# Write to CSV
with open('smoke_test_data/transaction_final.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerow(row1)
    writer.writerow(row2)

print("Written transaction_final.csv")

# Now create identity CSV: TransactionID, DeviceInfo, DeviceType
identity_rows = [
    [1, "device1", "typeA"],
    [2, "device2", "typeB"]
]
with open('smoke_test_data/identity_final.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["TransactionID", "DeviceInfo", "DeviceType"])
    writer.writerows(rows)

print("Written identity_final.csv")