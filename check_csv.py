import csv
with open('smoke_test_data/transaction_complete.csv', 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        print(f"Line {i}: {len(row)} fields")
        if i < 5:
            print(f"  First few: {row[:5]}")
        if len(row) != 54:
            print(f"  Mismatch: expected 54, got {len(row)}")