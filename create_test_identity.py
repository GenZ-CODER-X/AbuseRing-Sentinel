import csv

rows = [
    [1, "device1", "typeA"],
    [2, "device2", "typeB"]
]

with open('smoke_test_data/identity_test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["TransactionID", "DeviceInfo", "DeviceType"])
    writer.writerows(rows)

print("Written identity_test.csv")