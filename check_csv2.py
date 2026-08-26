with open('smoke_test_data/transaction_complete.csv', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        print(f"Line {i}: length {len(line)}")
        print(repr(line))
        if i > 4:
            break