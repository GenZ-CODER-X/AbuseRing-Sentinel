import polars as pl

# Read the CSV and see what columns it has
df = pl.scan_csv("smoke_test_data/transaction_complete.csv")
print("Columns in transaction CSV:", df.collect_schema().names())
print("Number of columns:", len(df.collect_schema().names()))

# Read identity CSV
df_id = pl.scan_csv("smoke_test_data/identity_complete.csv")
print("Columns in identity CSV:", df_id.collect_schema().names())
print("Number of columns:", len(df_id.collect_schema().names()))