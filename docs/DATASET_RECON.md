# Dataset reconnaissance

## Scope

This reconnaissance uses only the IEEE Fraud Detection training files supplied
at invocation time: `train_transaction.csv` and `train_identity.csv`. It does
not read test data, construct identity keys, construct graphs, calculate entity
fraud rates, or create ML/application artifacts.

## Run

```sh
python scripts/recon_dataset.py \
  --transaction /path/to/train_transaction.csv \
  --identity /path/to/train_identity.csv
```

Artifacts are written to `artifacts/recon` by default. Pass `--output-dir` to
choose another location.

## Parsing and reproducibility

For each input CSV, Polars derives a schema once by scanning the complete file
with `infer_schema_length=None`. Every subsequent scan of that source receives
that explicit schema. Parsing keeps empty fields as nulls, uses no extra null
tokens, disables date parsing, and fails rather than silently ignoring parse
errors. Results use exact counts rather than samples or approximations.

Top entity values are ordered by descending frequency and then ascending value,
making ties deterministic. Temporal quantiles use Polars `nearest`
interpolation at 0%, 1%, 5%, 25%, 50%, 75%, 95%, 99%, and 100%.

## Artifacts

- `schema.json`: source paths, exact dimensions, ordered column names and
  Polars physical dtypes, parsing policy, and transaction-ID duplication count.
- `missingness.csv`: exact null count and percentage for every transaction and
  identity column.
- `cardinality.csv`: transaction-only results for `card1`, `card2`, `card3`,
  `card5`, `addr1`, `addr2`, `P_emaildomain`, and `R_emaildomain`.
- `entity_stats.json`: deterministic top-20 frequency values for those same
  transaction-only candidate entities.
- `fraud_stats.json`: transaction target class counts and rate only.
- `temporal_stats.json`: transaction-time bounds, exact distinct count, null
  count, and fixed quantiles.
- `identity_coverage.json`: identity row/unique-ID/duplicate-ID counts,
  transaction ID coverage, and identity IDs absent from transactions.

For entity cardinality, nulls are excluded from distinct, singleton, and top
value calculations because null behavior is reported separately. Uniqueness
ratio and singleton percentage use the non-null row count as their denominator.
"Duplicate ... rows beyond first" is `row_count - unique_TransactionID_count`.

## Memory behavior

The script uses Polars lazy CSV scans and collects scalar aggregations. The only
non-scalar collection is one frequency table for each requested transaction
entity; it contains one row per distinct non-null value, never one row per
transaction. Identity coverage uses lazy semi/anti joins followed immediately
by counts; no transaction-identity join is materialized.

<!-- GENERATED SUMMARY START -->
## Generated results

| Measure | Result |
| --- | ---: |
| Train transaction rows / columns | 590,540 / 394 |
| Duplicate transaction-ID rows beyond first | 0 |
| Train identity rows / columns | 144,233 / 41 |
| `isFraud=0` / `isFraud=1` | 569,877 / 20,663 |
| Fraud rate | 3.49900091% |
| `TransactionDT` min / max | 86,400 / 15,811,131 |
| Unique non-null `TransactionDT` | 573,349 |
| Unique identity `TransactionID` | 144,233 |
| Duplicate identity-ID rows beyond first | 0 |
| Transaction IDs matching identity | 144,233 (24.423917%) |
| Identity IDs absent from transaction | 0 |
<!-- GENERATED SUMMARY END -->
