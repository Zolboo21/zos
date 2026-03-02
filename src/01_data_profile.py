"""Quick data profiling — prints structural overview of dataset.csv"""

import pandas as pd
import sys

DATA = "dataset.csv"
SAMPLE_N = 500_000  # sample for heavy ops

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA, parse_dates=["timestamp"])
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} cols\n")

# --- dtypes ---
print("=" * 60)
print("DTYPES")
print("=" * 60)
print(df.dtypes.to_string())

# --- nulls ---
print("\n" + "=" * 60)
print("NULL COUNTS & PERCENTAGES")
print("=" * 60)
nulls = df.isnull().sum()
null_pct = (nulls / len(df) * 100).round(2)
null_df = pd.DataFrame({"nulls": nulls, "pct": null_pct})
print(null_df[null_df["nulls"] > 0].to_string())
if null_df["nulls"].sum() == 0:
    print("No nulls found.")

# --- unique counts ---
print("\n" + "=" * 60)
print("UNIQUE VALUE COUNTS")
print("=" * 60)
for col in df.columns:
    print(f"  {col}: {df[col].nunique():,}")

# --- numeric stats ---
print("\n" + "=" * 60)
print("NUMERIC STATS")
print("=" * 60)
num_cols = df.select_dtypes(include="number").columns.tolist()
print(df[num_cols].describe().round(2).to_string())

# --- categorical value counts (top 10) ---
cat_cols = ["transaction_type", "channel", "status", "location_state", "merchant_category_code"]
print("\n" + "=" * 60)
print("CATEGORICAL VALUE COUNTS (top 10)")
print("=" * 60)
for col in cat_cols:
    if col not in df.columns:
        continue
    print(f"\n--- {col} ---")
    vc = df[col].value_counts(dropna=False).head(10)
    for val, cnt in vc.items():
        print(f"  {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

# --- duplicates ---
print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)
dup_txn = df["transaction_id"].duplicated().sum()
print(f"Duplicate transaction_ids: {dup_txn:,}")
full_dups = df.duplicated().sum()
print(f"Fully duplicate rows: {full_dups:,}")

# --- timestamp range ---
print("\n" + "=" * 60)
print("TIMESTAMP RANGE")
print("=" * 60)
print(f"Min: {df['timestamp'].min()}")
print(f"Max: {df['timestamp'].max()}")

# --- balance consistency check (sample) ---
print("\n" + "=" * 60)
print(f"BALANCE CONSISTENCY CHECK (sample of {SAMPLE_N:,})")
print("=" * 60)
sample = df.sample(n=min(SAMPLE_N, len(df)), random_state=42)
debit = sample[sample["transaction_type"] == "debit"]
credit = sample[sample["transaction_type"] == "credit"]

debit_ok = ((debit["balance_before_ngn"] - debit["amount"] - debit["balance_after_ngn"]).abs() < 0.01).mean()
credit_ok = ((credit["balance_before_ngn"] + credit["amount"] - credit["balance_after_ngn"]).abs() < 0.01).mean()
print(f"Debit balance consistent: {debit_ok*100:.2f}%")
print(f"Credit balance consistent: {credit_ok*100:.2f}%")

# --- negative amounts / balances ---
print("\n" + "=" * 60)
print("ANOMALY CHECK")
print("=" * 60)
print(f"Negative amounts: {(df['amount'] < 0).sum():,}")
print(f"Negative balance_before: {(df['balance_before_ngn'] < 0).sum():,}")
print(f"Negative balance_after: {(df['balance_after_ngn'] < 0).sum():,}")
print(f"Zero amounts: {(df['amount'] == 0).sum():,}")

print("\n" + "=" * 60)
print("SAMPLE ROWS")
print("=" * 60)
print(df.head(3).to_string())
print("\nDone.")
