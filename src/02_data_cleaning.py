"""Data cleaning — reads dataset.csv, cleans, saves to output/cleaned.csv"""

import pandas as pd
import numpy as np

DATA = "dataset.csv"
OUT = "output/cleaned.csv"

print("Loading data...")
df = pd.read_csv(DATA, parse_dates=["timestamp"])
n_orig = len(df)
print(f"Original shape: {df.shape}")

# --- 1. Drop full duplicates ---
df = df.drop_duplicates()
print(f"After dropping full duplicates: {len(df):,} (removed {n_orig - len(df):,})")

# --- 2. Drop duplicate transaction_ids (keep first) ---
n_before = len(df)
df = df.drop_duplicates(subset=["transaction_id"], keep="first")
print(f"After dropping duplicate transaction_ids: {len(df):,} (removed {n_before - len(df):,})")

# --- 3. Parse and validate timestamp ---
invalid_ts = df["timestamp"].isna().sum()
print(f"Invalid timestamps: {invalid_ts:,}")
if invalid_ts > 0:
    df = df.dropna(subset=["timestamp"])
    print(f"  Dropped rows with invalid timestamps. New len: {len(df):,}")

# --- 4. Validate amount ---
neg_amt = (df["amount"] < 0).sum()
zero_amt = (df["amount"] == 0).sum()
print(f"Negative amounts: {neg_amt:,}, Zero amounts: {zero_amt:,}")
if neg_amt > 0:
    df = df[df["amount"] >= 0]
    print(f"  Dropped negative amounts. New len: {len(df):,}")

# --- 5. Fix balance consistency ---
# For debits: balance_after should be balance_before - amount
# For credits: balance_after should be balance_before + amount
# We'll flag inconsistent rows rather than drop them
debit_mask = df["transaction_type"] == "debit"
credit_mask = df["transaction_type"] == "credit"

debit_expected = df.loc[debit_mask, "balance_before_ngn"] - df.loc[debit_mask, "amount"]
credit_expected = df.loc[credit_mask, "balance_before_ngn"] + df.loc[credit_mask, "amount"]

df["balance_consistent"] = True
df.loc[debit_mask, "balance_consistent"] = (
    (df.loc[debit_mask, "balance_after_ngn"] - debit_expected).abs() < 0.01
)
df.loc[credit_mask, "balance_consistent"] = (
    (df.loc[credit_mask, "balance_after_ngn"] - credit_expected).abs() < 0.01
)
inconsistent = (~df["balance_consistent"]).sum()
print(f"Balance inconsistent rows: {inconsistent:,} ({inconsistent/len(df)*100:.2f}%)")

# --- 6. Standardize categoricals ---
for col in ["transaction_type", "channel", "status", "location_state", "location_lga"]:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].replace("nan", np.nan)

print(f"\nTransaction types: {df['transaction_type'].unique()}")
print(f"Channels: {df['channel'].unique()}")
print(f"Statuses: {df['status'].unique()}")

# --- 7. Handle nulls in merchant fields ---
null_merchant_name = df["merchant_name"].isna().sum()
null_mcc = df["merchant_category_code"].isna().sum()
null_device = df["device_id"].isna().sum()
print(f"\nNull merchant_name: {null_merchant_name:,} ({null_merchant_name/len(df)*100:.1f}%)")
print(f"Null merchant_category_code: {null_mcc:,} ({null_mcc/len(df)*100:.1f}%)")
print(f"Null device_id: {null_device:,} ({null_device/len(df)*100:.1f}%)")

# Fill merchant nulls with 'unknown' — these are likely P2P or internal transfers
df["merchant_name"] = df["merchant_name"].fillna("unknown")
df["merchant_category_code"] = df["merchant_category_code"].fillna(-1)
df["merchant_category_code"] = df["merchant_category_code"].astype(int)

# device_id null likely means POS/ATM without device tracking — leave as NaN or fill
df["device_id"] = df["device_id"].fillna("unknown")

# --- 8. Extract time features ---
df["date"] = df["timestamp"].dt.date
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# --- 9. Amount outlier flagging (IQR method) ---
Q1 = df["amount"].quantile(0.25)
Q3 = df["amount"].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 3 * IQR  # use 3x for less aggressive flagging
df["amount_outlier"] = (df["amount"] > upper_fence).astype(int)
print(f"\nAmount outliers (>3x IQR, fence={upper_fence:,.0f}): {df['amount_outlier'].sum():,}")

# --- 10. Save ---
print(f"\nFinal shape: {df.shape}")
df.to_csv(OUT, index=False)
print(f"Saved to {OUT}")
print("Done.")
