"""Feature engineering — builds customer-level features from cleaned transaction data."""

import pandas as pd
import numpy as np

DATA = "output/cleaned.csv"
OUT = "output/customer_features.csv"

print("Loading cleaned data...")
df = pd.read_csv(DATA, parse_dates=["timestamp"])
print(f"Shape: {df.shape}")

# Use only successful transactions for behavioral features
df_success = df[df["status"] == "success"].copy()
print(f"Successful transactions: {len(df_success):,}")

# ============================================================
# 1. RFM Features (Recency, Frequency, Monetary)
# ============================================================
print("\n1. RFM features...")
ref_date = df["timestamp"].max()

rfm = df_success.groupby("customer_id").agg(
    recency_days=("timestamp", lambda x: (ref_date - x.max()).days),
    frequency=("transaction_id", "count"),
    monetary_total=("amount", "sum"),
    monetary_mean=("amount", "mean"),
    monetary_median=("amount", "median"),
    monetary_std=("amount", "std"),
    monetary_max=("amount", "max"),
    monetary_min=("amount", "min"),
).reset_index()
rfm["monetary_std"] = rfm["monetary_std"].fillna(0)

print(f"  RFM shape: {rfm.shape}")

# ============================================================
# 2. Transaction type features
# ============================================================
print("2. Transaction type features...")
type_counts = df_success.groupby(["customer_id", "transaction_type"]).size().unstack(fill_value=0)
type_counts.columns = [f"txn_type_{c}_count" for c in type_counts.columns]
type_total = type_counts.sum(axis=1)
type_ratios = type_counts.div(type_total, axis=0)
type_ratios.columns = [c.replace("_count", "_ratio") for c in type_ratios.columns]
type_feats = pd.concat([type_counts, type_ratios], axis=1).reset_index()

# ============================================================
# 3. Channel usage features
# ============================================================
print("3. Channel usage features...")
ch_counts = df_success.groupby(["customer_id", "channel"]).size().unstack(fill_value=0)
ch_counts.columns = [f"channel_{c}_count" for c in ch_counts.columns]
ch_total = ch_counts.sum(axis=1)
ch_ratios = ch_counts.div(ch_total, axis=0)
ch_ratios.columns = [c.replace("_count", "_ratio") for c in ch_ratios.columns]
# Primary channel
ch_primary = ch_counts.idxmax(axis=1).str.replace("channel_", "").str.replace("_count", "")
ch_feats = pd.concat([ch_counts, ch_ratios], axis=1).reset_index()
ch_feats["primary_channel"] = ch_primary.values
# Number of distinct channels used
ch_feats["n_channels_used"] = (ch_counts > 0).sum(axis=1).values

# ============================================================
# 4. Temporal behavior features
# ============================================================
print("4. Temporal behavior features...")
temp = df_success.groupby("customer_id").agg(
    first_txn_date=("timestamp", "min"),
    last_txn_date=("timestamp", "max"),
    weekend_txn_count=("is_weekend", "sum"),
    total_txn_count=("transaction_id", "count"),
    mean_hour=("hour", "mean"),
    std_hour=("hour", "std"),
    n_active_days=("date", "nunique"),
    n_active_months=("month", lambda x: x.nunique()),
).reset_index()

temp["tenure_days"] = (temp["last_txn_date"] - temp["first_txn_date"]).dt.days
temp["weekend_ratio"] = temp["weekend_txn_count"] / temp["total_txn_count"]
temp["txn_per_active_day"] = temp["total_txn_count"] / temp["n_active_days"].clip(lower=1)
temp["std_hour"] = temp["std_hour"].fillna(0)
temp = temp.drop(columns=["first_txn_date", "last_txn_date", "weekend_txn_count", "total_txn_count"])

# Peak hour (mode)
peak_hour = df_success.groupby("customer_id")["hour"].agg(lambda x: x.mode().iloc[0]).rename("peak_hour")
temp = temp.merge(peak_hour, on="customer_id")

# Day-of-week concentration (entropy)
dow_counts = df_success.groupby(["customer_id", "day_of_week"]).size().unstack(fill_value=0)
dow_probs = dow_counts.div(dow_counts.sum(axis=1), axis=0)
dow_entropy = -(dow_probs * np.log2(dow_probs.clip(lower=1e-10))).sum(axis=1)
temp["dow_entropy"] = dow_entropy.values  # higher = more evenly spread across days

# ============================================================
# 5. Status-based features (failure rates etc.)
# ============================================================
print("5. Status-based features...")
status_counts = df.groupby(["customer_id", "status"]).size().unstack(fill_value=0)
total_per_cust = status_counts.sum(axis=1)
status_feats = pd.DataFrame({"customer_id": status_counts.index})
status_feats["total_txn_all_status"] = total_per_cust.values
for col in status_counts.columns:
    status_feats[f"status_{col}_count"] = status_counts[col].values
    status_feats[f"status_{col}_rate"] = (status_counts[col] / total_per_cust).values

# ============================================================
# 6. Balance features
# ============================================================
print("6. Balance features...")
bal = df_success.groupby("customer_id").agg(
    avg_balance_before=("balance_before_ngn", "mean"),
    max_balance_before=("balance_before_ngn", "max"),
    min_balance_before=("balance_before_ngn", "min"),
    std_balance_before=("balance_before_ngn", "std"),
    balance_inconsistent_count=("balance_consistent", lambda x: (~x).sum()),
).reset_index()
bal["std_balance_before"] = bal["std_balance_before"].fillna(0)

# ============================================================
# 7. Geographic features
# ============================================================
print("7. Geographic features...")
geo = df_success.groupby("customer_id").agg(
    primary_state=("location_state", lambda x: x.mode().iloc[0]),
    n_unique_states=("location_state", "nunique"),
    n_unique_lgas=("location_lga", "nunique"),
).reset_index()

# ============================================================
# 8. Merchant features
# ============================================================
print("8. Merchant features...")
df_merch = df_success[df_success["merchant_name"] != "unknown"]
merch = df_merch.groupby("customer_id").agg(
    n_unique_merchants=("merchant_name", "nunique"),
    n_unique_mcc=("merchant_category_code", "nunique"),
).reset_index()

# ============================================================
# 9. Rolling / lag-inspired features (monthly aggregates)
# ============================================================
print("9. Monthly trend features...")
df_success["ym"] = df_success["timestamp"].dt.to_period("M")
monthly_cust = df_success.groupby(["customer_id", "ym"]).agg(
    monthly_count=("transaction_id", "count"),
    monthly_amount=("amount", "sum"),
).reset_index()

# Trend: compare last 3 months vs prior 3 months
all_periods = monthly_cust["ym"].unique()
all_periods_sorted = sorted(all_periods)
last_3 = all_periods_sorted[-3:]
prior_3 = all_periods_sorted[-6:-3]

recent = monthly_cust[monthly_cust["ym"].isin(last_3)].groupby("customer_id").agg(
    recent_3m_count=("monthly_count", "sum"),
    recent_3m_amount=("monthly_amount", "sum"),
).reset_index()

prior = monthly_cust[monthly_cust["ym"].isin(prior_3)].groupby("customer_id").agg(
    prior_3m_count=("monthly_count", "sum"),
    prior_3m_amount=("monthly_amount", "sum"),
).reset_index()

trend = recent.merge(prior, on="customer_id", how="outer").fillna(0)
trend["count_trend"] = (trend["recent_3m_count"] - trend["prior_3m_count"]) / trend["prior_3m_count"].clip(lower=1)
trend["amount_trend"] = (trend["recent_3m_amount"] - trend["prior_3m_amount"]) / trend["prior_3m_amount"].clip(lower=1)

# Monthly volatility
monthly_vol = monthly_cust.groupby("customer_id").agg(
    monthly_count_std=("monthly_count", "std"),
    monthly_amount_std=("monthly_amount", "std"),
    monthly_count_mean=("monthly_count", "mean"),
    monthly_amount_mean=("monthly_amount", "mean"),
).reset_index()
monthly_vol["monthly_count_cv"] = monthly_vol["monthly_count_std"] / monthly_vol["monthly_count_mean"].clip(lower=1)
monthly_vol["monthly_amount_cv"] = monthly_vol["monthly_amount_std"] / monthly_vol["monthly_amount_mean"].clip(lower=1)
monthly_vol = monthly_vol.fillna(0)

# ============================================================
# 10. Amount outlier features
# ============================================================
print("10. Outlier features...")
outlier_feats = df_success.groupby("customer_id").agg(
    outlier_txn_count=("amount_outlier", "sum"),
    outlier_txn_ratio=("amount_outlier", "mean"),
).reset_index()

# ============================================================
# MERGE ALL
# ============================================================
print("\nMerging all features...")
features = rfm
for feat_df in [type_feats, ch_feats, temp, status_feats, bal, geo, merch, trend, monthly_vol, outlier_feats]:
    features = features.merge(feat_df, on="customer_id", how="left")

features = features.fillna(0)

print(f"\nFinal feature matrix: {features.shape}")
print(f"Customers: {features['customer_id'].nunique():,}")
print(f"Features: {features.shape[1] - 1}")  # minus customer_id

# Show feature names
print(f"\nFeature columns ({features.shape[1] - 1}):")
for i, col in enumerate(features.columns[1:], 1):
    print(f"  {i:2d}. {col}")

features.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")
print("Done.")
