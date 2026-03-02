"""Predictive model — forecast next-month customer transaction count.

Approach:
  - Aggregate transactions to customer-month level
  - Build time series features: lags, rolling windows, seasonal indicators
  - Merge customer-level static features from feature engineering
  - Train LightGBM with proper temporal train/test split
  - Evaluate on the last 3 months (Oct–Dec 2024)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

DATA = "output/cleaned.csv"
CUST_FEATS = "output/customer_features.csv"
OUT = "output/"

# ============================================================
# 1. Build customer-month panel
# ============================================================
print("Loading cleaned data...")
df = pd.read_csv(DATA, usecols=[
    "customer_id", "timestamp", "amount", "status",
    "transaction_type", "channel", "is_weekend", "hour"
], parse_dates=["timestamp"])
print(f"Rows: {len(df):,}")

df_success = df[df["status"] == "success"].copy()
df_success["ym"] = df_success["timestamp"].dt.to_period("M")

print("\nAggregating to customer-month level...")
monthly = df_success.groupby(["customer_id", "ym"]).agg(
    txn_count=("amount", "count"),
    txn_amount=("amount", "sum"),
    txn_mean=("amount", "mean"),
    txn_max=("amount", "max"),
    debit_count=("transaction_type", lambda x: (x == "debit").sum()),
    credit_count=("transaction_type", lambda x: (x == "credit").sum()),
    weekend_count=("is_weekend", "sum"),
    mean_hour=("hour", "mean"),
    n_channels=("channel", "nunique"),
).reset_index()

# Create full customer x month panel (fill missing months with 0)
all_customers = monthly["customer_id"].unique()
all_months = sorted(monthly["ym"].unique())
print(f"Customers: {len(all_customers):,}, Months: {len(all_months)}")

idx = pd.MultiIndex.from_product([all_customers, all_months], names=["customer_id", "ym"])
panel = monthly.set_index(["customer_id", "ym"]).reindex(idx, fill_value=0).reset_index()
print(f"Full panel: {len(panel):,} rows ({len(all_customers):,} x {len(all_months)})")

# Also get all-status counts per customer-month (including failed)
all_status_monthly = df.groupby(["customer_id"]).apply(
    lambda g: g.assign(ym=g["timestamp"].dt.to_period("M"))
    .groupby("ym")["status"]
    .value_counts()
).unstack(fill_value=0)
# Simpler approach — vectorized
df["ym"] = df["timestamp"].dt.to_period("M")
status_monthly = df.groupby(["customer_id", "ym", "status"]).size().unstack(fill_value=0).reset_index()
status_monthly.columns.name = None
for col in ["failed", "pending", "reversed"]:
    if col not in status_monthly.columns:
        status_monthly[col] = 0
status_monthly = status_monthly.rename(columns={
    "failed": "failed_count", "pending": "pending_count",
    "reversed": "reversed_count", "success": "success_count_all"
})
status_monthly = status_monthly[["customer_id", "ym", "failed_count", "pending_count", "reversed_count"]]
panel = panel.merge(status_monthly, on=["customer_id", "ym"], how="left").fillna(0)

# ============================================================
# 2. Time series features (lag, rolling, seasonal)
# ============================================================
print("\nBuilding time series features...")
panel = panel.sort_values(["customer_id", "ym"]).reset_index(drop=True)

# Convert period to int for easier manipulation
panel["month_num"] = panel["ym"].apply(lambda x: x.month)
panel["year"] = panel["ym"].apply(lambda x: x.year)
panel["month_idx"] = (panel["year"] - 2023) * 12 + panel["month_num"]  # 1-24

# --- Lag features ---
for lag in [1, 2, 3, 6]:
    panel[f"lag_{lag}_count"] = panel.groupby("customer_id")["txn_count"].shift(lag)
    panel[f"lag_{lag}_amount"] = panel.groupby("customer_id")["txn_amount"].shift(lag)

# --- Rolling window features ---
for window in [3, 6]:
    rolling_count = panel.groupby("customer_id")["txn_count"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    panel[f"roll_{window}m_count_mean"] = rolling_count

    rolling_amount = panel.groupby("customer_id")["txn_amount"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    panel[f"roll_{window}m_amount_mean"] = rolling_amount

    rolling_std = panel.groupby("customer_id")["txn_count"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    )
    panel[f"roll_{window}m_count_std"] = rolling_std

# --- Trend features ---
panel["count_diff_1m"] = panel.groupby("customer_id")["txn_count"].diff(1)
panel["count_diff_3m"] = panel.groupby("customer_id")["txn_count"].diff(3)

# --- Months since last active (non-zero) ---
def months_since_active(series):
    result = pd.Series(index=series.index, dtype=float)
    last_active = None
    for i, (idx, val) in enumerate(series.items()):
        if i > 0 and last_active is not None:
            result.iloc[i] = i - last_active
        else:
            result.iloc[i] = 0
        if val > 0:
            last_active = i
    return result

# Vectorized approximation: cumsum trick
panel["was_active"] = (panel["txn_count"] > 0).astype(int)
panel["inactive_streak"] = panel.groupby("customer_id")["was_active"].transform(
    lambda x: x.groupby((x == 1).cumsum()).cumcount()
)

# --- Seasonal features ---
panel["sin_month"] = np.sin(2 * np.pi * panel["month_num"] / 12)
panel["cos_month"] = np.cos(2 * np.pi * panel["month_num"] / 12)
panel["is_q4"] = (panel["month_num"] >= 10).astype(int)
panel["is_q1"] = (panel["month_num"] <= 3).astype(int)

# --- Debit/credit ratio (lagged) ---
panel["debit_ratio_lag1"] = panel.groupby("customer_id").apply(
    lambda g: g["debit_count"].shift(1) / g["txn_count"].shift(1).clip(lower=1)
).values

# ============================================================
# 3. Merge static customer features
# ============================================================
print("Merging static customer features...")
cust_feats = pd.read_csv(CUST_FEATS)
# Select stable features (not time-dependent counts that would leak)
static_cols = [
    "customer_id",
    "monetary_mean", "monetary_std", "monetary_median",
    "avg_balance_before", "std_balance_before",
    "n_channels_used", "tenure_days",
    "weekend_ratio", "dow_entropy", "std_hour",
    "outlier_txn_ratio",
]
# Keep only columns that exist
static_cols = [c for c in static_cols if c in cust_feats.columns]
panel = panel.merge(cust_feats[static_cols], on="customer_id", how="left")

# ============================================================
# 4. Prepare modeling data
# ============================================================
print("\nPreparing train/test split...")

# Target: next month's transaction count
panel["target"] = panel.groupby("customer_id")["txn_count"].shift(-1)

# Drop rows where lags aren't available (first 6 months) or target is missing (last month)
panel_model = panel[
    (panel["month_idx"] >= 7) &  # need 6-month lags
    (panel["target"].notna())
].copy()

# Drop identifier and intermediate columns
drop_cols = ["customer_id", "ym", "target", "was_active", "year"]
feature_cols = [c for c in panel_model.columns if c not in drop_cols]
X = panel_model[feature_cols]
y = panel_model["target"]

print(f"Modeling data: {len(X):,} rows, {len(feature_cols)} features")
print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")

# Temporal split: train on months 7–21 (Jul 2023 – Sep 2024), test on 22–23 (Oct–Nov 2024)
# month_idx: 7=Jul2023 ... 21=Sep2024, 22=Oct2024, 23=Nov2024
# (We can't use Dec 2024 as target since there's no Jan 2025)
train_mask = panel_model["month_idx"] <= 21
test_mask = panel_model["month_idx"] > 21

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train: {len(X_train):,} rows (Jul 2023 – Sep 2024)")
print(f"Test:  {len(X_test):,} rows (Oct – Nov 2024)")

# ============================================================
# 5. Train LightGBM
# ============================================================
print("\nTraining LightGBM...")
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

# ============================================================
# 6. Evaluate
# ============================================================
y_pred = model.predict(X_test)
y_pred_clipped = np.clip(y_pred, 0, None)  # counts can't be negative

mae = mean_absolute_error(y_test, y_pred_clipped)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
r2 = r2_score(y_test, y_pred_clipped)
mape_safe = np.mean(np.abs(y_test - y_pred_clipped) / np.clip(y_test, 1, None)) * 100

print(f"\n{'='*50}")
print(f"TEST RESULTS (Oct–Nov 2024)")
print(f"{'='*50}")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.4f}")
print(f"MAPE (where count>=1): {mape_safe:.1f}%")
print(f"Best iteration: {model.best_iteration}")

# Baseline: predict last month's count
baseline_pred = X_test["lag_1_count"].values
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)
print(f"\nBaseline (lag-1 naive): MAE={baseline_mae:.3f}, R²={baseline_r2:.4f}")
print(f"Improvement over baseline: MAE {(1 - mae/baseline_mae)*100:.1f}%")

# ============================================================
# 7. Feature importance
# ============================================================
print("\nTop 20 features by gain:")
importance = pd.DataFrame({
    "feature": feature_cols,
    "gain": model.feature_importance(importance_type="gain"),
    "split": model.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)

for i, (_, row) in enumerate(importance.head(20).iterrows()):
    print(f"  {i+1:2d}. {row['feature']:30s} gain={row['gain']:14,.0f}")

importance.to_csv(f"{OUT}ts_model_feature_importance.csv", index=False)

# ============================================================
# 8. Plots
# ============================================================
print("\nGenerating plots...")

# 8a. Actual vs Predicted scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(y_test, y_pred_clipped, alpha=0.05, s=3, color="steelblue")
max_val = max(y_test.max(), y_pred_clipped.max())
ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Actual transaction count")
ax.set_ylabel("Predicted transaction count")
ax.set_title(f"Actual vs Predicted (R²={r2:.3f}, MAE={mae:.2f})")
ax.legend()
ax.set_xlim(0, min(max_val, 50))
ax.set_ylim(0, min(max_val, 50))

# 8b. Residual distribution
ax = axes[1]
residuals = y_test.values - y_pred_clipped
ax.hist(residuals, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Residual (actual - predicted)")
ax.set_ylabel("Count")
ax.set_title(f"Residual Distribution (mean={residuals.mean():.2f}, std={residuals.std():.2f})")

plt.tight_layout()
plt.savefig(f"{OUT}14_ts_actual_vs_predicted.png", dpi=150)
plt.close()
print("Saved: 14_ts_actual_vs_predicted.png")

# 8c. Feature importance plot
fig, ax = plt.subplots(figsize=(10, 8))
top_n = 20
top = importance.head(top_n)
ax.barh(range(top_n), top["gain"].values, color="steelblue")
ax.set_yticks(range(top_n))
ax.set_yticklabels(top["feature"].values)
ax.invert_yaxis()
ax.set_title("Time Series Model — Top 20 Features by Gain")
ax.set_xlabel("Gain")
plt.tight_layout()
plt.savefig(f"{OUT}15_ts_feature_importance.png", dpi=150)
plt.close()
print("Saved: 15_ts_feature_importance.png")

# 8d. Monthly aggregate: actual vs predicted
panel_test = panel_model[test_mask].copy()
panel_test["predicted"] = y_pred_clipped
monthly_agg = panel_test.groupby("month_idx").agg(
    actual_mean=("target", "mean"),
    predicted_mean=("predicted", "mean"),
    actual_total=("target", "sum"),
    predicted_total=("predicted", "sum"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.bar(monthly_agg["month_idx"] - 0.15, monthly_agg["actual_mean"], width=0.3, label="Actual", color="steelblue")
ax.bar(monthly_agg["month_idx"] + 0.15, monthly_agg["predicted_mean"], width=0.3, label="Predicted", color="coral")
ax.set_xlabel("Month")
ax.set_ylabel("Mean transaction count per customer")
ax.set_title("Monthly Mean: Actual vs Predicted")
ax.legend()
ax.set_xticks(monthly_agg["month_idx"])
ax.set_xticklabels(["Oct 2024", "Nov 2024"][:len(monthly_agg)])

ax = axes[1]
ax.bar(monthly_agg["month_idx"] - 0.15, monthly_agg["actual_total"], width=0.3, label="Actual", color="steelblue")
ax.bar(monthly_agg["month_idx"] + 0.15, monthly_agg["predicted_total"], width=0.3, label="Predicted", color="coral")
ax.set_xlabel("Month")
ax.set_ylabel("Total transactions")
ax.set_title("Monthly Total: Actual vs Predicted")
ax.legend()
ax.set_xticks(monthly_agg["month_idx"])
ax.set_xticklabels(["Oct 2024", "Nov 2024"][:len(monthly_agg)])

plt.tight_layout()
plt.savefig(f"{OUT}16_ts_monthly_comparison.png", dpi=150)
plt.close()
print("Saved: 16_ts_monthly_comparison.png")

# 8e. Prediction by activity segment
panel_test["activity_segment"] = pd.cut(
    panel_test["lag_1_count"],
    bins=[-1, 0, 2, 5, 10, 1000],
    labels=["Inactive (0)", "Low (1-2)", "Medium (3-5)", "High (6-10)", "Very High (>10)"]
)
seg_perf = panel_test.groupby("activity_segment", observed=True).apply(
    lambda g: pd.Series({
        "count": len(g),
        "actual_mean": g["target"].mean(),
        "predicted_mean": g["predicted"].mean(),
        "mae": mean_absolute_error(g["target"], g["predicted"]),
    })
).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(seg_perf))
ax.bar([i - 0.15 for i in x], seg_perf["actual_mean"], width=0.3, label="Actual", color="steelblue")
ax.bar([i + 0.15 for i in x], seg_perf["predicted_mean"], width=0.3, label="Predicted", color="coral")
ax.set_xticks(x)
ax.set_xticklabels(seg_perf["activity_segment"], rotation=15)
ax.set_ylabel("Mean transaction count (next month)")
ax.set_title("Prediction Accuracy by Customer Activity Segment")
ax.legend()
for i, row in seg_perf.iterrows():
    ax.annotate(f"MAE={row['mae']:.2f}\nn={int(row['count']):,}",
                xy=(i, max(row["actual_mean"], row["predicted_mean"])),
                ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}17_ts_segment_performance.png", dpi=150)
plt.close()
print("Saved: 17_ts_segment_performance.png")

# ============================================================
# 9. Save summary
# ============================================================
print("\nWriting summary...")
with open(f"{OUT}ts_model_summary.txt", "w") as f:
    f.write("TIME SERIES PREDICTIVE MODEL SUMMARY\n" + "=" * 60 + "\n\n")
    f.write("Task: Predict next-month transaction count per customer\n")
    f.write("Granularity: customer x month\n")
    f.write(f"Model: LightGBM (gradient boosted trees)\n")
    f.write(f"Features: {len(feature_cols)} (lags, rolling, seasonal, static customer)\n")
    f.write(f"Train: months 7–21 (Jul 2023 – Sep 2024), {len(X_train):,} rows\n")
    f.write(f"Test:  months 22–23 (Oct – Nov 2024), {len(X_test):,} rows\n")
    f.write(f"Best iteration: {model.best_iteration}\n\n")

    f.write("PERFORMANCE:\n")
    f.write(f"  MAE:  {mae:.3f}\n")
    f.write(f"  RMSE: {rmse:.3f}\n")
    f.write(f"  R²:   {r2:.4f}\n")
    f.write(f"  MAPE: {mape_safe:.1f}%\n\n")

    f.write(f"BASELINE (lag-1 naive):\n")
    f.write(f"  MAE:  {baseline_mae:.3f}\n")
    f.write(f"  R²:   {baseline_r2:.4f}\n")
    f.write(f"  Model improvement: {(1 - mae/baseline_mae)*100:.1f}% MAE reduction\n\n")

    f.write("TOP 20 FEATURES BY GAIN:\n")
    for i, (_, row) in enumerate(importance.head(20).iterrows()):
        f.write(f"  {i+1:2d}. {row['feature']:30s} gain={row['gain']:14,.0f}\n")

    f.write("\nPERFORMANCE BY ACTIVITY SEGMENT:\n")
    for _, row in seg_perf.iterrows():
        f.write(f"  {row['activity_segment']:20s} n={int(row['count']):>7,}  "
                f"actual={row['actual_mean']:.2f}  pred={row['predicted_mean']:.2f}  "
                f"MAE={row['mae']:.2f}\n")

print("\nAll outputs saved to output/")
print("Done.")
