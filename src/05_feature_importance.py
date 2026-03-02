"""Feature importance — LightGBM + SHAP analysis on customer features.

Target: predict next-quarter transaction frequency (recent_3m_count) from
behavioral features. This gives us feature importance in the context of
customer activity prediction — directly relevant to the task.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import shap

DATA = "output/customer_features.csv"
OUT = "output/"

print("Loading customer features...")
df = pd.read_csv(DATA)
print(f"Shape: {df.shape}")

# ============================================================
# Prepare data
# ============================================================
# Target: recent 3-month transaction count (proxy for future activity)
target_col = "recent_3m_count"

# Drop columns that leak the target or are identifiers
drop_cols = [
    "customer_id",
    "recent_3m_count", "recent_3m_amount",  # target & correlated
    "prior_3m_count", "prior_3m_amount",     # used to compute trend
    "count_trend", "amount_trend",           # derived from target
]

# Encode categoricals
cat_cols = ["primary_channel", "primary_state"]
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

y = df[target_col]
X = df.drop(columns=drop_cols, errors="ignore")

feature_names = X.columns.tolist()
print(f"\nFeatures: {len(feature_names)}")
print(f"Target: {target_col} (mean={y.mean():.1f}, std={y.std():.1f})")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

# ============================================================
# Train LightGBM
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
    num_boost_round=500,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nTest MAE: {mae:.2f}")
print(f"Test R²:  {r2:.4f}")

# ============================================================
# 1. Built-in feature importance (gain)
# ============================================================
print("\nExtracting feature importance (gain)...")
importance = pd.DataFrame({
    "feature": feature_names,
    "importance_gain": model.feature_importance(importance_type="gain"),
    "importance_split": model.feature_importance(importance_type="split"),
})
importance = importance.sort_values("importance_gain", ascending=False).reset_index(drop=True)

print("\nTop 20 features by gain:")
for i, row in importance.head(20).iterrows():
    print(f"  {i+1:2d}. {row['feature']:35s} gain={row['importance_gain']:12,.0f}  splits={row['importance_split']:,}")

# Plot: top 25 by gain
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

top_n = 25
top = importance.head(top_n)
axes[0].barh(range(top_n), top["importance_gain"].values, color="steelblue")
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top["feature"].values)
axes[0].invert_yaxis()
axes[0].set_title(f"Top {top_n} Features by Gain")
axes[0].set_xlabel("Gain")

axes[1].barh(range(top_n), top["importance_split"].values, color="coral")
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(top["feature"].values)
axes[1].invert_yaxis()
axes[1].set_title(f"Top {top_n} Features by Split Count")
axes[1].set_xlabel("Split Count")

plt.tight_layout()
plt.savefig(f"{OUT}11_feature_importance_lgbm.png", dpi=150)
plt.close()
print("Saved: 11_feature_importance_lgbm.png")

# ============================================================
# 2. SHAP analysis
# ============================================================
print("\nComputing SHAP values (on test set)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary (bar)
fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=25, show=False)
plt.title("SHAP Feature Importance (mean |SHAP|)")
plt.tight_layout()
plt.savefig(f"{OUT}12_shap_importance_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 12_shap_importance_bar.png")

# SHAP beeswarm
fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_values, X_test, max_display=25, show=False)
plt.title("SHAP Beeswarm Plot")
plt.tight_layout()
plt.savefig(f"{OUT}13_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 13_shap_beeswarm.png")

# SHAP mean absolute values — tabular
shap_abs_mean = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 20 features by mean |SHAP|:")
for i, row in shap_abs_mean.head(20).iterrows():
    print(f"  {i+1:2d}. {row['feature']:35s} mean|SHAP|={row['mean_abs_shap']:.3f}")

# ============================================================
# 3. Feature importance comparison table
# ============================================================
comparison = importance.merge(shap_abs_mean, on="feature")
comparison["rank_gain"] = comparison["importance_gain"].rank(ascending=False).astype(int)
comparison["rank_shap"] = comparison["mean_abs_shap"].rank(ascending=False).astype(int)
comparison = comparison.sort_values("rank_shap")
comparison.to_csv(f"{OUT}feature_importance_table.csv", index=False)
print(f"\nSaved: feature_importance_table.csv")

# ============================================================
# 4. Save summary
# ============================================================
print("\nWriting summary...")
with open(f"{OUT}feature_importance_summary.txt", "w") as f:
    f.write("FEATURE IMPORTANCE SUMMARY\n" + "=" * 60 + "\n\n")
    f.write(f"Model: LightGBM Regressor\n")
    f.write(f"Target: {target_col} (3-month transaction count)\n")
    f.write(f"Features used: {len(feature_names)}\n")
    f.write(f"Train size: {len(X_train):,}\n")
    f.write(f"Test size: {len(X_test):,}\n")
    f.write(f"Best iteration: {model.best_iteration}\n")
    f.write(f"Test MAE: {mae:.2f}\n")
    f.write(f"Test R²: {r2:.4f}\n\n")

    f.write("TOP 20 BY LGBM GAIN:\n")
    for i, row in importance.head(20).iterrows():
        f.write(f"  {i+1:2d}. {row['feature']:35s} gain={row['importance_gain']:12,.0f}\n")

    f.write("\nTOP 20 BY MEAN |SHAP|:\n")
    for i, row in shap_abs_mean.head(20).iterrows():
        f.write(f"  {i+1:2d}. {row['feature']:35s} mean|SHAP|={row['mean_abs_shap']:.4f}\n")

print("\nAll outputs saved to output/")
print("Done.")
