"""
07 — Customer Segmentation Model
K-Means clustering on customer behavioral features.
Uses top features from importance analysis + domain-relevant dimensions.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

OUT = "output"
df = pd.read_csv(f"{OUT}/customer_features.csv")
print(f"Loaded {len(df):,} customers, {df.shape[1]} columns")

# ── 1. Feature selection for segmentation ──────────────────────────
seg_features = [
    # Activity
    "frequency", "recency_days", "n_active_days", "n_active_months",
    "monthly_count_mean", "monthly_count_cv",
    # Monetary
    "monetary_mean", "monetary_median", "monetary_std",
    "avg_balance_before",
    # Channel mix (ratios)
    "channel_mobile_ratio", "channel_agent_ratio", "channel_web_ratio",
    "channel_atm_ratio", "channel_pos_ratio", "channel_ussd_ratio",
    "n_channels_used",
    # Temporal behavior
    "weekend_ratio", "std_hour", "dow_entropy",
    # Transaction quality
    "status_failed_rate", "status_reversed_rate", "status_pending_rate",
    # Trend
    "count_trend",
    # Outlier behavior
    "outlier_txn_ratio",
]
seg_features = list(dict.fromkeys(seg_features))
print(f"Segmentation features: {len(seg_features)}")

X_raw = df[seg_features].copy()

# Log-transform heavily skewed features
skewed = ["frequency", "n_active_days", "monthly_count_mean",
          "monetary_mean", "monetary_median", "monetary_std",
          "avg_balance_before"]
for col in skewed:
    X_raw[col] = np.log1p(X_raw[col])

# ── 2. Scaling ─────────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print(f"Scaled matrix: {X.shape}")

# ── 3. Optimal K search ───────────────────────────────────────────
K_range = range(2, 11)
inertias, silhouettes, chs, dbs = [], [], [], []

print("\nSearching K=2..10:")
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil = silhouette_score(X, labels, sample_size=min(20000, len(X)), random_state=42)
    silhouettes.append(sil)
    ch = calinski_harabasz_score(X, labels)
    chs.append(ch)
    db = davies_bouldin_score(X, labels)
    dbs.append(db)
    print(f"  K={k:2d}  silhouette={sil:.4f}  CH={ch:,.0f}  DB={db:.4f}  inertia={km.inertia_:,.0f}")

# ── 4. Plot elbow / silhouette ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(list(K_range), inertias, "o-", color="#2196F3")
axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia"); axes[0].set_title("Elbow Method")
axes[1].plot(list(K_range), silhouettes, "o-", color="#4CAF50")
axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score"); axes[1].set_title("Silhouette Score")
axes[2].plot(list(K_range), dbs, "o-", color="#FF5722")
axes[2].set_xlabel("K"); axes[2].set_ylabel("Davies-Bouldin Index"); axes[2].set_title("Davies-Bouldin (lower = better)")
for ax in axes:
    ax.set_xticks(list(K_range))
plt.tight_layout()
plt.savefig(f"{OUT}/18_seg_elbow_silhouette.png", dpi=150)
plt.close()
print("\nSaved: 18_seg_elbow_silhouette.png")

# ── 5. Select K=5 for business interpretability ───────────────────
# Silhouette scores are low across all K (common with high-dimensional
# customer behavioral data that forms a continuum rather than distinct blobs).
# K=2 maximizes silhouette but is too coarse. K=5 provides the best
# balance of silhouette (second-best) and business granularity —
# enough segments to capture distinct behavioral archetypes.
best_k = 5
print(f"\nChosen K: {best_k} (business interpretability + second-best silhouette)")

final_km = KMeans(n_clusters=best_k, n_init=20, max_iter=500, random_state=42)
df["segment"] = final_km.fit_predict(X)

sil_final = silhouette_score(X, df["segment"], sample_size=min(20000, len(X)), random_state=42)
ch_final = calinski_harabasz_score(X, df["segment"])
db_final = davies_bouldin_score(X, df["segment"])
print(f"Final model — K={best_k}, silhouette={sil_final:.4f}, CH={ch_final:,.0f}, DB={db_final:.4f}")

# ── 6. Profile segments and auto-label ─────────────────────────────
key_cols = ["frequency", "recency_days", "monetary_mean", "n_active_days",
            "monthly_count_mean", "n_channels_used", "count_trend",
            "status_failed_rate", "channel_mobile_ratio", "channel_agent_ratio",
            "avg_balance_before", "weekend_ratio", "outlier_txn_ratio"]
summary = df.groupby("segment")[key_cols].mean()

# Sort segments by activity level (frequency) for consistent labeling
seg_order = summary["frequency"].sort_values().index.tolist()
# Remap so segment 0 = lowest activity, segment 4 = highest
remap = {old: new for new, old in enumerate(seg_order)}
df["segment"] = df["segment"].map(remap)

# Refit metrics with remapped labels
seg_sizes = df["segment"].value_counts().sort_index()
summary = df.groupby("segment")[key_cols].mean()

# Auto-label based on profiles — rank each segment across key dimensions
labels = {}
freq_rank = summary["frequency"].rank()
monetary_rank = summary["monetary_mean"].rank()
recency_rank = summary["recency_days"].rank(ascending=False)  # lower recency = better

for seg in range(best_k):
    freq = summary.loc[seg, "frequency"]
    recency = summary.loc[seg, "recency_days"]
    monetary = summary.loc[seg, "monetary_mean"]
    fr = freq_rank[seg]
    mr = monetary_rank[seg]

    if fr == best_k:  # highest frequency
        labels[seg] = "Power Users"
    elif recency > 80:
        labels[seg] = "Dormant"
    elif mr >= best_k - 1 and fr <= 2:
        labels[seg] = "High-Value Infrequent"
    elif fr >= best_k - 1:
        labels[seg] = "Active Core"
    elif fr >= 3:
        labels[seg] = "Engaged Regular"
    elif mr <= 2:
        labels[seg] = "Budget Casual"
    else:
        labels[seg] = "Moderate"

df["segment_label"] = df["segment"].map(labels)

print(f"\nSegment sizes:")
for seg in range(best_k):
    cnt = seg_sizes[seg]
    print(f"  Segment {seg} ({labels[seg]:>20s}): {cnt:>7,} ({cnt/len(df):.1%})")

print("\nSegment Profiles (means):")
print(summary.round(2).to_string())

# ── 7. PCA visualization ──────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
var_explained = pca.explained_variance_ratio_

# Recompute centroids in PCA space (need to re-transform since segments remapped)
fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.Set1(np.linspace(0, 1, best_k))
for seg in range(best_k):
    mask = df["segment"] == seg
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[seg]], alpha=0.15, s=5,
               edgecolors="none", label=f"Seg {seg}: {labels[seg]} (n={seg_sizes[seg]:,})")
    cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
    ax.scatter(cx, cy, c="black", marker="X", s=200, edgecolors="white", linewidth=2, zorder=5)
    ax.annotate(f"{labels[seg]}", (cx, cy), fontsize=9, fontweight="bold",
                ha="center", va="bottom", xytext=(0, 12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
ax.set_title(f"Customer Segments (K={best_k}) — PCA Projection")
ax.legend(loc="upper right", fontsize=8, markerscale=5)
plt.tight_layout()
plt.savefig(f"{OUT}/19_seg_pca_clusters.png", dpi=150)
plt.close()
print("\nSaved: 19_seg_pca_clusters.png")

# ── 8. Heatmap of segment profiles ────────────────────────────────
norm_cols = ["frequency", "recency_days", "monetary_mean", "n_active_days",
             "monthly_count_mean", "n_channels_used", "weekend_ratio",
             "dow_entropy", "status_failed_rate", "status_reversed_rate",
             "count_trend", "outlier_txn_ratio", "channel_mobile_ratio",
             "channel_agent_ratio", "avg_balance_before"]

seg_means = df.groupby("segment")[norm_cols].mean()
seg_norm = (seg_means - seg_means.min()) / (seg_means.max() - seg_means.min() + 1e-10)

fig, ax = plt.subplots(figsize=(15, 5))
sns.heatmap(seg_norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
            xticklabels=[c.replace("_", "\n") for c in norm_cols],
            yticklabels=[f"Seg {i}: {labels[i]}\n(n={seg_sizes[i]:,})" for i in range(best_k)])
ax.set_title("Segment Profiles (min-max normalized)")
plt.tight_layout()
plt.savefig(f"{OUT}/20_seg_profile_heatmap.png", dpi=150)
plt.close()
print("Saved: 20_seg_profile_heatmap.png")

# ── 9. Segment distribution plots ─────────────────────────────────
plot_feats = ["frequency", "monetary_mean", "recency_days", "monthly_count_mean",
              "n_channels_used", "count_trend"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, feat in zip(axes.flat, plot_feats):
    for seg in range(best_k):
        data = df.loc[df["segment"] == seg, feat]
        upper = data.quantile(0.98)
        data_clipped = data.clip(upper=upper)
        ax.hist(data_clipped, bins=50, alpha=0.4, label=f"{labels[seg]}", density=True)
    ax.set_title(feat.replace("_", " ").title())
    ax.legend(fontsize=7)
plt.suptitle("Feature Distributions by Segment", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/21_seg_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 21_seg_distributions.png")

# ── 10. Channel mix by segment ────────────────────────────────────
ch_cols = ["channel_mobile_ratio", "channel_agent_ratio", "channel_web_ratio",
           "channel_atm_ratio", "channel_pos_ratio", "channel_ussd_ratio"]
ch_means = df.groupby("segment")[ch_cols].mean()
ch_means.columns = [c.replace("channel_", "").replace("_ratio", "").title() for c in ch_cols]

fig, ax = plt.subplots(figsize=(10, 5))
ch_means.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
ax.set_xlabel("Segment"); ax.set_ylabel("Avg Channel Ratio")
ax.set_title("Channel Mix by Segment")
ax.set_xticklabels([f"{labels[i]}\n(n={seg_sizes[i]:,})" for i in range(best_k)], rotation=0, fontsize=8)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/22_seg_channel_mix.png", dpi=150)
plt.close()
print("Saved: 22_seg_channel_mix.png")

# ── 11. Segment transition matrix (recent vs prior trend) ─────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Box plot of count_trend by segment
bp_data = [df.loc[df["segment"] == s, "count_trend"].clip(-5, 5) for s in range(best_k)]
bp = axes[0].boxplot(bp_data, labels=[labels[s] for s in range(best_k)], patch_artist=True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
axes[0].set_ylabel("Count Trend (recent / prior 3m)")
axes[0].set_title("Activity Trend by Segment")
axes[0].axhline(y=1, color="gray", linestyle="--", alpha=0.5)
axes[0].tick_params(axis="x", rotation=15)

# Segment size pie chart
axes[1].pie(seg_sizes.values, labels=[f"{labels[i]}\n{seg_sizes[i]:,}" for i in range(best_k)],
            colors=colors, autopct="%.1f%%", startangle=90)
axes[1].set_title("Segment Distribution")
plt.tight_layout()
plt.savefig(f"{OUT}/23_seg_trend_and_size.png", dpi=150)
plt.close()
print("Saved: 23_seg_trend_and_size.png")

# ── 12. Save outputs ──────────────────────────────────────────────
df[["customer_id", "segment", "segment_label"]].to_csv(f"{OUT}/customer_segments.csv", index=False)
print(f"\nSaved: customer_segments.csv ({len(df):,} rows)")

# Full profile
profile_cols = [
    "frequency", "recency_days", "monetary_mean", "monetary_median",
    "avg_balance_before", "n_active_days", "n_active_months",
    "n_channels_used", "monthly_count_mean",
    "channel_mobile_ratio", "channel_agent_ratio", "channel_web_ratio",
    "channel_atm_ratio", "channel_pos_ratio",
    "weekend_ratio", "dow_entropy",
    "status_failed_rate", "status_reversed_rate", "status_pending_rate",
    "count_trend", "outlier_txn_ratio", "monthly_count_cv",
]
profile = df.groupby("segment")[profile_cols].agg(["mean", "median"])
profile.columns = [f"{c}_{s}" for c, s in profile.columns]
profile.to_csv(f"{OUT}/segment_profiles.csv")

# Summary text
with open(f"{OUT}/seg_model_summary.txt", "w") as f:
    f.write("CUSTOMER SEGMENTATION MODEL SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Algorithm: K-Means\n")
    f.write(f"Features used: {len(seg_features)}\n")
    f.write(f"Scaling: StandardScaler (log-transform on skewed features)\n")
    f.write(f"Chosen K: {best_k}\n\n")
    f.write("CLUSTER QUALITY METRICS:\n")
    f.write(f"  Silhouette Score:      {sil_final:.4f}\n")
    f.write(f"  Calinski-Harabasz:     {ch_final:,.0f}\n")
    f.write(f"  Davies-Bouldin:        {db_final:.4f}\n\n")
    f.write("K SEARCH RESULTS:\n")
    for i, k in enumerate(K_range):
        marker = " <--" if k == best_k else ""
        f.write(f"  K={k:2d}  sil={silhouettes[i]:.4f}  CH={chs[i]:>10,.0f}  DB={dbs[i]:.4f}{marker}\n")
    f.write(f"\nSEGMENT SIZES:\n")
    for seg in range(best_k):
        cnt = seg_sizes[seg]
        f.write(f"  Segment {seg} ({labels[seg]:>20s}): {cnt:>7,} ({cnt/len(df):.1%})\n")
    f.write(f"\nSEGMENT PROFILES (means):\n")
    f.write(summary.round(2).to_string() + "\n")
    f.write(f"\nSEGMENT LABELS:\n")
    for seg in range(best_k):
        f.write(f"  {seg}: {labels[seg]}\n")
    f.write(f"\nPCA VARIANCE EXPLAINED:\n")
    f.write(f"  PC1: {var_explained[0]:.1%}\n")
    f.write(f"  PC2: {var_explained[1]:.1%}\n")
    f.write(f"  Total (2 components): {sum(var_explained):.1%}\n")

print("Saved: seg_model_summary.txt")
print("\nDone!")
