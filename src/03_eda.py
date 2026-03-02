"""EDA — reads cleaned data, produces summary stats and plots in output/"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATA = "output/cleaned.csv"
OUT = "output/"

print("Loading cleaned data...")
df = pd.read_csv(DATA, parse_dates=["timestamp"])
print(f"Shape: {df.shape}")

sns.set_theme(style="whitegrid")
FIGSIZE = (12, 6)

# ---------- 1. Transaction volume over time (monthly) ----------
print("Plot 1: Monthly transaction volume...")
monthly = df.groupby(df["timestamp"].dt.to_period("M")).agg(
    txn_count=("transaction_id", "count"),
    total_amount=("amount", "sum"),
    avg_amount=("amount", "mean"),
).reset_index()
monthly["timestamp"] = monthly["timestamp"].dt.to_timestamp()

fig, ax1 = plt.subplots(figsize=FIGSIZE)
ax1.bar(monthly["timestamp"], monthly["txn_count"], width=20, alpha=0.7, label="Txn Count", color="steelblue")
ax1.set_ylabel("Transaction Count", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax2 = ax1.twinx()
ax2.plot(monthly["timestamp"], monthly["total_amount"] / 1e9, color="coral", marker="o", label="Total Amount (B)")
ax2.set_ylabel("Total Amount (Billions NGN)", color="coral")
ax2.tick_params(axis="y", labelcolor="coral")
plt.title("Monthly Transaction Volume & Amount")
fig.tight_layout()
plt.savefig(f"{OUT}01_monthly_volume.png", dpi=150)
plt.close()

# ---------- 2. Transaction type distribution ----------
print("Plot 2: Transaction type distribution...")
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
df["transaction_type"].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[0], colors=["#4C72B0", "#DD8452"])
axes[0].set_title("Transaction Type")
axes[0].set_ylabel("")
df["status"].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[1], colors=["#55A868", "#C44E52", "#8172B2"])
axes[1].set_title("Transaction Status")
axes[1].set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}02_type_status_dist.png", dpi=150)
plt.close()

# ---------- 3. Channel distribution ----------
print("Plot 3: Channel distribution...")
fig, ax = plt.subplots(figsize=(10, 5))
ch_counts = df["channel"].value_counts()
ch_counts.plot.barh(ax=ax, color="steelblue")
for i, (val, cnt) in enumerate(ch_counts.items()):
    ax.text(cnt + len(df)*0.005, i, f"{cnt:,} ({cnt/len(df)*100:.1f}%)", va="center")
ax.set_title("Transactions by Channel")
ax.set_xlabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT}03_channel_dist.png", dpi=150)
plt.close()

# ---------- 4. Amount distribution ----------
print("Plot 4: Amount distribution...")
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
# Log-scale histogram
df["amount"].clip(upper=df["amount"].quantile(0.99)).hist(bins=50, ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("Amount Distribution (clipped at 99th pct)")
axes[0].set_xlabel("Amount (NGN)")
axes[0].set_ylabel("Count")
# Log scale
df["amount"][df["amount"] > 0].apply(np.log10).hist(bins=50, ax=axes[1], color="coral", edgecolor="white")
axes[1].set_title("Log10(Amount) Distribution")
axes[1].set_xlabel("Log10(Amount)")
plt.tight_layout()
plt.savefig(f"{OUT}04_amount_dist.png", dpi=150)
plt.close()

# ---------- 5. Hourly pattern ----------
print("Plot 5: Hourly transaction pattern...")
fig, ax = plt.subplots(figsize=FIGSIZE)
hourly = df.groupby("hour").agg(count=("transaction_id", "count"), avg_amt=("amount", "mean")).reset_index()
ax.bar(hourly["hour"], hourly["count"], color="steelblue", alpha=0.7, label="Count")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Transaction Count", color="steelblue")
ax2 = ax.twinx()
ax2.plot(hourly["hour"], hourly["avg_amt"], color="coral", marker="o", label="Avg Amount")
ax2.set_ylabel("Avg Amount (NGN)", color="coral")
ax.set_title("Transaction Pattern by Hour of Day")
ax.set_xticks(range(24))
plt.tight_layout()
plt.savefig(f"{OUT}05_hourly_pattern.png", dpi=150)
plt.close()

# ---------- 6. Day-of-week pattern ----------
print("Plot 6: Day-of-week pattern...")
fig, ax = plt.subplots(figsize=(10, 5))
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow = df.groupby("day_of_week").agg(count=("transaction_id", "count")).reset_index()
ax.bar(dow["day_of_week"], dow["count"], color="steelblue")
ax.set_xticks(range(7))
ax.set_xticklabels(dow_labels)
ax.set_title("Transactions by Day of Week")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT}06_dow_pattern.png", dpi=150)
plt.close()

# ---------- 7. Top 15 states ----------
print("Plot 7: Top states...")
fig, ax = plt.subplots(figsize=(10, 6))
top_states = df["location_state"].value_counts().head(15)
top_states.plot.barh(ax=ax, color="steelblue")
ax.set_title("Top 15 States by Transaction Count")
ax.set_xlabel("Count")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}07_top_states.png", dpi=150)
plt.close()

# ---------- 8. Top 15 merchants ----------
print("Plot 8: Top merchants...")
fig, ax = plt.subplots(figsize=(10, 6))
top_merch = df[df["merchant_name"] != "unknown"]["merchant_name"].value_counts().head(15)
top_merch.plot.barh(ax=ax, color="coral")
ax.set_title("Top 15 Merchants by Transaction Count")
ax.set_xlabel("Count")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}08_top_merchants.png", dpi=150)
plt.close()

# ---------- 9. Customer activity distribution ----------
print("Plot 9: Customer activity distribution...")
cust_txn = df.groupby("customer_id").size()
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
cust_txn.clip(upper=cust_txn.quantile(0.99)).hist(bins=50, ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("Transactions per Customer (clipped 99th pct)")
axes[0].set_xlabel("Number of Transactions")

cust_amt = df.groupby("customer_id")["amount"].sum()
cust_amt.clip(upper=cust_amt.quantile(0.99)).hist(bins=50, ax=axes[1], color="coral", edgecolor="white")
axes[1].set_title("Total Spend per Customer (clipped 99th pct)")
axes[1].set_xlabel("Total Amount (NGN)")
plt.tight_layout()
plt.savefig(f"{OUT}09_customer_activity.png", dpi=150)
plt.close()

# ---------- 10. Failed transaction analysis ----------
print("Plot 10: Failed transaction analysis...")
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
fail_by_channel = df.groupby("channel")["status"].apply(lambda x: (x == "failed").mean() * 100)
fail_by_channel.sort_values().plot.barh(ax=axes[0], color="tomato")
axes[0].set_title("Failure Rate by Channel (%)")
axes[0].set_xlabel("Failure Rate %")

fail_by_hour = df.groupby("hour")["status"].apply(lambda x: (x == "failed").mean() * 100)
axes[1].plot(fail_by_hour.index, fail_by_hour.values, marker="o", color="tomato")
axes[1].set_title("Failure Rate by Hour (%)")
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Failure Rate %")
axes[1].set_xticks(range(24))
plt.tight_layout()
plt.savefig(f"{OUT}10_failed_txn_analysis.png", dpi=150)
plt.close()

# ---------- Summary stats to file ----------
print("\nWriting summary stats...")
with open(f"{OUT}eda_summary.txt", "w") as f:
    f.write("EDA SUMMARY\n" + "=" * 60 + "\n\n")
    f.write(f"Total transactions: {len(df):,}\n")
    f.write(f"Unique customers: {df['customer_id'].nunique():,}\n")
    f.write(f"Unique accounts: {df['account_id'].nunique():,}\n")
    f.write(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    f.write(f"Total amount: {df['amount'].sum():,.0f} NGN\n\n")

    f.write("Transaction type:\n")
    for val, cnt in df["transaction_type"].value_counts().items():
        f.write(f"  {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)\n")

    f.write("\nChannel:\n")
    for val, cnt in df["channel"].value_counts().items():
        f.write(f"  {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)\n")

    f.write("\nStatus:\n")
    for val, cnt in df["status"].value_counts().items():
        f.write(f"  {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)\n")

    f.write(f"\nAvg transactions per customer: {len(df)/df['customer_id'].nunique():.1f}\n")
    f.write(f"Median amount: {df['amount'].median():,.0f} NGN\n")
    f.write(f"Mean amount: {df['amount'].mean():,.0f} NGN\n")
    f.write(f"Max amount: {df['amount'].max():,.0f} NGN\n")

    f.write(f"\nBalance inconsistent rows: {(~df['balance_consistent']).sum():,}\n")
    f.write(f"Amount outliers: {df['amount_outlier'].sum():,}\n")

print("All plots saved to output/")
print("Done.")
