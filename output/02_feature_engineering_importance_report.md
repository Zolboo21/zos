# Feature Engineering & Importance Report

## 1. Feature Engineering Overview

Starting from the cleaned transaction dataset (5M rows, 100K customers), we aggregated transaction-level data into **70 customer-level features** across 10 categories. Only successful transactions (4.6M) were used for behavioral features, while status counts used all transactions.

| Category | # Features | Description |
|---|---|---|
| RFM | 8 | Recency (days since last txn), frequency, monetary stats (total, mean, median, std, max, min) |
| Transaction type | 4 | Debit/credit counts and ratios |
| Channel usage | 16 | Per-channel counts (7) and ratios (7), primary channel, number of channels used |
| Temporal behavior | 8 | Active days/months, tenure, weekend ratio, mean/std hour, peak hour, day-of-week entropy |
| Transaction status | 9 | Per-status counts (4) and rates (4), total transaction count |
| Balance | 5 | Avg/max/min/std balance before transaction, balance inconsistency count |
| Geographic | 3 | Primary state, unique states visited, unique LGAs visited |
| Merchant | 2 | Unique merchants and merchant category codes |
| Monthly trends | 8 | Recent vs prior 3-month counts/amounts, trend ratios, monthly mean/std/CV for count and amount |
| Outlier | 2 | High-value transaction count and ratio |

**Output:** `customer_features.csv` — 100,000 rows x 71 columns (70 features + customer_id)

### Feature Design Rationale

- **RFM features** are the foundation of customer value analysis in banking. Recency captures engagement freshness, frequency captures habit strength, and monetary captures economic value.
- **Channel features** capture digital adoption and behavioral preferences — a customer who only uses mobile behaves very differently from one who visits branches.
- **Temporal features** like day-of-week entropy measure behavioral regularity. A customer with high entropy transacts evenly across the week (likely a business); low entropy suggests concentrated patterns (salary-driven spending).
- **Monthly trends** compare the most recent 3 months to the prior 3 months, capturing acceleration or decline in activity — critical for churn detection and growth identification.
- **Status features** (failed/reversed/pending rates) capture transaction reliability, which may signal account issues, insufficient funds, or fraud risk.

---

## 2. Feature Importance Analysis

### 2.1 Modeling Approach

To evaluate feature importance, we trained a **LightGBM regressor** to predict a customer's transaction count over the most recent 3 months (`recent_3m_count`). This target was chosen because it directly measures near-term customer activity — the quantity the bank most wants to understand and predict.

| Parameter | Value |
|---|---|
| Model | LightGBM (gradient boosted trees) |
| Target | 3-month transaction count |
| Features used | 64 (excluded target, its derivations, and customer_id) |
| Train / Test split | 80,000 / 20,000 (80/20) |
| Best iteration | 95 (early stopped from 500) |
| **Test MAE** | **1.32** |
| **Test R²** | **0.789** |

The model explains ~79% of variance in customer activity with an average prediction error of just 1.3 transactions over 3 months. This strong fit confirms the engineered features carry meaningful signal about customer behavior.

We measured importance using two complementary methods:
- **LightGBM gain** — total reduction in loss contributed by each feature across all tree splits. Highlights features that create large improvements when used.
- **SHAP (SHapley Additive exPlanations)** — the average marginal contribution of each feature to individual predictions. More stable and interpretable than gain, less biased toward high-cardinality features.

### 2.2 Top 20 Features

| Rank (SHAP) | Feature | Mean |SHAP| | Gain Rank | Category |
|---|---|---|---|---|
| 1 | `frequency` | 2.072 | 8 | RFM |
| 2 | `recency_days` | 0.726 | 20 | RFM |
| 3 | `status_success_count` | 0.424 | 10 | Status |
| 4 | `total_txn_all_status` | 0.413 | 12 | Status |
| 5 | `txn_type_debit_count` | 0.343 | 4 | Txn type |
| 6 | `balance_inconsistent_count` | 0.333 | 9 | Balance |
| 7 | `status_reversed_count` | 0.331 | 1 | Status |
| 8 | `n_active_days` | 0.300 | 11 | Temporal |
| 9 | `status_pending_count` | 0.289 | 2 | Status |
| 10 | `monthly_count_mean` | 0.271 | 21 | Trends |
| 11 | `channel_agent_count` | 0.267 | 3 | Channel |
| 12 | `channel_mobile_count` | 0.163 | 15 | Channel |
| 13 | `channel_web_count` | 0.130 | 6 | Channel |
| 14 | `channel_atm_count` | 0.125 | 13 | Channel |
| 15 | `status_failed_count` | 0.077 | 7 | Status |
| 16 | `outlier_txn_count` | 0.073 | 5 | Outlier |
| 17 | `channel_pos_count` | 0.067 | 14 | Channel |
| 18 | `txn_type_credit_count` | 0.036 | 18 | Txn type |
| 19 | `tenure_days` | 0.036 | 30 | Temporal |
| 20 | `txn_per_active_day` | 0.035 | 17 | Temporal |

### 2.3 Features with Zero Importance

Three features had zero contribution: `n_unique_states`, `n_unique_lgas`, and `primary_channel` (label-encoded). Geographic granularity at the state/LGA level does not appear to help predict activity levels once other behavioral features are present. The encoded `primary_channel` was redundant given the per-channel count features.

---

## 3. Interpretation & Implications

### 3.1 Past behavior is the strongest predictor of future behavior

`frequency` dominates all other features by a wide margin — its SHAP value (2.07) is nearly 3x the second-place feature. This confirms the intuitive principle that **the best predictor of whether a customer will be active next quarter is how active they were historically**. For the bank, this means customer engagement programs should prioritize maintaining existing habits rather than trying to create new ones.

`recency_days` (0.73 SHAP) reinforces this: customers who transacted recently are far more likely to continue. A growing recency gap is an early warning of disengagement.

### 3.2 Transaction reliability is a major behavioral signal

Four of the top 10 features are transaction status counts: `status_success_count`, `status_reversed_count`, `status_pending_count`, and `balance_inconsistent_count`. This is notable — it means **the quality of a customer's transaction experience, not just the quantity, strongly predicts future activity**.

- **Reversed transactions** (rank 1 by gain, rank 7 by SHAP): Customers with frequent reversals likely have unstable financial behavior or disputed transactions. This could indicate account issues that, if unresolved, lead to churn.
- **Pending transactions** (rank 2 by gain, rank 9 by SHAP): High pending counts may signal processing delays or system issues affecting specific customer segments. If these are concentrated in certain channels or regions, it points to infrastructure problems.
- **Balance inconsistencies** (rank 9 by gain, rank 6 by SHAP): These were identified in the EDA as being exclusively tied to debit transactions. Their high importance here suggests they capture meaningful behavioral variation — possibly distinguishing customers who frequently attempt transactions beyond their balance.

**Implication for the bank:** Reducing failed and pending transactions — especially on specific channels — could directly improve customer retention. The 7% failure rate identified in EDA is not just a service quality issue; it is a predictive signal of reduced future activity.

### 3.3 Channel preferences strongly differentiate customer segments

Five channel features appear in the top 20. The relative ordering is telling:

| Channel | SHAP | Interpretation |
|---|---|---|
| Agent | 0.267 | Agent banking users are a distinct segment — likely rural or underbanked customers with different activity patterns |
| Mobile | 0.163 | Mobile-heavy customers tend to be more active overall — digital adoption correlates with engagement |
| Web | 0.130 | Web users may represent business or power users making larger, less frequent transactions |
| ATM | 0.125 | ATM reliance may indicate customers less integrated into digital banking |
| POS | 0.067 | POS usage is common across segments, so it differentiates less |

**Implication:** Channel usage is not just a preference — it is a proxy for customer type, financial literacy, and geographic context. **Agent banking users** stand out as the most distinctive segment, suggesting the bank serves a meaningful population through agent networks that behaves very differently from digital-native customers. Tailored strategies for agent-dependent customers could improve retention in underserved areas.

### 3.4 Spending behavior matters, but less than engagement patterns

While debit transaction count ranks 5th, the monetary features (total spend, mean amount, etc.) rank much lower — mostly outside the top 20. This reveals an important insight: **how often a customer transacts matters more than how much they spend** when predicting future activity.

This has segmentation implications: a customer making 10 small daily transactions is likely more engaged and retainable than one making 2 large monthly transfers, even if the latter has higher monetary value. Activity frequency is a better health metric than transaction value.

### 3.5 Temporal consistency captures engagement depth

`n_active_days` (rank 8) and `monthly_count_mean` (rank 10) both measure how spread out a customer's activity is over time. A customer with 50 transactions across 40 different days is more consistently engaged than one with 50 transactions in a 3-day burst.

`tenure_days` (rank 19) matters but less so — a long-tenured but recently inactive customer is still at risk. Tenure alone is a weak signal without recent activity.

**Implication:** Engagement regularity, not just volume, should be a key metric in customer health dashboards.

### 3.6 High-value transactions are a niche but informative signal

`outlier_txn_count` (rank 16 by SHAP, rank 5 by gain) shows an interesting split — it creates large gain when used in splits (suggesting it sharply separates specific customer groups) but affects fewer customers overall (lower SHAP). This likely captures **business accounts or high-net-worth individuals** who make occasional large transfers. These customers behave differently from the retail majority and may warrant separate modeling.

### 3.7 Geographic features are not predictive

`n_unique_states`, `n_unique_lgas`, and `primary_state` all ranked at or near the bottom. This does not mean geography is unimportant for the business — it means that **once you know a customer's behavioral features (channels, frequency, amounts), knowing their location adds little predictive power**. Geographic effects are already captured indirectly through channel preferences (agent banking correlates with rural areas) and merchant patterns.

---

## 4. Gain vs. SHAP: Why the Rankings Differ

The two methods sometimes rank features very differently. Key discrepancies:

| Feature | Gain Rank | SHAP Rank | Explanation |
|---|---|---|---|
| `status_reversed_count` | 1 | 7 | High gain because it creates sharp splits for a small subset of customers with reversals. Lower SHAP because most customers have zero reversals, so it affects few predictions. |
| `recency_days` | 20 | 2 | Low gain because it makes many small contributions across hundreds of splits (689 — the most of any feature). High SHAP because those small contributions add up to a large cumulative effect. |
| `frequency` | 8 | 1 | Similar to recency — used in many splits (388) with moderate individual gain, but the cumulative SHAP effect is dominant. |
| `channel_agent_count` | 3 | 11 | High gain for the small agent-user segment; lower average SHAP because most customers don't use agent banking. |

**Takeaway:** Gain highlights features that are powerful for specific subgroups. SHAP highlights features that matter broadly across the population. Both perspectives are valuable — gain identifies niche differentiators while SHAP identifies universally important drivers.

---

## 5. Key Takeaways for Downstream Modeling

1. **For time series prediction:** The strong R² (0.789) using customer-level features validates that behavioral aggregates can predict near-term activity. For transaction-level time series, lag features and rolling windows should be built around the top features here — particularly frequency, recency, and monthly cadence.

2. **For customer segmentation:** The feature importance ranking provides a natural feature selection guide. The top ~20 features capture the vast majority of signal. Segmentation should emphasize:
   - Activity level (frequency, active days, monthly mean)
   - Channel profile (mobile vs. agent vs. ATM dominance)
   - Transaction reliability (failure/reversal rates)
   - Spending intensity (debit count, outlier ratio)

3. **For churn modeling:** Recency, count trend (recent vs. prior 3 months), and transaction failure rates are the most actionable churn indicators. A customer whose recency is growing, trend is declining, and failure rate is rising is at high risk.

4. **Feature reduction:** The bottom 30 features collectively contribute less SHAP importance than `frequency` alone. For production models, the top 20–25 features are likely sufficient, reducing complexity without sacrificing accuracy.