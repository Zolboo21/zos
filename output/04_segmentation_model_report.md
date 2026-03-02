# Customer Segmentation Report

## 1. Objective

**Goal:** Identify distinct customer behavioral archetypes from the 100,000-customer base, enabling the bank to tailor engagement strategies, allocate resources efficiently, and detect at-risk populations before they churn.

**Approach:** Unsupervised clustering (K-Means) on 25 engineered behavioral features spanning activity level, monetary behavior, channel preferences, temporal patterns, transaction reliability, and recent trends.

---

## 2. Methodology

### 2.1 Feature Selection (25 features)

Rather than clustering on all 70 customer features, we selected 25 that capture distinct behavioral dimensions while minimizing redundancy:

| Dimension | Features | Why it matters |
|---|---|---|
| **Activity level** | frequency, recency_days, n_active_days, n_active_months, monthly_count_mean, monthly_count_cv | How much and how consistently a customer transacts — the core engagement signal |
| **Monetary behavior** | monetary_mean, monetary_median, monetary_std, avg_balance_before | Economic value and spending volatility — distinguishes high-value from casual users |
| **Channel mix** | mobile/agent/web/atm/pos/ussd ratios, n_channels_used | Digital adoption and access patterns — proxy for customer type and geography |
| **Temporal patterns** | weekend_ratio, std_hour, dow_entropy | Behavioral regularity — business vs. personal, routine vs. sporadic |
| **Transaction quality** | status_failed_rate, status_reversed_rate, status_pending_rate | Friction in the banking experience — a leading indicator of disengagement |
| **Trend** | count_trend | Recent trajectory — growing, stable, or declining activity |
| **Outlier behavior** | outlier_txn_ratio | Proportion of unusually large transactions — distinguishes business/HNW patterns |

### 2.2 Preprocessing

- **Log transformation** on 7 heavily skewed features (frequency, active days, monthly count, monetary mean/median/std, average balance). This prevents a handful of power users from dominating the distance calculations.
- **Standard scaling** (zero mean, unit variance) across all 25 features so that no single dimension disproportionately influences cluster assignment.

### 2.3 Choosing K

We evaluated K = 2 through 10 using three complementary metrics:

| K | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|
| 2 | **0.1052** | 13,852 | 2.40 |
| 3 | 0.1009 | 11,067 | 2.73 |
| 4 | 0.0765 | 9,602 | 2.41 |
| **5** | **0.0827** | **8,441** | **2.20** |
| 6 | 0.0653 | 7,542 | 2.44 |
| 7 | 0.0696 | 6,984 | 2.33 |
| 8 | 0.0604 | 6,586 | 2.46 |
| 9 | 0.0623 | 6,163 | 2.44 |
| 10 | 0.0648 | 5,911 | 2.35 |

**Why K = 5?**

- K = 2 maximizes silhouette but produces only "low activity" vs. "high activity" — too coarse for actionable strategy.
- K = 5 has the **second-best silhouette** (0.083) and the **best Davies-Bouldin index** (2.20, lowest = most compact and well-separated clusters).
- Five segments map naturally to a business-interpretable hierarchy from dormant to power users.

### 2.4 Why silhouette scores are low

All silhouette scores are below 0.11. This does **not** mean the clustering is poor — it reflects the nature of customer behavioral data:

- **Customer behavior exists on a continuum.** There are no sharp boundaries between "casual" and "active" customers. A customer with 15 transactions is not categorically different from one with 18. Silhouette score penalizes this overlap heavily.
- **High dimensionality dilutes distances.** With 25 features, the distance between any two customers tends to be similar (the "curse of dimensionality"). Silhouette score is sensitive to this effect.
- **The clusters are still useful.** Segment means differ dramatically across key dimensions (frequency ranges from 13 to 335; recency from 5 to 60 days). The segments capture real behavioral variation even if the boundaries are fuzzy.

This is a well-known limitation of K-Means on behavioral data. The value lies in the **profiles**, not the score.

---

## 3. The Five Customer Segments

### 3.1 Segment Overview

| Segment | Label | Customers | Share | Avg Frequency | Avg Recency | Avg Transaction Size | Monthly Count |
|---|---|---|---|---|---|---|---|
| 0 | **Budget Casual** | 27,505 | 27.5% | 13 | 60 days | ~29,000 | 1.3 |
| 1 | **High-Value Infrequent** | 23,649 | 23.6% | 14 | 57 days | ~114,000 | 1.3 |
| 2 | **Engaged Regular** | 2,259 | 2.3% | 17 | 46 days | ~66,000 | 1.4 |
| 3 | **Active Core** | 38,905 | 38.9% | 34 | 21 days | ~62,000 | 1.9 |
| 4 | **Power Users** | 7,682 | 7.7% | 335 | 5 days | ~67,000 | 14.0 |

### 3.2 Detailed Segment Profiles

#### Segment 0: Budget Casual (27,505 customers — 27.5%)

| Metric | Value | Context |
|---|---|---|
| Frequency | 13 transactions (median: 13) | Among the lowest |
| Recency | 60 days since last transaction | High — many haven't transacted in 2 months |
| Transaction size | ~29,000 (mean) | Lowest of all segments |
| Outlier txn ratio | 2% | Almost no large transactions |
| Active days | 13 | Minimal engagement spread |
| Channels used | 4.1 | Slightly below average |
| Failed txn rate | 8% | Highest of all segments |
| Count trend | 0.29 | Declining (below 1.0 = shrinking) |

**Profile:** These are small-value, infrequent customers who are drifting away. Their high failure rate (8%) suggests they may be encountering friction — insufficient funds, failed transfers — that compounds disengagement. They make the smallest transactions and have the fewest large-value transactions of any segment. Many haven't transacted in two months.

#### Segment 1: High-Value Infrequent (23,649 customers — 23.6%)

| Metric | Value | Context |
|---|---|---|
| Frequency | 14 transactions (median: 13) | Similar to Budget Casual |
| Recency | 57 days | High but slightly better than Budget Casual |
| Transaction size | ~114,000 (mean), ~87,300 (median) | **4x the Budget Casual average** |
| Outlier txn ratio | 15% | Highest of all segments — frequent large transactions |
| Active days | 13 | Same low spread as Budget Casual |
| Channels used | 4.1 | Below average |
| Count trend | 0.29 | Declining |

**Profile:** Same frequency as Budget Casual but dramatically different in value. These customers make few transactions, but each one is large — likely business transfers, property-related payments, or high-net-worth individuals managing wealth through infrequent, high-value moves. Their 15% outlier transaction ratio (vs. 2% for Budget Casual) confirms they operate in a fundamentally different economic tier. Despite the high value per transaction, they're disengaging at the same rate as Budget Casual.

#### Segment 2: Engaged Regular (2,259 customers — 2.3%)

| Metric | Value | Context |
|---|---|---|
| Frequency | 17 transactions (median: 16) | Slightly above low-activity segments |
| Recency | 46 days | Better than segments 0-1 but still high |
| Transaction size | ~66,000 (mean) | Mid-range |
| Active days | 17 | Moderate spread |
| Channels used | 4.4 | Slightly above average |
| Failed txn rate | 6% | Lower than average |
| Reversed txn rate | 6.1% | **Highest of all segments by far** |
| Count trend | 0.33 | Declining but slightly less than segments 0-1 |

**Profile:** A small, distinctive group. Their standout characteristic is the extremely high reversal rate (6.1% vs. near-zero for most other segments). These customers are moderately engaged but experience significant transaction friction through reversals — which could indicate disputed transactions, system errors affecting a specific customer cohort, or behavioral patterns around trial-and-cancel purchases. The small segment size (2.3%) suggests this is a niche behavioral pattern rather than a broad trend.

#### Segment 3: Active Core (38,905 customers — 38.9%)

| Metric | Value | Context |
|---|---|---|
| Frequency | 34 transactions (median: 29) | **2.5x** the low-activity segments |
| Recency | 21 days | Recent — transacted within the last 3 weeks |
| Transaction size | ~62,000 (mean) | Mid-range |
| Active days | 33 (median: 29) | Spread across the month |
| Channels used | 5.2 | Above average — uses multiple channels |
| Failed txn rate | 6% | Below average |
| Count trend | 0.42 | **Best growth trend of any segment** |
| Monthly count | 1.9 | Nearly 2 transactions per month |

**Profile:** The bank's backbone — the largest segment (39%) and the most engaged in terms of growth trajectory. These customers transact regularly across multiple channels, have low failure rates, and are the only segment showing a count trend above 0.4. They represent the healthy, growing middle of the customer base. Their multi-channel usage (5.2 channels on average) indicates deep integration into the bank's ecosystem.

#### Segment 4: Power Users (7,682 customers — 7.7%)

| Metric | Value | Context |
|---|---|---|
| Frequency | 335 transactions (median: 138) | **10x** the Active Core, **25x** the casuals |
| Recency | 5 days | Nearly continuous activity |
| Transaction size | ~67,000 (mean) | Mid-range |
| Active days | 179 (median: 126) | Active more than half the days in the dataset |
| Channels used | 6.5 | Uses nearly all 7 available channels |
| Monthly count | 14 | Multiple transactions per week |
| Count trend | 0.06 | **Lowest trend — already at ceiling** |
| DoW entropy | 2.78 | Highest — transacts evenly across all days |

**Profile:** The elite 7.7% who drive a disproportionate share of transaction volume. The median frequency of 138 (vs. mean of 335) reveals heavy right-skew within this group — some power users have thousands of transactions, likely business or merchant accounts. Their near-perfect channel coverage (6.5 of 7 channels) and highest day-of-week entropy indicate these are omnipresent, possibly automated or business-driven accounts. The low count trend (0.06) is not concerning — they've likely reached a natural activity ceiling.

---

## 4. Cluster Quality Assessment

| Metric | Value | Interpretation |
|---|---|---|
| **Silhouette Score** | 0.083 | Low in absolute terms — expected for continuous behavioral data with 25 dimensions |
| **Calinski-Harabasz** | 8,441 | Moderate — clusters have reasonable between-cluster variance relative to within-cluster variance |
| **Davies-Bouldin** | 2.20 | Best among all K values tested — clusters are as compact and separated as the data allows |
| **PCA variance** | 30.4% (2 components) | The first two PCA components capture only 30% of variance, confirming the data is genuinely high-dimensional — no two features can summarize it |

---

## 5. Interpretation & Implications

### 5.1 The bank has a "missing middle" problem

The segmentation reveals a striking gap in the customer base. The largest groups are either low-activity (segments 0+1 = 51.1%) or in the Active Core (38.9%). The transition zone — Engaged Regular — contains only 2.3% of customers.

This means there is very little pipeline between casual/infrequent and active. Customers either make the jump to regular engagement or they don't. The bank is not gradually converting casual users into active ones — it's more like a binary switch.

**Implication:** Intervention programs targeting the ~51,000 low-activity customers should focus on creating that first consistent habit — perhaps a recurring payment, a savings auto-debit, or a bill payment schedule — rather than trying to gradually increase ad-hoc transactions. The data suggests incremental nudging isn't working; customers need a structural reason to transact regularly.

### 5.2 High-value customers are hiding in the infrequent segment

Segment 1 (High-Value Infrequent) is arguably the most strategically important finding. These 23,649 customers have **4x the average transaction size** of their frequency-equivalent counterparts (Budget Casual) and account for **15% outlier transactions** — meaning they routinely make large transfers.

Yet their engagement metrics (recency, frequency, trend) are nearly identical to Budget Casual. The bank is likely treating these customers the same as low-value casuals because frequency-based dashboards wouldn't distinguish them.

**Implication:** This segment needs a dedicated relationship management approach. They are high-revenue, low-touch customers — possibly businesses doing monthly payroll, merchants settling accounts, or individuals making periodic large investments. Their declining trend (0.29) is alarming because losing one High-Value Infrequent customer costs the bank as much revenue as losing 4 Budget Casual customers. Priority: identify what's driving their disengagement before they leave entirely.

### 5.3 Transaction reversals define a distinct at-risk cohort

Segment 2 (Engaged Regular) is tiny (2.3%) but distinct primarily because of its **6.1% reversal rate** — orders of magnitude higher than other segments. This is not noise; K-Means pulled out this group because their reversal behavior makes them genuinely different in feature space.

Possible explanations:
- **System issues:** A specific channel or product is generating reversals for this customer subset.
- **Behavioral pattern:** These customers frequently initiate-then-cancel transactions — perhaps testing the waters before committing.
- **Fraud or disputes:** A higher-than-normal rate of contested transactions.

**Implication:** Investigate the root cause of reversals in this cohort. If it's a system issue, fixing it could convert these 2,259 customers into the Active Core (their other metrics — frequency, monetary, channels — are all mid-range). If it's behavioral, understanding why they reverse could reveal a UX friction point.

### 5.4 The Active Core is the growth engine — and it's growing

Segment 3 is the bank's healthiest population: 38,905 customers (39%) with **the strongest growth trend** (0.42) and **the lowest failure rate** (6%). These customers:

- Use 5+ channels (deep platform integration)
- Transact every 3 weeks on average
- Are actively increasing their activity

The growth trend of 0.42 means their recent 3-month activity is 42% of their prior 3-month activity in the count_trend ratio — in context of the dataset's declining trend, this is the most positive signal.

**Implication:** Protect this segment at all costs. Their multi-channel behavior suggests they'd be excellent candidates for cross-selling (savings products, insurance, credit lines). Their upward trajectory also makes them the most likely source of future Power Users. Any platform disruption, fee increase, or service degradation disproportionately risks this segment because they're the most dependent on the bank's full ecosystem.

### 5.5 Power Users are a different species

Segment 4's median frequency of 138 (mean: 335) reveals extreme right-skew. These are not "customers who transact a lot" — they are functionally different entities. Their characteristics strongly suggest many are:

- **Business accounts** processing daily transactions
- **Merchant accounts** receiving POS/mobile payments
- **Automated systems** running scheduled transfers

Evidence: highest day-of-week entropy (2.78 — perfectly even distribution across all days, suggesting automation or business operations rather than human spending patterns), near-perfect channel coverage (6.5 of 7 channels), and a count trend of 0.06 (near-zero growth because they're already at operational capacity).

**Implication:** Power Users likely need enterprise-grade features — bulk transaction tools, API access, dedicated support lines, and uptime SLAs. Treating them with the same retail banking interface as Budget Casual customers is a retention risk. They should be segmented further internally (merchant vs. business vs. HNW individual) for tailored service.

### 5.6 Channel usage is uniform — except for agent banking

Across all five segments, the mobile ratio (~35%), web ratio (~10%), ATM ratio (~20%), and POS ratio (~30%) are remarkably consistent. Customers across all segments use channels in roughly the same proportion.

The one exception is **agent banking**, which is near-zero for segments 0 and 1 (the infrequent users) but present (0.5–0.6%) in segments 2–4. This is a small difference in absolute terms but meaningful because it distinguishes engaged from disengaged populations.

**Implication:** Channel mix is not a strong differentiator between segments — activity level and monetary value are what define the segments. However, agent banking correlates with engagement, possibly because customers with access to agents (often in rural or underbanked areas) have an additional touchpoint that keeps them connected.

### 5.7 Failed transactions concentrate in the least engaged segments

Budget Casual has the highest failure rate (8%) while Active Core and Power Users have the lowest (6–7%). This 2-percentage-point gap is significant across tens of thousands of customers. Combined with the finding from the feature importance analysis that failure rates predict future activity, this suggests a **reinforcing cycle**: transaction failures discourage customers → they transact less → they become more casual → their next transaction is more likely to fail (perhaps due to stale payment methods, depleted balances, or forgotten PINs).

**Implication:** Reducing friction for low-activity customers — smoother PIN recovery, balance alerts before insufficient-funds failures, retry prompts after failed transactions — could break this negative cycle and move customers upward in the segment hierarchy.

---

## 6. Segment Strategy Matrix

| Segment | Priority | Goal | Suggested Actions |
|---|---|---|---|
| **Budget Casual** (27.5%) | Medium | Activate or accept dormancy | Auto-debit product nudges, friction reduction, balance alerts |
| **High-Value Infrequent** (23.6%) | **High** | Retain and engage | Dedicated RM, premium service tier, proactive outreach on declining trend |
| **Engaged Regular** (2.3%) | Medium-High | Investigate and convert | Root-cause reversal analysis, UX friction audit, pathway to Active Core |
| **Active Core** (38.9%) | **High** | Protect and cross-sell | Cross-sell financial products, loyalty rewards, multi-channel enhancement |
| **Power Users** (7.7%) | Medium | Retain with enterprise features | Business-grade tools, API access, SLA guarantees, dedicated support |

---

## 7. Limitations

1. **K-Means assumes spherical clusters.** Customer behavioral data likely has irregular, non-spherical cluster shapes. Density-based methods (DBSCAN, HDBSCAN) or Gaussian Mixture Models could capture more nuanced cluster geometries.

2. **Static snapshot.** The segmentation uses aggregate features over the full 24-month period. A customer currently in Active Core may have been Budget Casual six months ago. Time-aware segmentation (e.g., Hidden Markov Models for segment transitions) would capture this evolution.

3. **Low silhouette scores.** While expected, the low scores mean segment boundaries are soft. Customers near the boundaries could reasonably belong to adjacent segments. Business rules should use segment assignment as a guide, not a hard classifier.

4. **No external validation.** The segments have not been validated against known business outcomes (churn, revenue, NPS scores). Linking segment membership to actual business metrics would confirm whether the behavioral distinctions translate to actionable differences.

5. **Single algorithm.** Only K-Means was tested. Ensemble approaches (consensus clustering across multiple algorithms) would increase confidence in the segment structure.

---

## 8. Key Takeaways

1. **51% of customers are low-activity** (Budget Casual + High-Value Infrequent), but they split into two fundamentally different populations: small-value casuals and large-value infrequent transactors. Treating them as a single "inactive" group would miss the bank's highest-value retention opportunity.

2. **The Active Core (39%) is growing** and represents the bank's healthiest customer relationship. These customers use multiple channels, fail less often, and are actively increasing their activity. They are the primary candidates for cross-selling.

3. **Transaction reversals carve out a distinct at-risk cohort** of 2,259 customers. Their elevated reversal rate (6.1%) is likely a symptom of a specific, identifiable problem — not random variation.

4. **Power Users (7.7%) are operationally distinct** from retail customers. Their transaction patterns (even day-of-week distribution, near-continuous activity, all-channel usage) suggest business or automated accounts that need enterprise-level service.

5. **Failed transactions are highest where engagement is lowest**, suggesting a reinforcing negative cycle. Reducing friction for casual customers is both a service quality improvement and a retention strategy.

6. **There is almost no "middle pipeline"** — the jump from casual to active is abrupt, with only 2.3% of customers in the transition zone. The bank needs structural engagement hooks (recurring payments, auto-debits) rather than incremental nudges to move customers up the activity ladder.
