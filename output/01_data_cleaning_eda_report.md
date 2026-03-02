# Data Cleaning & EDA Report

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total transactions | 5,000,000 |
| Unique customers | 100,000 |
| Unique accounts | 100,000 |
| Date range | Jan 2023 — Dec 2024 (2 years) |
| Total volume | 328.1 billion NGN |
| Avg txns per customer | 50 |
| Columns (original) | 15 |

**Columns:** `transaction_id`, `account_id`, `customer_id`, `timestamp`, `amount`, `balance_before_ngn`, `balance_after_ngn`, `transaction_type`, `channel`, `merchant_category_code`, `merchant_name`, `location_lga`, `location_state`, `device_id`, `status`

---

## 2. Data Quality Issues

### 2.1 Missing Values

| Column | Null Count | % |
|---|---|---|
| merchant_category_code | 2,998,137 | 60.0% |
| merchant_name | 2,998,137 | 60.0% |
| device_id | 2,750,464 | 55.0% |

- Merchant fields are null together — likely P2P transfers, salary payments, or internal bank transactions that don't involve a named merchant.
- device_id nulls likely come from POS/ATM terminals or branch transactions that don't capture device info.

### 2.2 Duplicates

- **0** duplicate transaction IDs.
- **0** fully duplicate rows.

### 2.3 Balance Inconsistencies

- **346,733 rows (6.93%)** where `balance_after` does not match `balance_before ± amount`.
- All inconsistencies are on **debit** transactions. Credit transactions are 100% consistent.
- These likely correspond to **failed or reversed** transactions where the balance was not actually debited, but the record still exists.

### 2.4 Anomalies

- No negative amounts or balances.
- No zero-amount transactions.
- No invalid timestamps.
- The data is structurally clean — issues are limited to missing merchant/device info and the balance inconsistencies noted above.

---

## 3. Cleaning Steps Applied

1. **Duplicate removal** — no duplicates found, no rows dropped.
2. **Timestamp validation** — all valid, parsed to datetime.
3. **Amount validation** — no negatives or zeros.
4. **Balance consistency flag** — added `balance_consistent` boolean column.
5. **Categorical standardization** — lowercased and trimmed `transaction_type`, `channel`, `status`, `location_state`, `location_lga`.
6. **Null handling:**
   - `merchant_name` → filled with `"unknown"`
   - `merchant_category_code` → filled with `-1`
   - `device_id` → filled with `"unknown"`
7. **Time feature extraction** — added `date`, `year`, `month`, `day`, `hour`, `day_of_week`, `is_weekend`.
8. **Outlier flagging** — added `amount_outlier` flag using 3x IQR method (threshold: 168,000 NGN). 398,964 transactions flagged.

**Cleaned dataset:** 5,000,000 rows x 24 columns → `output/cleaned.csv`

---

## 4. Exploratory Data Analysis

### 4.1 Transaction Types

| Type | Count | % |
|---|---|---|
| Debit | 3,250,541 | 65.0% |
| Credit | 1,749,459 | 35.0% |

Customers spend roughly twice as often as they receive — typical for retail banking where salary credits are infrequent but spending is daily.

### 4.2 Transaction Status

| Status | Count | % |
|---|---|---|
| Success | 4,599,963 | 92.0% |
| Failed | 349,765 | 7.0% |
| Pending | 40,383 | 0.8% |
| Reversed | 9,889 | 0.2% |

A 7% failure rate is notable. Failed transactions are potential friction points worth investigating for customer experience improvement.

### 4.3 Channels

| Channel | Count | % |
|---|---|---|
| Mobile | 1,749,043 | 35.0% |
| POS | 1,501,370 | 30.0% |
| ATM | 999,469 | 20.0% |
| Web | 500,493 | 10.0% |
| Branch | 149,372 | 3.0% |
| USSD | 75,336 | 1.5% |
| Agent | 24,917 | 0.5% |

Mobile dominates, followed by POS. Digital channels (mobile + web) account for 45% of all transactions. Physical channels (POS + ATM + branch) account for 53%.

### 4.4 Amount Distribution

| Stat | Value (NGN) |
|---|---|
| Min | 100 |
| 25th percentile | 4,000 |
| Median | 13,300 |
| Mean | 65,622 |
| 75th percentile | 45,000 |
| Max | 5,000,000 |

The distribution is heavily right-skewed (mean 5x the median). Most transactions are small everyday payments, with a long tail of large transfers.

### 4.5 Temporal Patterns

**Monthly volume:** Transaction counts and total amounts show a generally upward trend across 2023–2024, with seasonal spikes visible around December (holiday spending).

**Hourly pattern:** Transactions peak during business hours (9 AM – 6 PM), with the highest volume around midday. Activity drops sharply after midnight and picks up again around 6 AM.

**Day-of-week pattern:** Weekdays see higher transaction volumes than weekends. The distribution is relatively flat Monday–Friday with a visible dip on Saturday and Sunday.

### 4.6 Geographic Distribution (Top 10 States)

| State | Count | % |
|---|---|---|
| Lagos | 1,225,836 | 24.5% |
| Abuja (FCT) | 542,759 | 10.9% |
| Rivers | 446,381 | 8.9% |
| Kano | 288,046 | 5.8% |
| Anambra | 239,518 | 4.8% |
| Oyo | 238,988 | 4.8% |
| Imo | 186,312 | 3.7% |
| Kaduna | 159,387 | 3.2% |
| Delta | 135,097 | 2.7% |
| Edo | 125,496 | 2.5% |

Lagos alone accounts for nearly a quarter of all transactions. The top 3 states (Lagos, Abuja, Rivers) represent 44.3% of total volume — reflecting Nigeria's economic concentration in these commercial hubs.

### 4.7 Top Merchants

Merchant data is only available for 40% of transactions (2M rows). Among known merchants, telecom (Airtel, MTN, Glo), grocery (Shoprite, Grand Square), and fuel stations are the most common — consistent with everyday consumer spending.

### 4.8 Customer Activity

- Average of 50 transactions per customer over 2 years.
- Distribution is right-skewed: most customers have moderate activity, with a subset of power users driving disproportionate volume.
- Total spend per customer also follows a long-tail distribution.

### 4.9 Failed Transaction Analysis

- Failure rates vary by channel but are broadly similar across most channels.
- Failure rates are relatively stable across hours of the day, with no dramatic spikes that would suggest systemic outages.

---

## 5. Key Takeaways for Modeling

1. **Customer segmentation** should leverage transaction frequency, amount patterns, channel preferences, and geographic data. The 1:1 customer-to-account mapping simplifies aggregation.

2. **Time series prediction** has strong temporal features to work with — clear hourly, daily, and monthly patterns. The 2-year window provides enough history for seasonal modeling.

3. **Feature engineering priorities:**
   - Lag features (previous N transactions per customer)
   - Rolling aggregates (7-day, 30-day transaction count/amount)
   - Channel usage ratios per customer
   - Failed transaction rates per customer
   - Recency/frequency/monetary (RFM) features
   - Weekend vs weekday behavior ratios

4. **Balance inconsistencies** (6.93%) should be handled carefully — either filtered out or treated as a separate signal (possible indicator of risky transactions).

5. **Amount outliers** (8%) are a meaningful segment, not noise — they represent large transfers that may indicate business accounts or high-value customers.