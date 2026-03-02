# Khan Bank — Customer Behavior Analysis

A data science project analyzing 5M transactions across 100K customers (Jan 2023 – Dec 2024, ~328B NGN volume) to predict customer activity and identify behavioral segments.

## Setup

```bash
./prepare.sh
```

This installs [uv](https://github.com/astral-sh/uv), creates a virtual environment, syncs dependencies, and downloads the dataset from Hugging Face (`Noysaa/data_scientist_task`) into `notebook/dataset.csv`.

## Notebooks

Run in order from the `notebook/` directory:

| # | Notebook | Description |
|---|---|---|
| 1 | `01_analysis.ipynb` | Data cleaning & EDA |
| 2 | `02_feature.ipynb` | Feature engineering (70 customer-level features) + SHAP importance |
| 3 | `03_prediction.ipynb` | Time series model — predicts next-month transaction count per customer |
| 4 | `04_segmentation.ipynb` | K-Means clustering into behavioral archetypes |

## Key Results

- **Predictive model:** LightGBM on a customer×month panel (1.7M rows, 47 features). Top drivers: recent transaction count, lag features, rolling trends.
- **Segmentation:** K-Means (k=5) on 25 behavioral features. Segments range from dormant/low-activity users to high-frequency digital-first customers.
- **Feature importance:** Transaction frequency and recency dominate; channel mix and failure rates are secondary signals.

## Notes

Detailed methodology and findings are in `notes/`:
- `01_data_cleaning_eda_report.md`
- `02_feature_engineering_importance_report.md`
- `03_predictive_model_report.md`
- `04_segmentation_model_report.md`
