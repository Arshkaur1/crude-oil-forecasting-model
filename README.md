# Crude Oil Forecasting Model

A machine learning pipeline to forecast **WTI crude oil prices** using historical price data, macroeconomic indicators, and market sentiment. The project produces a realistic, smoothed 30-day price forecast and delivers stakeholder-ready visualizations through Tableau dashboards.

---

## Project Overview

| | |
|---|---|
| **Target Variable** | WTI Crude Oil Price (USD/barrel) |
| **Forecast Horizon** | 30 days (Jan 2025) |
| **Data Range** | Jan 2018 – Dec 2024 |
| **Best Model** | XGBoost Regressor |
| **Visualization** | Tableau + Matplotlib |

---

## Repository Structure

```
COFM/
│
├── Final_COFM_Notebook.ipynb       # Full ML pipeline (data → features → model → forecast)
│
├── data/
│   ├── wti_crude_clean.csv             # WTI daily closing prices (Yahoo Finance)
│   ├── brent_crude_clean.csv           # Brent daily closing prices
│   ├── us_cpi_clean.csv                # US CPI (FRED API)
│   ├── us_fed_rate_clean.csv           # US Federal Funds Rate (FRED API)
│   ├── us_dxy_index_clean.csv          # US Dollar Index (DXY)
│   ├── opec_production_clean.csv       # OPEC monthly crude production (EIA API)
│   ├── us_crude_inventories_clean.csv  # US weekly crude inventories (EIA API)
│   ├── us_crude_production_clean.csv   # US field production (EIA API)
│   ├── google_sentiment_clean.csv      # Google News sentiment scores
│   ├── crude_oil_master_final.csv      # Merged master dataset
│   └── future_forecast_jan2025.csv     # Final 30-day forecast output
│
├── outputs/
│   ├── X_scaled.csv                    # Scaled feature matrix
│   ├── y_target.csv                    # Target variable
│   └── combined_wti_forecast.csv       # Actuals + forecast with confidence bands
│
└── README.md
```

---

## Data Sources

| Source | Data | API / Library |
|--------|------|--------------|
| Yahoo Finance | WTI (`CL=F`), Brent (`BZ=F`), DXY (`DX-Y.NYB`) | `yfinance` |
| FRED (St. Louis Fed) | US CPI (`CPIAUCSL`), Fed Funds Rate (`FEDFUNDS`) | REST API |
| EIA | OPEC production, US crude inventories, US field production | REST API v2 |
| Google News | Market sentiment scores | `GoogleNews` + `VADER` |

---

## Feature Engineering

Raw data was transformed into a rich feature set to capture price dynamics and market signals:

| Feature Type | Variables |
|---|---|
| **Lag features (lag-1)** | WTI price, Brent price, DXY index, Google Sentiment |
| **Percentage change (1d, 3d, 7d)** | WTI, Brent, DXY, Sentiment |
| **Rolling averages** | 7-day and 30-day means for price and sentiment |
| **Volatility (std dev)** | 7-day and 30-day windows for WTI, Brent, DXY, Sentiment |
| **Lagged volatility** | 1-day and 3-day lags on 7-day volatility features |

Missing values were handled with forward-fill after chronological sorting. Features were standardized using `StandardScaler` prior to model training.

---

## Modeling

### Train / Test Split
A **time-based 80/20 split** was used to prevent data leakage — no random shuffling.

### Models Trained
- **Naive Baseline** — predicts tomorrow's price = today's price
- **Random Forest Regressor** — tuned via `RandomizedSearchCV` with `TimeSeriesSplit` (5 folds)
- **XGBoost Regressor** — tuned via `RandomizedSearchCV` with `TimeSeriesSplit` (5 folds)

### Hyperparameter Search Spaces

**Random Forest**
```python
{
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
```

**XGBoost**
```python
{
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
```

---

## Model Evaluation

Performance was measured using **MAE**, **RMSE**, and **R²** on the held-out test set:

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Naive Baseline | — | — | — |
| Random Forest | ✓ | ✓ | ✓ |
| **XGBoost** ⭐ | **Best** | **Best** | **Best** |

> XGBoost outperformed both the Naive Baseline and Random Forest across all metrics and was selected as the final forecasting model.

---

## Forecasting

Using the best XGBoost model, WTI crude oil prices were forecast for **January 2025** (30 days):

- **Iterative forecasting** — each day's prediction feeds into the next as a lag feature
- **Smoothing** applied to reduce noise and produce realistic price trajectories
- **Clipping** used to keep projections within market-plausible bounds
- **Confidence bands** (±2%) exported for dashboard visualization

---

## Visualizations

Python (Matplotlib) and **Tableau** were used for all visualizations:

- Historical WTI price trend (2018–2024)
- Actual vs. Predicted prices on the test set
- 30-day WTI price forecast with confidence interval
- Feature importance comparison: Random Forest vs. XGBoost
- Correlation heatmap of engineered features

> The final interactive dashboard was built in **Tableau**, integrating all model outputs and forecast data for stakeholder presentation.

---

## Tech Stack

```
Python 3.x
├── pandas, numpy          – Data manipulation
├── scikit-learn           – Random Forest, preprocessing, cross-validation
├── xgboost                – Gradient boosted trees
├── yfinance               – Market price data
├── requests               – FRED & EIA API calls
├── vaderSentiment         – Sentiment scoring
├── GoogleNews             – News headline retrieval
└── matplotlib             – Python-side visualization

```

