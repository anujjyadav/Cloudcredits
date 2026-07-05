# Walmart Store Sales Forecasting

A complete time series forecasting project that predicts weekly department-level sales across 45 Walmart stores. The project compares six forecasting approaches — from classical statistical models to modern gradient boosting — and identifies the best performer using rigorous evaluation metrics.

---

## Results at a Glance

| Rank | Model | RMSE | R² |
|------|-------|------|----|
|  1 | XGBoost | 2,516 | 0.9870 |
|  2 | LightGBM | 2,576 | 0.9863 |
|  3 | Linear Regression | 3,360 | 0.9775 |
| 4 | Holt-Winters* | — | 0.5376 |
| 5 | ARIMA(2,1,2)* | — | −0.1092 |

*Classical models evaluated on the aggregated total-sales series; ML models on the full store × department level.

---

## Project Structure

```
walmart-sales-forecasting/
│
├── walmart_sales_forecasting.ipynb   # Main notebook
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── train.csv                         # Historical weekly sales (421,570 rows)
├── features.csv                      # Store-level features (8,190 rows)
├── stores.csv                        # Store metadata (45 rows)
└── test.csv                          # Prediction target dates (115,064 rows)
```

---

## Dataset

| File | Rows | Key Columns |
|------|------|-------------|
| `train.csv` | 421,570 | Store, Dept, Date, Weekly_Sales, IsHoliday |
| `features.csv` | 8,190 | Temperature, Fuel_Price, MarkDown1–5, CPI, Unemployment |
| `stores.csv` | 45 | Store, Type (A/B/C), Size |
| `test.csv` | 115,064 | Store, Dept, Date, IsHoliday |

**Date range:** February 2010 – October 2012  
**Stores:** 45 across three types (A: large, B: medium, C: small)  
**Departments:** Up to 99 per store

---

## Notebook Walkthrough

The notebook follows a structured 11-step ML workflow:

| Step | Description |
|------|-------------|
| **1** | Import libraries and configure global settings |
| **2** | Load all four CSVs, merge them, handle missing MarkDown values |
| **3** | Exploratory Data Analysis — sales trends, monthly seasonality, store-type comparisons, holiday uplift |
| **4** | Time series decomposition (trend, seasonal, residual), ADF stationarity test, ACF/PACF plots |
| **5** | Feature engineering — lag features, rolling statistics, holiday flags, cyclic date encodings |
| **6** | Temporal train/validation split (cutoff: 2012-05-01) |
| **7** | Train six models: ARIMA, Holt-Winters, Linear Regression, Random Forest, XGBoost, LightGBM |
| **8** | Evaluate all models on MAE, RMSE, R², MAPE with side-by-side bar charts |
| **9** | Deep-dive into the best model: feature importance and predicted vs actual scatter |
| **10** | Overlay all model forecasts on the same aggregated sales chart |
| **11** | Final leaderboard and key business insights |

---

## Models Used

### Classical Time Series
- **ARIMA(2,1,2)** — AutoRegressive Integrated Moving Average fitted to the aggregated weekly total-sales series.
- **Holt-Winters Exponential Smoothing** — Additive trend and seasonal components with a period of 52 weeks.

### Machine Learning (store × department level)
- **Linear Regression** — Scaled baseline using StandardScaler.
- **Random Forest** — 100 trees, max depth 12, used as an ensemble baseline.
- **XGBoost** — 300 estimators, learning rate 0.05, depth 8, early stopping on validation set.
- **LightGBM** — 300 estimators, 63 leaves, early stopping; fastest to train at scale.

---

## Feature Engineering

The following features were constructed before training ML models:

| Category | Features |
|----------|----------|
| **Lag features** | Sales_Lag1, Sales_Lag2, Sales_Lag4, Sales_Lag52 |
| **Rolling statistics** | Rolling_Mean_4, Rolling_Mean_12, Rolling_Std_4 |
| **Group aggregates** | Dept_Avg_Sales, Store_Avg_Sales |
| **Holiday flags** | IsBlackFriday, IsChristmas, IsSuperBowl, IsLaborDay, IsThanksgiving |
| **Cyclic encodings** | Week_sin, Week_cos, Month_sin, Month_cos |
| **Calendar** | Year, Month, Week, Quarter, DayOfYear |
| **Store metadata** | Type (encoded), Size |
| **Economic** | Temperature, Fuel_Price, CPI, Unemployment, Total_MarkDown |

---

## Key Findings

1. **Lag features dominate** — Sales_Lag1, Sales_Lag52, and Rolling_Mean_4 are consistently the top predictors, capturing immediate, seasonal, and short-term autocorrelation.
2. **ML models outperform classical ones** — XGBoost achieves R² = 0.987 vs Holt-Winters' 0.538, because they leverage cross-store feature interactions that ARIMA cannot.
3. **Holiday weeks drive significant uplift** — Super Bowl, Thanksgiving, and Christmas weeks each show 10–25% higher sales on average.
4. **Store type matters** — Type A stores average ~2× the weekly sales of Type C stores; store size is among the top 5 most important features.
5. **Q4 is the peak period** — Weeks 47–52 (November–December) consistently represent the highest-sales stretch, amplified by MarkDown promotions.
6. **ARIMA struggles at the aggregated level** — The aggregated series is non-stationary and lacks the exogenous signals that differentiate store-department pairs.

---

## How to Run

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Place the dataset files (`train.csv`, `features.csv`, `stores.csv`, `test.csv`) in the project folder.

3. Open and run:

```bash
jupyter notebook walmart_sales_forecasting.ipynb
```

Run all cells from top to bottom (`Kernel → Restart & Run All`).

---

## Requirements

See `requirements.txt` for pinned versions. Core dependencies:

- Python 3.9+
- pandas, numpy, matplotlib, seaborn
- statsmodels (ARIMA, Holt-Winters, decomposition)
- scikit-learn (Linear Regression, Random Forest, metrics)
- xgboost, lightgbm
- jupyter, nbformat, nbconvert

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | mean(\|y − ŷ\|) | Average absolute dollar error per week |
| **RMSE** | √mean((y − ŷ)²) | Penalises large errors more heavily |
| **R²** | 1 − SS_res/SS_tot | Proportion of variance explained (1.0 = perfect) |
| **MAPE** | mean(\|y − ŷ\| / y) × 100 | Percentage error, scale-independent |

---

## License

This project is for educational  use. The dataset originates from the [Walmart Store Sales Forecasting Kaggle competition]
