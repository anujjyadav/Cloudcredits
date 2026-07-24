#  Energy Consumption Analysis

A complete end-to-end machine learning project that analyzes and predicts energy consumption patterns using time series analysis and regression models, applied to a real-world dataset containing power readings across three distribution zones with correlated weather variables.

---

##  Project Overview

This project follows a structured machine learning pipeline:

1. **Problem Definition** — Multi-output regression to predict power consumption across three zones
2. **Data Collection & Preparation** — Load and clean the 10-minute interval dataset
3. **Exploratory Data Analysis (EDA)** — Visualize trends, correlations, and seasonal patterns
4. **Feature Engineering** — Extract temporal features, lag features, and rolling statistics
5. **Train/Test Split** — Chronological split to respect time-series ordering
6. **Model Selection** — Compare Linear Regression, Random Forest, XGBoost, and more
7. **Model Training** — Train models on each zone independently
8. **Model Evaluation** — RMSE, MAE, and R² metrics
9. **Hyperparameter Tuning** — Grid search / manual tuning for best performance
10. **Results Visualization** — Predicted vs Actual plots, feature importances, residual analysis

---

##  Project Structure

```
Energy Consumption Analysis/
│
├── powerconsumption.csv          # Raw dataset
├── Energy_Consumption_Analysis.ipynb  # Main Jupyter Notebook
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

##  Dataset Description

| Column | Description |
|---|---|
| `Datetime` | Timestamp at 10-minute intervals (Jan–Dec 2017) |
| `Temperature` | Ambient temperature (°C) |
| `Humidity` | Relative humidity (%) |
| `WindSpeed` | Wind speed (m/s) |
| `GeneralDiffuseFlows` | General diffuse solar radiation (W/m²) |
| `DiffuseFlows` | Diffuse solar radiation flows (W/m²) |
| `PowerConsumption_Zone1` | Power consumption in Zone 1 (kW) |
| `PowerConsumption_Zone2` | Power consumption in Zone 2 (kW) |
| `PowerConsumption_Zone3` | Power consumption in Zone 3 (kW) |

- **Records**: ~52,418 rows (10-minute intervals across a full year)
- **Source**: Tétouan City Power Consumption Dataset (Morocco, 2017)

---

##  Getting Started

### 1. Clone or Download the Project

```bash
git clone <https://github.com/anujjyadav/Cloudcredits>
cd "Energy Consumption Analysis"
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook Energy_Consumption_Analysis.ipynb
```

---

##  Models Used

| Model | Type | Notes |
|---|---|---|
| Linear Regression | Baseline | Fast, interpretable |
| Ridge Regression | Regularized | Handles multicollinearity |
| Random Forest | Ensemble | Captures non-linearity |
| XGBoost | Gradient Boosting | Best overall performance |

---

##  Evaluation Metrics

- **RMSE** — Root Mean Squared Error (penalizes large errors)
- **MAE** — Mean Absolute Error (robust to outliers)
- **R²** — Coefficient of Determination (goodness of fit)

---

##  Key Findings

- Energy consumption follows strong **daily and weekly seasonality**
- **Temperature** and **hour of day** are the most significant predictors
- **XGBoost** consistently achieves the highest R² (>0.96) across all zones
- Solar diffuse flows correlate with daytime consumption peaks

---

##  Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

---

##  License

This project is for educational purposes. Dataset sourced from the UCI Machine Learning Repository.
