 Customer Churn Prediction
# Telco Customer Dataset | Internship ML Project

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-orange?logo=xgboost)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-blue?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

# Project Overview

A complete end-to-end **machine learning pipeline** to predict whether a telecom customer will **churn** (leave the service) based on their historical usage, billing, and subscription data.

This project follows a structured **10-step internship framework**:

| Step | Description |
|------|-------------|
| 1 | Define the Problem |
| 2 | Collect and Prepare Data |
| 3 | Exploratory Data Analysis (EDA) |
| 4 | Feature Engineering |
| 5 | Split the Data |
| 6 | Choose a Model |
| 7 | Train the Model |
| 8 | Evaluate the Model |
| 9 | Improve the Model (Hyperparameter Tuning) |
| 10 | Deploy the Model |

---

# Problem Statement

**Type:** Binary Classification  
**Target:** `Churn Value` → `1` (Churned) / `0` (Stayed)  
**Business Question:** *Which customers are at risk of leaving, and why?*

> Acquiring a new customer costs **5–7× more** than retaining an existing one.  
> A **5% improvement** in retention can boost profits by **25–95%**.

---

#  Project Structure

```
Customer-Churn-Prediction/
│
├──  Customer_Churn_Prediction.ipynb   ← Main Jupyter Notebook 
├──  churn_analysis.py                 ← Standalone Python pipeline script
├──  app.py                            ← Flask web application (Step 10)
├──  Telco_customer_churn.xlsx         ← Raw dataset (7,043 records)
│
├──  Saved Model Artifacts
│   ├── best_model.pkl                   ← Tuned XGBoost model
│   ├── scaler.pkl                       ← Fitted StandardScaler
│   └── feature_columns.pkl             ← List of 25 feature names
│
├──  plots/                            ← 17 auto-generated visualizations
│   ├── 00_summary_dashboard.png        ← Full results dashboard
│   ├── 01_churn_distribution.png
│   ├── 02_churn_by_categories.png
│   ├── 03_tenure_analysis.png
│   ├── 04_charges_analysis.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_services_churn_heatmap.png
│   ├── 07_cltv_score.png
│   ├── 08_engineered_features.png
│   ├── 09_smote_balance.png
│   ├── 10_roc_curves.png
│   ├── 11_confusion_matrices.png
│   ├── 12_model_comparison.png
│   ├── 13_prob_distribution.png
│   ├── 14_cross_validation.png
│   ├── 15_feature_importance.png
│   └── 16_threshold_analysis.png
│
└──  templates/
    └── index.html                       ← Flask  prediction UI
```

---

# Dataset

**Source:** Telco Customer Churn Dataset  
**File:** `Telco_customer_churn.xlsx`

| Property | Value |
|----------|-------|
| Records | 7,043 customers |
| Features | 33 columns |
| Target | `Churn Value` (0 = Stay, 1 = Churn) |
| Churn Rate | ~26.5% |

**Key columns used:**

| Column | Type | Description |
|--------|------|-------------|
| `Tenure Months` | Numeric | How long the customer has been with the company |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `Monthly Charges` | Numeric | Monthly billing amount |
| `Total Charges` | Numeric | Total amount charged |
| `Internet Service` | Categorical | DSL / Fiber optic / No |
| `Payment Method` | Categorical | Electronic check / Mailed check / etc. |
| `Online Security` | Categorical | Yes / No / No internet service |
| `CLTV` | Numeric | Customer Lifetime Value score |

---

#  Installation & Setup

# Prerequisites
- Python 3.10+
- pip

# 1. Clone the repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install pandas openpyxl numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib flask jupyter
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

---

# How to Run

### Option 1 — Jupyter Notebook (Recommended)
```bash
jupyter notebook Customer_Churn_Prediction.ipynb
```
- Open in browser, then press **`Kernel → Restart & Run All`**
- All 17 plots will render **inline** inside the notebook

# Option 2 — Flask Web Application
```bash
python app.py
```
Then open your browser at: **http://localhost:5000**

- Fill in customer details in the form
- Click **Predict Churn Probability**
- See real-time churn probability, risk level, and retention suggestions

# Option 3 — Standalone Python Script
```bash
python churn_analysis.py
```
- Runs the full 10-step pipeline in the terminal
- Prints all model metrics
- Saves all plots to the `plots/` folder

---

# Methodology

# Step 2 — Data Preparation
- Dropped non-predictive columns (CustomerID, geo-coordinates, Churn Label, Churn Reason)
- Renamed columns (removed spaces)
- Converted `TotalCharges` from object → float
- Imputed missing values: **median** for numeric, **mode** for categorical
- Result: **zero null values** after cleaning

# Step 3 — Exploratory Data Analysis
Generated **7 EDA visualizations** covering:
- Churn distribution (pie + bar)
- Churn rate by 10 categorical features
- Tenure analysis (histogram + KDE + boxplot)
- Monthly & Total Charges (violin + density)
- Full correlation heatmap
- Services vs Churn heatmap
- CLTV & Churn Score distributions

# Step 4 — Feature Engineering
Created **5 new features** from existing data:

| Feature | Formula | Insight |
|---------|---------|---------|
| `tenure_group` | Bin tenure into 6 groups | New vs loyal customers |
| `avg_monthly_spend` | TotalCharges / (tenure + 1) | True per-month cost |
| `service_count` | Count of 6 add-on services | Engagement level |
| `has_phone_internet` | Phone=Yes AND Internet≠No | Multi-service flag |
| `charge_per_service` | MonthlyCharges / (service_count + 1) | Value per service |

**Final feature count: 25**

# Step 5 — Data Splitting & Balancing
- **80% / 20%** stratified train/test split
- `StandardScaler` applied to normalize all features
- **SMOTE** (Synthetic Minority Oversampling Technique) applied to training set to fix the ~73:27 class imbalance → balanced to 50:50

# Steps 6 & 7 — Models Trained

| Model | Reason for Selection |
|-------|---------------------|
| Logistic Regression | Fast interpretable baseline |
| Random Forest | Ensemble method, handles non-linearity |
| **XGBoost**  | Best performer, gradient boosting |
| SVM (RBF) | Effective in high-dimensional feature space |

# Step 8 — Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.80 | 0.63 | 0.77 | 0.69 | 0.85 |
| Random Forest | 0.82 | 0.68 | 0.75 | 0.71 | 0.87 |
| XGBoost | 0.82 | 0.69 | 0.74 | 0.71 | 0.87 |
| SVM | 0.80 | 0.63 | 0.78 | 0.70 | 0.85 |
| **Tuned XGBoost** | **0.83** | **0.70** | **0.76** | **0.73** | **0.88+** |

# Step 9 — Hyperparameter Tuning

**Method:** `RandomizedSearchCV`  
**Iterations:** 40  
**CV:** 5-Fold Stratified  
**Objective:** Maximize ROC-AUC

**Best parameters found:**
```python
{
    'learning_rate':    0.1,
    'max_depth':        6,
    'n_estimators':     200,
    'subsample':        0.9,
    'colsample_bytree': 0.7,
    'gamma':            0.5,
    'reg_alpha':        1.0,
    'reg_lambda':       1.0,
    'min_child_weight': 1,
}
```

**5-Fold Cross-Validation ROC-AUC:** ~0.93 ± 0.01 *(on SMOTE-balanced training data)*

# Step 10 — Deployment
- Model saved as `best_model.pkl` using **joblib**
- Flask web app with a **LinkedIn-style** professional UI
- Real-time predictions with churn probability %, risk level (Low / Medium / High), and retention action suggestion

---

# Key Visualizations

| Plot | Description |
|------|-------------|
| `00_summary_dashboard.png` | Dark-mode results dashboard with all key outputs |
| `01_churn_distribution.png` | Pie + bar chart of churn proportion |
| `02_churn_by_categories.png` | Churn rate across 10 categorical features |
| `03_tenure_analysis.png` | Histogram, KDE, boxplot for tenure |
| `04_charges_analysis.png` | Violin + density for Monthly & Total Charges |
| `05_correlation_heatmap.png` | Full feature correlation matrix |
| `10_roc_curves.png` | ROC curves for all 4 models overlaid |
| `11_confusion_matrices.png` | Confusion matrices for all models |
| `14_cross_validation.png` | 5-Fold CV bar chart for tuned XGBoost |
| `15_feature_importance.png` | Feature importance bar + top-10 pie chart |
| `16_threshold_analysis.png` | Precision / Recall / F1 vs decision threshold |

---

# Key Business Insights

| Finding | Insight | Recommended Action |
|---------|---------|-------------------|
| Month-to-month contracts | Churn 3–4× more than 2-year | Incentivise annual/2-year upgrades |
| First 12 months tenure | Highest churn risk period | Onboarding loyalty program |
| Fiber optic internet | Higher churn than DSL | Premium fiber support service |
| No online security | Much higher churn | Bundle security in standard packages |
| Electronic check payment | Highest churn of all payment types | Migrate customers to auto-pay |
| Senior citizens | Churn more than non-seniors | Dedicated senior support line |

---

# Web Application

The Flask app provides a professional prediction interface:

- **URL:** `http://localhost:5000`
- **Input:** Customer profile form (24 fields across 3 sections)
- **Output:**
  - Circular probability gauge (0–100%)
  - Churn / Stay verdict banner
  - Risk level tag: 🟢 Low / 🟡 Medium / 🔴 High
  - Retention suggestion text
  - Metric breakdown (churn %, retention %, decision)

---

# Dependencies

```
pandas>=1.5
numpy>=1.23
openpyxl>=3.0
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.1
xgboost>=1.7
imbalanced-learn>=0.10
joblib>=1.2
flask>=2.2
jupyter>=1.0
notebook>=6.5
```

---

# requirements.txt

```
pandas
numpy
openpyxl
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
joblib
flask
jupyter
notebook
```

---

# Author

**Internship ML Project**  
Built using Python 3.10, Scikit-Learn, XGBoost, Flask, and Jupyter Notebook.

---

# License

This project is licensed under the **MIT License** — free to use for educational and internship purposes.

---

# Acknowledgements

- **Dataset:** IBM Telco Customer Churn Dataset
- **Libraries:** Scikit-Learn, XGBoost, imbalanced-learn, Flask, Matplotlib, Seaborn, Pandas

---

> *"The goal is not to predict the future, but to be prepared for it."*
