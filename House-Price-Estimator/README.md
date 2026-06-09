#  House Price Estimator

A machine learning project that predicts house prices based on features like location, area, bedrooms, bathrooms, and more. Built using Python, pandas, scikit-learn, and XGBoost. The project covers the complete machine learning workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and prediction.



#  Project Files

| File | Description |
|------|-------------|
| `House Price Prediction Dataset.csv` | The dataset used for training and testing |
| `house_price_estimator.ipynb` | Main notebook with all steps and results |
| `house_price_model.pkl` | Trained XGBoost model ready to use |
| `house_price_scaler.pkl` | Scaler used to normalize input data |
| `01_price_distribution.png` | Price distribution chart |
| `02_eda_relationships.png` | Feature vs price relationships |
| `03_boxplots_categorical.png` | Price spread by category |
| `04_feature_distributions.png` | Distribution of all features |
| `05_correlation_heatmap.png` | Correlation between features |
| `06_garage_floors_impact.png` | Garage and floors effect on price |
| `07_bedrooms_bathrooms.png` | Bedrooms and bathrooms vs price |
| `08_train_test_split.png` | Train and test split ratio |
| `09_model_comparison.png` | Comparison of all models |
| `10_model_diagnostics.png` | Actual vs predicted prices |
| `11_feature_importance.png` | Most important features |
| `12_xgb_final_diagnostics.png` | XGBoost model diagnostics |
| `13_final_model_ranking.png` | Final ranking of all models |
| `14_sample_predictions.png` | Sample predictions chart |


# Repository Structure

House-Price-Estimator/
│
├── house_price_estimator.ipynb
├── house_price_model.pkl
├── house_price_scaler.pkl
├── House Price Prediction Dataset.csv
├── README.md
│
├── images/
│   ├── 01_price_distribution.png
│   ├── 02_eda_relationships.png
│   ├── ...
│   └── 14_sample_predictions.png

---

# Dataset

- **File:** `House Price Prediction Dataset.csv`
- **Rows:** 2000 houses
- **Columns:** 10
- **Target:** `Price` (house sale price in USD)

| Column | Type | Values |
|--------|------|--------|
| Area | Number | Square footage of the house |
| Bedrooms | Number | 1 to 5 |
| Bathrooms | Number | 1 to 4 |
| Floors | Number | 1, 2, or 3 |
| YearBuilt | Number | 1900 to 2023 |
| Location | Text | Downtown, Urban, Suburban, Rural |
| Condition | Text | Excellent, Good, Fair, Poor |
| Garage | Text | Yes or No |
| Price | Number | Target variable (USD) |

---

# Steps in the Notebook

# Step 1 — Define the Problem
Identify that this is a **regression problem** where we predict a continuous number (house price).

# Step 2 — Load and Prepare Data
Read the CSV file, check for missing values, remove the unnecessary ID column, and confirm the data is clean.

# Step 3 — Exploratory Data Analysis (EDA)
Create 7 charts to understand:
- How prices are distributed
- Which locations have higher prices
- How area, condition, garage, bedrooms, and year built affect price
- Correlations between all features

# Step 4 — Feature Engineering
Create 13 new features from existing ones to help the model learn better patterns:
- `HouseAge` — how old the house is
- `TotalRooms` — bedrooms + bathrooms
- `AreaPerRoom` — space per room
- `FloorArea` — area × floors
- `QualityScore` — combined score from location, condition, garage
- `LocationMeanPrice` — average price of the neighborhood
- One-hot encoded columns for Location, Condition, and Garage

# Step 5 — Split the Data
Split into **80% training** and **20% testing** sets. Apply `StandardScaler` to normalize all values.

# Step 6 — Choose Models
Select 8 different machine learning algorithms to compare:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Decision Tree
6. Random Forest
7. Gradient Boosting
8. XGBoost

# Step 7 — Train the Models
Train all 8 models on the training data and make predictions on the test data.

# Step 8 — Evaluate the Models
Compare all models using 3 metrics:
- **R² Score** — higher is better (1.0 = perfect)
- **RMSE** — lower is better (average error in dollars)
- **MAE** — lower is better (mean absolute error)

# Step 9 — Improve the Model
Tune the best model (XGBoost) using **GridSearchCV** with 5-fold cross-validation to find the best settings automatically.

# Step 10 — Deploy the Model
Save the trained model and scaler as `.pkl` files. A `predict_house_price()` function is created so anyone can predict a price by entering house details.

---

# How to Run

# Run the Full Notebook

**Step 1** — Open a terminal or PowerShell and go to the project folder:
```
cd "d:\House Price Estimator"
```

**Step 2** — Launch Jupyter Notebook:
```
jupyter notebook
```

**Step 3** — A browser window will open. Click on `house_price_estimator.ipynb`.

**Step 4** — All outputs are already saved. To re-run everything fresh:
- Go to **Kernel → Restart & Run All**

---

# Run in VS Code

1. Open VS Code
2. Click **File → Open Folder**
3. Select `d:\House Price Estimator`
4. Open `house_price_estimator.ipynb`
5. Click **Run All** at the top

---

# Predict a New House Price

Use the saved model directly in any Python script:

```python
import joblib
import numpy as np
import pandas as pd

model  = joblib.load(r"d:\House Price Estimator\house_price_model.pkl")
scaler = joblib.load(r"d:\House Price Estimator\house_price_scaler.pkl")
```

Then call the `predict_house_price()` function defined in Step 10 of the notebook:

```python
price = predict_house_price(
    area       = 2500,
    bedrooms   = 3,
    bathrooms  = 2,
    floors     = 2,
    year_built = 2010,
    location   = "Suburban",
    condition  = "Good",
    garage     = "Yes"
)

print(f"Estimated Price: ${price:,.0f}")
```

---

# Model Results

# Model Results

Multiple regression models were trained and compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

Detailed evaluation metrics and visual comparisons are available in the notebook.
The final model was selected after comparing multiple regression algorithms and evaluating their performance using R², RMSE, and MAE metrics.

# Requirements

Install all required libraries with one command:

```
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost joblib jupyter
```

---

# Key Takeaways

## Key Takeaways

- Performed comprehensive exploratory data analysis (EDA)
- Engineered additional features to improve learning
- Compared multiple regression algorithms
- Evaluated model performance using R², RMSE, and MAE
- Identified important factors influencing house prices

---

# Project Info

- **Type:** Supervised Machine Learning — Regression
- **Language:** Python 3.10
- **Tools:** Jupyter Notebook, scikit-learn, XGBoost, pandas, matplotlib, seaborn
- **Dataset Size:** 2000 samples
- **Best Model:** XGBoost (GridSearchCV tuned, 5-fold cross-validation)


# Author

Anuj Yadav

Data Science Internship Project
CloudCredits Technologies



