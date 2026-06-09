
"""

  Customer Churn Prediction -- Complete ML Pipeline
  Internship Project | Telco Customer Churn Dataset
============================================================
Steps:
  1. Define the Problem
  2. Collect and Prepare Data
  3. Exploratory Data Analysis (EDA)
  4. Feature Engineering
  5. Split the Data
  6. Choose a Model
  7. Train the Model
  8. Evaluate the Model
  9. Improve the Model (Hyperparameter Tuning)
 10. Deploy the Model (save artifacts for Flask app)
============================================================
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', palette='muted')

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)



print("=" * 60)
print("STEP 1: DEFINE THE PROBLEM")
print("=" * 60)
print("""
Objective  : Predict whether a customer will churn (leave the service).
Target     : 'Churn' column  ->  Yes = 1  /  No = 0
Type       : Binary Classification
Business   : Identify at-risk customers so the company can
             proactively offer retention incentives.
""")


print("=" * 60)
print("STEP 2: COLLECT AND PREPARE DATA")
print("=" * 60)

df = pd.read_excel('Telco_customer_churn.xlsx')
print(f"\nDataset loaded  ->  Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):\n{list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")


df.drop(columns=['CustomerID'], errors='ignore', inplace=True)
        
df.columns = df.columns.str.strip()


churn_col = [c for c in df.columns if 'churn' in c.lower()][0]
print(f"\nTarget column detected: '{churn_col}'")


if df[churn_col].dtype == object:
    df[churn_col] = df[churn_col].map({'Yes': 1, 'No': 0,
                                        'True': 1, 'False': 0,
                                        True: 1, False: 0})

df.rename(columns={churn_col: 'Churn'}, inplace=True)


if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

for c in num_cols:
    df[c].fillna(df[c].median(), inplace=True)
for c in cat_cols:
    df[c].fillna(df[c].mode()[0], inplace=True)

print(f"\nAfter cleaning — Missing Values:\n{df.isnull().sum().sum()} total nulls")
print(f"\nTarget distribution:\n{df['Churn'].value_counts()}")
print(f"Churn rate: {df['Churn'].mean()*100:.2f}%")


print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

PALETTE = {'Churn': '#E74C3C', 'No Churn': '#2ECC71',
           0: '#2ECC71', 1: '#E74C3C'}


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Churn Distribution', fontsize=16, fontweight='bold', y=1.02)

churn_counts = df['Churn'].value_counts()
labels = ['No Churn', 'Churn']
colors = ['#2ECC71', '#E74C3C']

axes[0].pie(churn_counts, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
axes[0].set_title('Proportion of Churn', fontweight='bold')

axes[1].bar(labels, churn_counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[1].set_title('Count of Churn vs No Churn', fontweight='bold')
axes[1].set_ylabel('Number of Customers')
for i, v in enumerate(churn_counts.values):
    axes[1].text(i, v + 30, str(v), ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/01_churn_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Plot 1 saved: 01_churn_distribution.png")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Churn by Service Features', fontsize=16, fontweight='bold')

for col, ax, title in [
    ('Contract' if 'Contract' in df.columns else cat_cols[0], axes[0], 'Churn by Contract Type'),
    ('InternetService' if 'InternetService' in df.columns else cat_cols[1], axes[1], 'Churn by Internet Service'),
]:
    if col in df.columns:
        ct = df.groupby(col)['Churn'].mean().reset_index()
        bars = ax.bar(ct[col], ct['Churn'] * 100, color=sns.color_palette('Set2', len(ct)))
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Churn Rate (%)')
        ax.set_xlabel(col)
        for bar, val in zip(bars, ct['Churn'] * 100):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/02_churn_by_contract.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Plot 2 saved: 02_churn_by_contract.png")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Tenure Analysis', fontsize=16, fontweight='bold')

if 'tenure' in df.columns:
    for churn_val, label, color in [(0, 'No Churn', '#2ECC71'), (1, 'Churn', '#E74C3C')]:
        subset = df[df['Churn'] == churn_val]['tenure']
        axes[0].hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
    axes[0].set_title('Tenure Distribution by Churn', fontweight='bold')
    axes[0].set_xlabel('Tenure (months)')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    sns.boxplot(data=df, x='Churn', y='tenure', palette=['#2ECC71', '#E74C3C'], ax=axes[1])
    axes[1].set_title('Tenure Boxplot by Churn', fontweight='bold')
    axes[1].set_xlabel('Churn (0=No, 1=Yes)')
    axes[1].set_ylabel('Tenure (months)')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/03_tenure_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Plot 3 saved: 03_tenure_distribution.png")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Charges Analysis by Churn', fontsize=16, fontweight='bold')

for col, ax, title in [
    ('MonthlyCharges', axes[0], 'Monthly Charges vs Churn'),
    ('TotalCharges', axes[1], 'Total Charges vs Churn'),
]:
    if col in df.columns:
        sns.violinplot(data=df, x='Churn', y=col, palette=['#2ECC71', '#E74C3C'], ax=ax, inner='box')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Churn (0=No, 1=Yes)')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/04_charges_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Plot 4 saved: 04_charges_boxplot.png")


fig, ax = plt.subplots(figsize=(14, 10))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 9})
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Plot 5 saved: 05_correlation_heatmap.png")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Churn by Payment Method & Senior Citizen Status', fontsize=14, fontweight='bold')

if 'PaymentMethod' in df.columns:
    pm = df.groupby('PaymentMethod')['Churn'].mean().sort_values(ascending=False).reset_index()
    bars = axes[0].barh(pm['PaymentMethod'], pm['Churn'] * 100,
                        color=sns.color_palette('RdYlGn_r', len(pm)))
    axes[0].set_title('Churn Rate by Payment Method', fontweight='bold')
    axes[0].set_xlabel('Churn Rate (%)')
    for bar, val in zip(bars, pm['Churn'] * 100):
        axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

if 'SeniorCitizen' in df.columns:
    sc = df.groupby('SeniorCitizen')['Churn'].mean().reset_index()
    sc['Label'] = sc['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'})
    axes[1].bar(sc['Label'], sc['Churn'] * 100, color=['#3498DB', '#E67E22'], edgecolor='white', linewidth=2)
    axes[1].set_title('Churn Rate by Senior Citizen', fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)')
    for i, val in enumerate(sc['Churn'] * 100):
        axes[1].text(i, val + 0.5, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/06_payment_senior_churn.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Plot 6 saved: 06_payment_senior_churn.png")


print("\n" + "=" * 60)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 60)

df_fe = df.copy()


if 'tenure' in df_fe.columns:
    df_fe['tenure_group'] = pd.cut(df_fe['tenure'],
                                    bins=[0, 12, 24, 36, 48, 60, 200],
                                    labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61+'])
    print("  [OK] Created: tenure_group")


if 'TotalCharges' in df_fe.columns and 'tenure' in df_fe.columns:
    df_fe['avg_monthly_spend'] = df_fe['TotalCharges'] / (df_fe['tenure'] + 1)
    print("  [OK] Created: avg_monthly_spend")


service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
existing_services = [c for c in service_cols if c in df_fe.columns]
if existing_services:
    for c in existing_services:
        df_fe[c + '_bin'] = (df_fe[c].str.lower() == 'yes').astype(int)
    df_fe['service_count'] = df_fe[[c + '_bin' for c in existing_services]].sum(axis=1)
    df_fe.drop(columns=[c + '_bin' for c in existing_services], inplace=True)
    print(f"  [OK] Created: service_count (from {existing_services})")


if 'PhoneService' in df_fe.columns and 'InternetService' in df_fe.columns:
    df_fe['has_phone_internet'] = (
        (df_fe['PhoneService'].str.lower() == 'yes') &
        (df_fe['InternetService'].str.lower() != 'no')
    ).astype(int)
    print("  [OK] Created: has_phone_internet")

y = df_fe['Churn'].astype(int)
X = df_fe.drop(columns=['Churn'])


cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
for c in cat_features:
    X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    print(f"  [OK] Encoded: {c}")

print(f"\nFinal feature set: {X.shape[1]} features, {X.shape[0]} samples")
print(f"Features: {list(X.columns)}")


print("\n" + "=" * 60)
print("STEP 5: SPLIT THE DATA")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train set: {X_train.shape[0]} samples")
print(f"  Test  set: {X_test.shape[0]} samples")
print(f"  Train churn rate: {y_train.mean()*100:.2f}%")
print(f"  Test  churn rate: {y_test.mean()*100:.2f}%")


scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
print(f"\n  After SMOTE → Train: {X_train_sm.shape[0]} samples")
print(f"  Class balance after SMOTE: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")


print("\n" + "=" * 60)
print("STEP 6 & 7: CHOOSE AND TRAIN MODELS")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost':             XGBClassifier(n_estimators=200, random_state=42,
                                         use_label_encoder=False, eval_metric='logloss',
                                         n_jobs=-1, verbosity=0),
    'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]
    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_proba),
    }
    print(f"\n  ── {name}")
    print(f"     Accuracy : {results[name]['accuracy']:.4f}")
    print(f"     Precision: {results[name]['precision']:.4f}")
    print(f"     Recall   : {results[name]['recall']:.4f}")
    print(f"     F1-Score : {results[name]['f1']:.4f}")
    print(f"     ROC-AUC  : {results[name]['roc_auc']:.4f}")


print("\n" + "=" * 60)
print("STEP 8: EVALUATE THE MODEL")
print("=" * 60)


fig, ax = plt.subplots(figsize=(10, 7))
colors_roc = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']
for (name, res), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, color=color, lw=2.5,
            label=f"{name} (AUC = {res['roc_auc']:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — All Models', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/07_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Plot 7 saved: 07_roc_curves.png")


best_name = max(results, key=lambda n: results[n]['roc_auc'])
best_res  = results[best_name]
cm = confusion_matrix(y_test, best_res['y_pred'])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=2, linecolor='white',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'], ax=ax,
            annot_kws={'size': 16, 'weight': 'bold'})
ax.set_title(f'Confusion Matrix — {best_name}', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/08_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Plot 8 saved: 08_confusion_matrix.png")

print(f"\nBest Model: {best_name}")
print(classification_report(y_test, best_res['y_pred'], target_names=['No Churn', 'Churn']))


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics_df = pd.DataFrame({
    'Model':     list(results.keys()),
    'Accuracy':  [r['accuracy']  for r in results.values()],
    'F1-Score':  [r['f1']        for r in results.values()],
    'ROC-AUC':   [r['roc_auc']   for r in results.values()],
    'Recall':    [r['recall']    for r in results.values()],
}).set_index('Model')

metrics_df[['Accuracy', 'F1-Score', 'ROC-AUC']].plot(kind='bar', ax=axes[0],
    color=['#3498DB', '#2ECC71', '#E74C3C'], edgecolor='white', linewidth=1.2)
axes[0].set_title('Accuracy / F1 / ROC-AUC', fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0.5, 1.0)
axes[0].tick_params(axis='x', rotation=20)
axes[0].legend(loc='lower right')

metrics_df['Recall'].plot(kind='bar', ax=axes[1], color='#9B59B6', edgecolor='white', linewidth=1.2)
axes[1].set_title('Recall (Sensitivity)', fontweight='bold')
axes[1].set_ylabel('Recall Score')
axes[1].set_ylim(0.5, 1.0)
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/09_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Plot 9 saved: 09_model_comparison.png")


print("\n" + "=" * 60)
print("STEP 9: IMPROVE THE MODEL")
print("=" * 60)

print("  Running RandomizedSearchCV on XGBoost (this may take a minute)...")

xgb_param_grid = {
    'n_estimators':       [100, 200, 300, 400],
    'max_depth':          [3, 4, 5, 6, 7],
    'learning_rate':      [0.01, 0.05, 0.1, 0.2],
    'subsample':          [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':   [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma':              [0, 0.1, 0.2, 0.5],
    'reg_alpha':          [0, 0.1, 0.5, 1.0],
    'reg_lambda':         [1.0, 1.5, 2.0],
    'min_child_weight':   [1, 3, 5],
}

xgb_base = XGBClassifier(random_state=42, use_label_encoder=False,
                           eval_metric='logloss', n_jobs=-1, verbosity=0)

rs_cv = RandomizedSearchCV(
    xgb_base, xgb_param_grid, n_iter=40,
    scoring='roc_auc', cv=StratifiedKFold(n_splits=5),
    random_state=42, n_jobs=-1, verbose=0
)
rs_cv.fit(X_train_sm, y_train_sm)

best_xgb = rs_cv.best_estimator_
print(f"  Best params: {rs_cv.best_params_}")
print(f"  Best CV ROC-AUC: {rs_cv.best_score_:.4f}")

y_pred_tuned  = best_xgb.predict(X_test_sc)
y_proba_tuned = best_xgb.predict_proba(X_test_sc)[:, 1]

print(f"\n  ── Tuned XGBoost on Test Set")
print(f"     Accuracy : {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"     Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"     Recall   : {recall_score(y_test, y_pred_tuned):.4f}")
print(f"     F1-Score : {f1_score(y_test, y_pred_tuned):.4f}")
print(f"     ROC-AUC  : {roc_auc_score(y_test, y_proba_tuned):.4f}")

print(classification_report(y_test, y_pred_tuned, target_names=['No Churn', 'Churn']))


cv_scores = cross_val_score(best_xgb, X_train_sm, y_train_sm,
                             cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
print(f"  5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

importances = best_xgb.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)
top20 = feat_imp.tail(20)

fig, ax = plt.subplots(figsize=(10, 8))
colors_imp = sns.color_palette('RdYlGn', len(top20))
bars = ax.barh(top20.index, top20.values, color=colors_imp, edgecolor='white', linewidth=1.2)
ax.set_title('Top 20 Feature Importances (Tuned XGBoost)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=12)
for bar, val in zip(bars, top20.values):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/10_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Plot 10 saved: 10_feature_importance.png")


print("\n" + "=" * 60)
print("STEP 10: SAVE MODEL ARTIFACTS FOR DEPLOYMENT")
print("=" * 60)

joblib.dump(best_xgb,       'best_model.pkl')
joblib.dump(scaler,          'scaler.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')

print("  [OK] Saved: best_model.pkl")
print("  [OK] Saved: scaler.pkl")
print("  [OK] Saved: feature_columns.pkl")
print("\n  Run  ->  python app.py  to launch the Flask prediction web app.")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE — All plots saved to /plots folder")
print("=" * 60)
