"""
Flask Web App for Customer Churn Prediction
Run: python app.py  →  http://localhost:5000
"""

from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model    = joblib.load(os.path.join(BASE_DIR, 'best_model.pkl'))
scaler   = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
features = joblib.load(os.path.join(BASE_DIR, 'feature_columns.pkl'))


def encode_input(form_data):
    """Convert form fields into a model-ready feature vector."""
    gender_map   = {'Male': 0, 'Female': 1}
    yn_map       = {'No': 0, 'Yes': 1}
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    internet_map = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
    payment_map  = {
        'Electronic check': 0, 'Mailed check': 1,
        'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3,
    }
    ns_map = {'No': 0, 'Yes': 1, 'No internet service': 0, 'No phone service': 0}

    tenure         = float(form_data.get('tenure', 1))
    monthly_charges = float(form_data.get('MonthlyCharges', 0))
    total_charges   = float(form_data.get('TotalCharges', 0))

    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    service_count = sum(
        1 for c in service_cols if form_data.get(c, 'No') == 'Yes'
    )
    avg_monthly_spend = total_charges / (tenure + 1)
    has_phone_internet = int(
        form_data.get('PhoneService', 'No') == 'Yes' and
        form_data.get('InternetService', 'No') != 'No'
    )

    tenure_val = tenure
    tenure_group_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    if   tenure_val <= 12:  tg = 0
    elif tenure_val <= 24:  tg = 1
    elif tenure_val <= 36:  tg = 2
    elif tenure_val <= 48:  tg = 3
    elif tenure_val <= 60:  tg = 4
    else:                   tg = 5

    raw = {
        'gender':             gender_map.get(form_data.get('gender', 'Male'), 0),
        'SeniorCitizen':      int(form_data.get('SeniorCitizen', 0)),
        'Partner':            yn_map.get(form_data.get('Partner', 'No'), 0),
        'Dependents':         yn_map.get(form_data.get('Dependents', 'No'), 0),
        'tenure':             tenure,
        'PhoneService':       yn_map.get(form_data.get('PhoneService', 'No'), 0),
        'MultipleLines':      ns_map.get(form_data.get('MultipleLines', 'No'), 0),
        'InternetService':    internet_map.get(form_data.get('InternetService', 'No'), 0),
        'OnlineSecurity':     ns_map.get(form_data.get('OnlineSecurity', 'No'), 0),
        'OnlineBackup':       ns_map.get(form_data.get('OnlineBackup', 'No'), 0),
        'DeviceProtection':   ns_map.get(form_data.get('DeviceProtection', 'No'), 0),
        'TechSupport':        ns_map.get(form_data.get('TechSupport', 'No'), 0),
        'StreamingTV':        ns_map.get(form_data.get('StreamingTV', 'No'), 0),
        'StreamingMovies':    ns_map.get(form_data.get('StreamingMovies', 'No'), 0),
        'Contract':           contract_map.get(form_data.get('Contract', 'Month-to-month'), 0),
        'PaperlessBilling':   yn_map.get(form_data.get('PaperlessBilling', 'No'), 0),
        'PaymentMethod':      payment_map.get(form_data.get('PaymentMethod', 'Electronic check'), 0),
        'MonthlyCharges':     monthly_charges,
        'TotalCharges':       total_charges,
        'tenure_group':       tg,
        'avg_monthly_spend':  avg_monthly_spend,
        'service_count':      service_count,
        'has_phone_internet': has_phone_internet,
    }

    row = pd.DataFrame([raw])
    
    for col in features:
        if col not in row.columns:
            row[col] = 0
    row = row[features]
    return row


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        row       = encode_input(form_data)
        row_sc    = scaler.transform(row)
        proba     = model.predict_proba(row_sc)[0][1]
        prediction = int(proba >= 0.5)

        risk = 'Low'
        if proba >= 0.7:    risk = 'High'
        elif proba >= 0.4:  risk = 'Medium'

        return jsonify({
            'prediction': prediction,
            'probability': round(float(proba) * 100, 2),
            'risk_level': risk,
            'message': 'Customer is likely to CHURN' if prediction == 1 else 'Customer is likely to STAY'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
