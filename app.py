"""
Diabetes Screening Flask Application
=====================================
Web interface for diabetes risk screening using the trained screening model.

Risk Tiers:
- Low Risk (< 0.30): Annual wellness check
- Medium Risk (0.30 - 0.60): Lifestyle counseling, 6-month follow-up
- High Risk (>= 0.60): Clinical evaluation recommended

Features Used: Age, Gender, BMI, Hypertension, Heart Disease, Diet, Physical Activity
"""

import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# =============================================================================
# CONFIGURATION & LOADING
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'diabetes_screening_model.pkl')

# Load Model Package
print(f"Loading screening model from: {MODEL_PATH}")
with open(MODEL_PATH, 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
scaler = model_package['scaler']
feature_names = model_package['feature_names']
numerical_cols = model_package['numerical_cols']
risk_thresholds = model_package['risk_thresholds']

# Initialize Explainer
try:
    from src.explainability import RiskExplainer
    print("✅ Initializing Risk Explainer...")
    if 'feature_means' in model_package:
        explainer = RiskExplainer(model, feature_names, model_package['feature_means'])
        print("   Explainer ready.")
    else:
        print("⚠️ Warning: 'feature_means' not found in model package. Explainability disabled.")
        explainer = None
except Exception as e:
    print(f"⚠️ Error initializing explainer: {e}")
    explainer = None

print(f"✅ Loaded {model_package['model_name']} model")
print(f"   Features: {len(feature_names)}")
print(f"   Risk thresholds: Low < {risk_thresholds['low']}, High >= {risk_thresholds['high']}")

# Categorical features for encoding
CATEGORICAL_FEATURES = ['gender', 'Age_Band', 'BMI_Category', 'Diet', 'PhysicalActivity']

# =============================================================================
# FEATURE ENGINEERING (Must match training)
# =============================================================================

def engineer_features(df_in):
    """
    Create clinically meaningful features for diabetes screening.
    MUST match the function in diabetes_screening_model.py exactly.
    """
    df = df_in.copy()
    
    # Age-Based Features
    age_bins = [0, 30, 45, 60, 120]
    age_labels = ['young', 'middle', 'senior', 'elderly']
    df['Age_Band'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    df['Age_Risk_Flag'] = (df['age'] >= 45).astype(int)
    
    # BMI-Based Features
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['underweight', 'normal', 'overweight', 'obese']
    df['BMI_Category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)
    df['Obesity_Flag'] = (df['bmi'] >= 30).astype(int)
    
    # Interaction Features
    df['Age_BMI_Interaction'] = df['age'] * df['bmi']
    
    # Cardiovascular Risk Score
    df['Cardio_Risk_Score'] = df['hypertension'] + df['heart_disease']
    
    # Lifestyle Risk Scoring
    diet_map = {'Healthy': 0, 'Mixed': 1, 'Unhealthy': 2}
    df['Diet_Score'] = df['Diet'].map(diet_map).fillna(1)
    
    activity_map = {'Active': 0, 'Moderately Active': 1, 'Sedentary': 2}
    df['Activity_Score'] = df['PhysicalActivity'].map(activity_map).fillna(1)
    
    df['Lifestyle_Risk_Score'] = df['Diet_Score'] + df['Activity_Score']
    
    return df


def assign_risk_tier(probability):
    """
    Assign risk tier based on probability.
    
    Returns dict with tier info and recommendations.
    """
    if probability < risk_thresholds['low']:
        return {
            'tier': 'Low Risk',
            'tier_code': 'low',
            'tier_class': 'risk-low',
            'emoji': '🟢',
            'description': 'Low likelihood of diabetes based on current risk factors.',
            'recommendation': 'Maintain healthy lifestyle. Annual wellness check recommended.'
        }
    elif probability < risk_thresholds['high']:
        return {
            'tier': 'Medium Risk',
            'tier_code': 'medium',
            'tier_class': 'risk-medium',
            'emoji': '🟡',
            'description': 'Moderate risk factors present. Preventive action advised.',
            'recommendation': 'Lifestyle modifications suggested. Consider follow-up in 6 months.'
        }
    else:
        return {
            'tier': 'High Risk',
            'tier_code': 'high',
            'tier_class': 'risk-high',
            'emoji': '🔴',
            'description': 'Elevated risk profile detected.',
            'recommendation': 'Clinical evaluation strongly recommended. Please consult a healthcare provider.'
        }


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Received prediction request: {data}")

        # 1. Create DataFrame from input
        input_df = pd.DataFrame([{
            'gender': data.get('gender'),
            'age': float(data.get('age')),
            'hypertension': int(data.get('hypertension')),
            'heart_disease': int(data.get('heart_disease')),
            'bmi': float(data.get('bmi')),
            'Diet': data.get('diet'),
            'PhysicalActivity': data.get('physical_activity')
        }])

        # 2. Apply Feature Engineering
        input_df = engineer_features(input_df)

        # 3. One-Hot Encode
        input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int)

        # 4. Align Columns with Model
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        input_encoded = input_encoded[feature_names]

        # 5. Scale Numerical Features
        if numerical_cols:
            input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

        # 6. Predict
        probability = model.predict_proba(input_encoded)[0][1]

        # 7. Assign Risk Tier
        risk_info = assign_risk_tier(probability)

        # 8. Calculate Explanations
        impact_factors = []
        if explainer:
            try:
                impact_factors = explainer.explain(input_encoded)
                # Take top 4 factors
                impact_factors = impact_factors[:4]
            except Exception as e:
                print(f"Explanation error: {e}")

        print(f"Prediction: {risk_info['tier']} ({probability:.4f})")

        return jsonify({
            'status': 'success',
            'prediction': int(probability >= 0.5),
            'probability': float(probability),
            'result': f"{risk_info['emoji']} {risk_info['tier']}",
            'tier': risk_info['tier'],
            'tier_code': risk_info['tier_code'],
            'tier_class': risk_info['tier_class'],
            'description': risk_info['description'],
            'advice': risk_info['recommendation'],
            'impact_factors': impact_factors
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=5000)
