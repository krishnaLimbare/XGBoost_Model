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
CATEGORICAL_FEATURES = ['gender', 'Age_Group', 'BMI_Category', 'Diet', 'PhysicalActivity']

# =============================================================================
# FEATURE ENGINEERING (Must match training)
# =============================================================================

def engineer_features(df_in):
    """
    Create clinically meaningful features for diabetes screening.
    MUST match the function in train_model.py exactly.
    """
    df_out = df_in.copy()
    
    # ==========================================================================
    # 1. ENCODE LIFESTYLE FACTORS (Ordinal Encoding for Risk Calc)
    # ==========================================================================
    
    # Diet Score: Unhealthy=2, Mixed=1, Healthy=0
    diet_map = {'Unhealthy': 2, 'Mixed': 1, 'Healthy': 0}
    df_out['Diet_Score'] = df_out['Diet'].map(diet_map).fillna(1) 
    
    # Activity Score: Sedentary=2, Moderately Active=1, Active=0
    activity_map = {'Sedentary': 2, 'Moderately Active': 1, 'Active': 0}
    df_out['Activity_Score'] = df_out['PhysicalActivity'].map(activity_map).fillna(1)
    
    # Lifestyle Risk Score (0-4)
    df_out['Lifestyle_Risk_Score'] = df_out['Diet_Score'] + df_out['Activity_Score']

    # ==========================================================================
    # 2. AGE & BMI FEATURES
    # ==========================================================================
    
    age_bins = [0, 30, 45, 60, 120]
    age_labels = ['young', 'middle', 'senior', 'elderly']
    df_out['Age_Group'] = pd.cut(df_out['age'], bins=age_bins, labels=age_labels, right=False)
    
    df_out['Age_Risk_Flag'] = (df_out['age'] >= 45).astype(int)
    
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['underweight', 'normal', 'overweight', 'obese']
    df_out['BMI_Category'] = pd.cut(df_out['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)
    
    df_out['Obesity_Flag'] = (df_out['bmi'] >= 30).astype(int)
    
    # ==========================================================================
    # 3. COMPOSITE RISK SCORES
    # ==========================================================================
    
    # Cardiovascular Risk Score (0-2)
    df_out['Cardiovascular_Risk'] = df_out['hypertension'] + df_out['heart_disease']
    
    # Genetic Risk 
    # df_out['Genetic_Risk'] is already passed in input_df, so no need to rename FamilyHistory here 
    # unless we want consistency. In app inputs we pass 'FamilyHistory' and map it to 'Genetic_Risk'
    # Let's ensure 'Genetic_Risk' exists.
    if 'Genetic_Risk' not in df_out.columns and 'FamilyHistory' in df_out.columns:
        df_out['Genetic_Risk'] = df_out['FamilyHistory']
    
    # ==========================================================================
    # 4. INTERACTIONS
    # ==========================================================================
    
    df_out['Age_BMI_Interaction'] = df_out['age'] * df_out['bmi']
    
    # Lifestyle-BMI Interaction
    df_out['Lifestyle_BMI_Interaction'] = df_out['Lifestyle_Risk_Score'] * df_out['bmi']
    
    # Genetic-Age Interaction
    df_out['Genetic_Age_Interaction'] = df_out['Genetic_Risk'] * df_out['age']
    
    # ==========================================================================
    # 5. METABOLIC RISK SCORE
    # ==========================================================================
    
    df_out['Metabolic_Risk_Score'] = (
        df_out['Age_Risk_Flag'] + 
        df_out['Obesity_Flag'] + 
        df_out['Cardiovascular_Risk'] +
        df_out['Lifestyle_Risk_Score'] +
        df_out['Genetic_Risk']
    )
    
    # ==========================================================================
    # 6. METABOLIC STRAIN
    # ==========================================================================
    
    df_out['Metabolic_Strain'] = np.log1p(df_out['age']) * df_out['bmi']
    
    return df_out


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
            'PhysicalActivity': data.get('physical_activity'),
            # New Features
            'waist_circumference_cm': float(data.get('waist_circumference')),
            'sedentary_hours_per_day': float(data.get('sedentary_hours')),
            'sugary_drink_frequency': float(data.get('sugary_drinks')),
            'processed_food_frequency': float(data.get('processed_food')),
            'fruit_veg_frequency': float(data.get('fruit_veg')),
            'FamilyHistory': int(data.get('family_history')),
            'Genetic_Risk': int(data.get('family_history')) # Alias
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
                # Take all relevant factors
                impact_factors = impact_factors[:10]
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
