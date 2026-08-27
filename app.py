"""
Diabetes Screening Flask Application
=====================================
Web interface for diabetes risk screening using the trained screening model.

IMPORTANT - READ BEFORE USING
-----------------------------
This model is trained on SYNTHETIC data whose labels were generated from a
hand-written logistic risk formula. It is a software engineering demonstration,
NOT a clinically validated instrument, and must not be used to inform any real
medical decision.

Feature engineering is NOT defined in this file. It lives in
src/diabetes_screening_model.py and is shared with the training script so the
two cannot drift apart. See test_serving_parity.py.

Risk tiers and the decision threshold are read from the model package rather
than hardcoded, and are expressed as calibrated absolute risk relative to
population prevalence.

Model inputs (7): Age, Gender, BMI, Hypertension, Heart Disease, Diet,
Physical Activity.
"""

import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

# Shared training/serving feature pipeline - single source of truth.
from src.diabetes_screening_model import build_model_matrix, RAW_FEATURE_COLS

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

# Operating point chosen at training time (max specificity subject to a
# sensitivity floor). Never hardcode 0.5 here - prevalence is ~13.7%, so 0.5 is
# not a meaningful cut-point for this problem.
decision_threshold = model_package.get('decision_threshold', 0.5)

DISCLAIMER = model_package.get(
    'data_disclaimer',
    'SYNTHETIC DATA - NOT CLINICALLY VALIDATED. This tool must not be used to '
    'inform any real medical decision.'
)

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
print(f"   Decision threshold: {decision_threshold}")
print(f"   Calibrated: {model_package.get('calibrated', False)}")
print(f"   Model inputs: {RAW_FEATURE_COLS}")
print("\n" + "!" * 70)
print("  " + DISCLAIMER)
print("!" * 70 + "\n")


# =============================================================================
# FEATURE ENGINEERING
# -----------------------------------------------------------------------------
# There is deliberately NO feature engineering code in this file.
#
# This module previously kept its own copy of engineer_features(). It drifted
# from the training version (Age_Band -> Age_Group, Cardio_Risk_Score ->
# Cardiovascular_Risk), and because missing columns were silently zero-filled,
# every served prediction ran with Cardio_Risk_Score = 0 and all Age_Band
# one-hots = 0 - dropping the model's second-largest coefficient and shifting
# roughly 4% of users into the wrong risk tier.
#
# The serving path is now build_model_matrix() in src/diabetes_screening_model.py,
# the same function the training script uses. Do not reimplement it here.
# See test_serving_parity.py, which fails if the two paths ever diverge again.
# =============================================================================


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

        # 1. Build the raw input frame.
        #    Only RAW_FEATURE_COLS reach the model. The form still collects
        #    waist circumference, sedentary hours, sugary drinks, processed
        #    food, fruit/veg and family history, but THIS MODEL WAS NOT TRAINED
        #    ON THEM and they are echoed back as 'ignored_inputs' so the UI can
        #    be honest about it rather than silently discarding them.
        input_df = pd.DataFrame([{
            'gender': data.get('gender'),
            'age': float(data.get('age')),
            'hypertension': int(data.get('hypertension')),
            'heart_disease': int(data.get('heart_disease')),
            'bmi': float(data.get('bmi')),
            'Diet': data.get('diet'),
            'PhysicalActivity': data.get('physical_activity'),
        }])

        ignored_inputs = [
            key for key in (
                'waist_circumference', 'sedentary_hours', 'sugary_drinks',
                'processed_food', 'fruit_veg', 'family_history',
            )
            if data.get(key) not in (None, '')
        ]

        # 2. Build the model matrix using the SHARED training-time pipeline.
        #    This raises rather than zero-filling if the pipeline has drifted.
        input_encoded = build_model_matrix(
            input_df, feature_names, scaler, numerical_cols
        )

        # 3. Predict (probabilities are calibrated - see training notes)
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
            'prediction': int(probability >= decision_threshold),
            'probability': float(probability),
            'result': f"{risk_info['emoji']} {risk_info['tier']}",
            'tier': risk_info['tier'],
            'tier_code': risk_info['tier_code'],
            'tier_class': risk_info['tier_class'],
            'description': risk_info['description'],
            'advice': risk_info['recommendation'],
            'impact_factors': impact_factors,
            'ignored_inputs': ignored_inputs,
            'decision_threshold': float(decision_threshold),
            'calibrated': bool(model_package.get('calibrated', False)),
            'disclaimer': DISCLAIMER,
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
