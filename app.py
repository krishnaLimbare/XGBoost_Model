import os
import pickle
import pandas as pd
import numpy as np
import yaml
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# =============================================================================
# CONFIGURATION & LOADING
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'diabetes_model.pkl')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
X_TRAIN_PATH = os.path.join(DATA_PROCESSED_DIR, 'X_train.csv')

# Load Model
print(f"Loading model from: {MODEL_PATH}")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load Training Columns to ensure correct One-Hot Encoding
model_columns = []

# Try to get features from model first (Most reliable)
try:
    if hasattr(model, 'feature_names_in_'):
        print("Loading feature names directly from model...")
        model_columns = model.feature_names_in_.tolist()
    elif hasattr(model, 'get_booster'):
        print("Loading feature names from XGBoost booster...")
        model_columns = model.get_booster().feature_names
except Exception as e:
    print(f"Could not extract feature names from model: {e}")

# Fallback to X_train.csv if needed
if not model_columns:
    print(f"Loading training data features from: {X_TRAIN_PATH}")
    try:
        X_train_df = pd.read_csv(X_TRAIN_PATH)
        model_columns = X_train_df.columns.tolist()
    except Exception as e:
        print(f"Error loading X_train.csv: {e}")
        print("WARNING: Feature alignment might fail if X_train.csv is missing or stale.")

print(f"Model expects these {len(model_columns)} columns.")

# Define Categorical Features (Must match training)
CATEGORICAL_FEATURES = ['gender', 'smoking_history', 'occupation', 'drinking', 'altitude']

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

# =============================================================================
# FEATURE ENGINEERING FUNCTION (SHARED - MUST MATCH TRAIN_MODEL.PY)
# =============================================================================
def engineer_features(df_in):
    """
    Creates composite features and proxy biomarkers.
    Must be identical in train_model.py and app.py
    """
    df_out = df_in.copy()
    
    # 1. Comorbidity Score
    df_out['Comorbidity_Score'] = df_out['hypertension'] + df_out['heart_disease'] + df_out['family_history']
    
    # 2. Age-BMI Interaction
    df_out['Age_BMI_Interaction'] = df_out['age'] * df_out['bmi']
    
    # 3. Lifestyle Risk Score
    smoking_map = {
        'never': 0, 'No Info': 0.5, 'current': 2, 'former': 1, 
        'ever': 1, 'not current': 0.5
    }
    df_out['smoking_score'] = df_out['smoking_history'].map(smoking_map).fillna(0)
    
    drinking_map = {
        'non_drinker': 0, 'light': 0.5, 'moderate': 1, 'heavy': 2
    }
    df_out['drinking_score'] = df_out['drinking'].map(drinking_map).fillna(0)
    
    # Occupation logic handled via input string, assuming raw input
    occ_map = {
        'office_worker': 2, 'student': 2, 'retired': 1, 'unemployed': 1,
        'healthcare': 1, 'professional': 2, 'service_industry': 0, 
        'manual_labor': 0, 'self_employed': 1
    }
    if 'occupation' in df_out.columns:
         df_out['sedentary_score'] = df_out['occupation'].map(occ_map).fillna(1)
    else:
         df_out['sedentary_score'] = 1 
         
    df_out['Lifestyle_Risk'] = df_out['smoking_score'] + df_out['drinking_score'] + df_out['sedentary_score']
    
    # 4. Metabolic Strain Index
    df_out['Metabolic_Strain'] = np.log1p(df_out['age']) * df_out['bmi']

    # 5. Age-Family Interaction (Genetic risk often manifests later)
    df_out['Age_Family_Interaction'] = df_out['age'] * df_out['family_history']

    # 6. Cumulative Smoking Risk (Age * Smoking Score)
    df_out['Cumulative_Smoking_Risk'] = df_out['age'] * df_out['smoking_score']

    # 7. Hyper-Comorbidity Interaction
    df_out['Hyper_Comorbidity'] = (df_out['hypertension'] + df_out['heart_disease']) * df_out['family_history']
    
    df_out = df_out.drop(columns=['smoking_score', 'drinking_score', 'sedentary_score'], errors='ignore')
    
    return df_out

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Received prediction request: {data}")

        # 1. Create DataFrame from input
        input_df = pd.DataFrame([data])

        # 1b. Feature Engineering (Apply BEFORE encoding)
        input_df = engineer_features(input_df)

        # 2. One-Hot Encode (get_dummies)
        # We process categorical columns same as training
        # Use drop_first=True to match training behavior more closely, but we handle alignment manually regardless.
        # Actually in app.py it was using drop_first=False. 
        # But training uses True. We should use True here too ideally to generate same features.
        # But previously it aligned columns anyway.
        # Let's use drop_first=True to be fully consistent with training code in train_model.py
        input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int)

        # 3. Align Columns with Model
        # Add missing columns (init with 0)
        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Remove extra columns (if any)
        input_encoded = input_encoded[model_columns]
        
        # Ensure order matches
        input_encoded = input_encoded[model_columns]

        # 4. Predict
        # prediction = model.predict(input_encoded)[0] # Threshold dependent
        probability = model.predict_proba(input_encoded)[0][1]

        # 4b. Heuristic Adjustment for High-Genetic-Risk Lean Cases
        # Issue: Model underestimates lean diabetics with strong genetic/lifestyle risk
        # Apply minimum probability floor for specific high-risk profiles
        original_probability = probability
        
        # Check if patient matches high-genetic-risk profile
        is_lean = data['bmi'] < 25
        has_family_history = data['family_history'] == 1
        is_older = data['age'] >= 45
        is_current_smoker = data['smoking_history'] == 'current'
        has_comorbidity = data['hypertension'] == 1 or data['heart_disease'] == 1
        
        # Apply floor for lean individuals with genetic risk
        if is_lean and has_family_history and is_older:
            # Base adjustment
            min_prob = 0.35
            
            # Additional adjustments
            if is_current_smoker:
                min_prob = 0.45  # Current smoker + family history = at least Moderate
            if has_comorbidity:
                min_prob = max(min_prob, 0.50)  # Comorbidity boosts to High tier
                
            probability = max(probability, min_prob)
            
            if probability > original_probability:
                print(f"Applied genetic risk adjustment: {original_probability:.4f} -> {probability:.4f}")

        # 5. Determine Tier
        if probability < 0.20:
            tier = "Very Low Risk 🌱"
            tier_class = "risk-very-low"
            advice = "Great job! Maintain your healthy lifestyle."
        elif probability < 0.45:
            tier = "Low Risk 🛡️"
            tier_class = "risk-low"
            advice = "Good health status. Keep monitoring your vitals."
        elif probability < 0.65:
            tier = "Moderate Risk ⚠️"
            tier_class = "risk-moderate"
            advice = "Caution: Some risk factors present. Consult a doctor for preventative checks."
        elif probability < 0.85:
            tier = "High Risk 🧡"
            tier_class = "risk-high"
            advice = "Warning: High probability of diabetes. Medical attention recommended."
        else:
            tier = "Critical Risk 🚨"
            tier_class = "risk-critical"
            advice = "URGENT: Very high risk detected. Please see a specialist immediately."

        result_text = f"{tier}"
        print(f"Prediction: {result_text} ({probability:.4f})")

        return jsonify({
            'status': 'success',
            'prediction': int(probability > 0.5),  # 0 or 1 based on std threshold
            'result': result_text,
            'tier': tier,
            'tier_class': tier_class,
            'advice': advice,
            'probability': float(probability)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
