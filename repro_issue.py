import pandas as pd
import pickle
import os
import numpy as np

# Load model and feature info
BASE_DIR = r"c:\Users\Asus\Downloads\XGBoost_Model"
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'diabetes_model.pkl')
X_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'X_train.csv')

print(f"Loading model from: {MODEL_PATH}")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print(f"Loading feature columns from: {X_TRAIN_PATH}")
X_train_df = pd.read_csv(X_TRAIN_PATH)
model_columns = X_train_df.columns.tolist()
print(f"Model expects {len(model_columns)} columns: {model_columns}")

# Define Categorical Features (Must match app.py)
CATEGORICAL_FEATURES = ['gender', 'smoking_history', 'occupation', 'drinking', 'altitude']

def predict_case(case_name, data_dict):
    print(f"\n--- Testing Case: {case_name} ---")
    print(f"Input: {data_dict}")
    
    input_df = pd.DataFrame([data_dict])
    
    # One-Hot Encode
    input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_FEATURES, drop_first=False)
    
    # Align Columns
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    # Remove extra and reorder
    input_encoded = input_encoded[model_columns]
    
    # Predict
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]
    
    print(f"Prediction: {prediction} ('DIABETES DETECTED' if 1 else 'NO DIABETES')")
    print(f"Probability: {probability:.4f}")

# Extreme Case 1: Very high risk
extreme_high_risk = {
    'gender': 'Male',
    'age': 80,
    'hypertension': 1,
    'heart_disease': 1,
    'smoking_history': 'current',
    'bmi': 45.0,
    'HbA1c_level': 9.0, # High
    'blood_glucose_level': 300, # High
    'occupation': 'Office job', # Sedentary?
    'drinking': 'heavy',
    'altitude': 'low',
    'family_history': 1,
    'physical_activity_level': 1, # Low
    'sleep_hours': 4,
    'stress_level': 10
}

# Extreme Case 2: Very high risk but maybe formatted differently (strings vs ints?)
extreme_high_risk_2 = {
    'gender': 'Female',
    'age': 75,
    'hypertension': 1,
    'heart_disease': 1,
    'smoking_history': 'current',
    'bmi': 40.0,
    # Intentionally missing HtA1c/Glucose if they were removed from model, check model cols
    'occupation': 'Unemployed',
    'drinking': 'heavy',
    'altitude': 'low',
    'family_history': 1,
    'physical_activity_level': 1, 
    'sleep_hours': 5,
    'stress_level': 9
}

predict_case("Extreme High Risk (With Medical Values)", extreme_high_risk)
predict_case("Extreme High Risk 2 (Potentially missing vals)", extreme_high_risk_2)
