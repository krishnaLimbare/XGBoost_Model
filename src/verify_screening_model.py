"""
Test script for diabetes screening model predictions
"""
import pickle
import pandas as pd
import numpy as np

# Load model
with open('models/diabetes_screening_model.pkl', 'rb') as f:
    pkg = pickle.load(f)

model = pkg['model']
scaler = pkg['scaler']
feature_names = pkg['feature_names']
numerical_cols = pkg['numerical_cols']

def engineer_features(df_in):
    df = df_in.copy()
    age_bins = [0, 30, 45, 60, 120]
    age_labels = ['young', 'middle', 'senior', 'elderly']
    df['Age_Band'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    df['Age_Risk_Flag'] = (df['age'] >= 45).astype(int)
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['underweight', 'normal', 'overweight', 'obese']
    df['BMI_Category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)
    df['Obesity_Flag'] = (df['bmi'] >= 30).astype(int)
    df['Age_BMI_Interaction'] = df['age'] * df['bmi']
    df['Cardio_Risk_Score'] = df['hypertension'] + df['heart_disease']
    diet_map = {'Healthy': 0, 'Mixed': 1, 'Unhealthy': 2}
    df['Diet_Score'] = df['Diet'].map(diet_map).fillna(1)
    activity_map = {'Active': 0, 'Moderately Active': 1, 'Sedentary': 2}
    df['Activity_Score'] = df['PhysicalActivity'].map(activity_map).fillna(1)
    df['Lifestyle_Risk_Score'] = df['Diet_Score'] + df['Activity_Score']
    return df

def predict(data):
    input_df = pd.DataFrame([data])
    input_df = engineer_features(input_df)
    cat_cols = ['gender', 'Age_Band', 'BMI_Category', 'Diet', 'PhysicalActivity']
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True, dtype=int)
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_names]
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])
    prob = model.predict_proba(input_encoded)[0][1]
    tier = 'Low' if prob < 0.30 else ('Medium' if prob < 0.60 else 'High')
    return prob, tier

print("=" * 60)
print("DIABETES SCREENING MODEL - PREDICTION TESTS")
print("=" * 60)

# Test Case 1: Low Risk (Young, healthy)
case1 = {'gender': 'Female', 'age': 25, 'hypertension': 0, 'heart_disease': 0, 
         'bmi': 22, 'Diet': 'Healthy', 'PhysicalActivity': 'Active'}
prob1, tier1 = predict(case1)
print(f"\nTest 1: Young, healthy female (25yo, BMI 22, no conditions, healthy lifestyle)")
print(f"  Expected: Low Risk")
print(f"  Result:   Probability={prob1:.3f}, Tier={tier1}")

# Test Case 2: Medium Risk (Middle age, some risk factors)
case2 = {'gender': 'Male', 'age': 50, 'hypertension': 1, 'heart_disease': 0, 
         'bmi': 28, 'Diet': 'Mixed', 'PhysicalActivity': 'Moderately Active'}
prob2, tier2 = predict(case2)
print(f"\nTest 2: Middle-aged male (50yo, BMI 28, hypertension, mixed lifestyle)")
print(f"  Expected: Medium Risk")
print(f"  Result:   Probability={prob2:.3f}, Tier={tier2}")

# Test Case 3: High Risk (Elderly, multiple risk factors)
case3 = {'gender': 'Male', 'age': 65, 'hypertension': 1, 'heart_disease': 1, 
         'bmi': 35, 'Diet': 'Unhealthy', 'PhysicalActivity': 'Sedentary'}
prob3, tier3 = predict(case3)
print(f"\nTest 3: Elderly male (65yo, BMI 35, both conditions, poor lifestyle)")
print(f"  Expected: High Risk")
print(f"  Result:   Probability={prob3:.3f}, Tier={tier3}")

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
all_pass = tier1 == 'Low' and tier2 == 'Medium' and tier3 == 'High'
print(f"Test 1 (Low Risk):    {'PASS' if tier1 == 'Low' else 'FAIL'}")
print(f"Test 2 (Medium Risk): {'PASS' if tier2 == 'Medium' else 'FAIL'}")  
print(f"Test 3 (High Risk):   {'PASS' if tier3 == 'High' else 'FAIL'}")
print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
