
import pickle
import pandas as pd
import numpy as np
import os
import sys

# Load Model
MODEL_PATH = 'models/diabetes_screening_model.pkl'

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found!")
        sys.exit(1)
    
    with open(MODEL_PATH, 'rb') as f:
        pkg = pickle.load(f)
    print("✅ Model loaded successfully.")
    return pkg

def get_personas():
    """Define specific test cases to verify model logic."""
    return [
        {
            "name": "The Super Athlete (Low Risk)",
            "data": {
                "gender": "Male", "age": 25, "hypertension": 0, "heart_disease": 0, "bmi": 21.0,
                "Diet": "Healthy", "PhysicalActivity": "Active",
                "waist_circumference_cm": 75, "sedentary_hours_per_day": 3,
                "sugary_drink_frequency": 0, "processed_food_frequency": 0, "fruit_veg_frequency": 7,
                "family_history": 0
            }
        },
        {
            "name": "The Text-Book High Risk (High Risk)",
            "data": {
                "gender": "Male", "age": 60, "hypertension": 1, "heart_disease": 1, "bmi": 36.0,
                "Diet": "Unhealthy", "PhysicalActivity": "Sedentary",
                "waist_circumference_cm": 120, "sedentary_hours_per_day": 12,
                "sugary_drink_frequency": 7, "processed_food_frequency": 7, "fruit_veg_frequency": 0,
                "family_history": 1
            }
        },
        {
            "name": "The 'Skinny Fat' (Hidden Risk)",
            "description": "Normal BMI but High Waist & Poor Lifestyle",
            "data": {
                "gender": "Male", "age": 45, "hypertension": 0, "heart_disease": 0, "bmi": 24.0, 
                "Diet": "Unhealthy", "PhysicalActivity": "Sedentary",
                "waist_circumference_cm": 105, 
                "sedentary_hours_per_day": 10,
                "sugary_drink_frequency": 5, "processed_food_frequency": 5, "fruit_veg_frequency": 1,
                "family_history": 0
            }
        },
        {
            "name": "The Active Overweight (Mitigated Risk)",
            "description": "High BMI but Active & Healthy Diet",
            "data": {
                "gender": "Female", "age": 45, "hypertension": 0, "heart_disease": 0, "bmi": 31.0, 
                "Diet": "Healthy", "PhysicalActivity": "Active",
                "waist_circumference_cm": 95, 
                "sedentary_hours_per_day": 4, 
                "sugary_drink_frequency": 1, "processed_food_frequency": 1, "fruit_veg_frequency": 6,
                "family_history": 0
            }
        }
    ]

# Feature Engineering from App.py (Must match exactly)
def engineer_features(df_in):
    df_out = df_in.copy()
    
    diet_map = {'Unhealthy': 2, 'Mixed': 1, 'Healthy': 0}
    df_out['Diet_Score'] = df_out['Diet'].map(diet_map).fillna(1) 
    
    activity_map = {'Sedentary': 2, 'Moderately Active': 1, 'Active': 0}
    df_out['Activity_Score'] = df_out['PhysicalActivity'].map(activity_map).fillna(1)
    
    df_out['Lifestyle_Risk_Score'] = df_out['Diet_Score'] + df_out['Activity_Score']

    age_bins = [0, 30, 45, 60, 120]
    age_labels = ['young', 'middle', 'senior', 'elderly']
    df_out['Age_Group'] = pd.cut(df_out['age'], bins=age_bins, labels=age_labels, right=False)
    
    df_out['Age_Risk_Flag'] = (df_out['age'] >= 45).astype(int)
    
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['underweight', 'normal', 'overweight', 'obese']
    df_out['BMI_Category'] = pd.cut(df_out['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)
    
    df_out['Obesity_Flag'] = (df_out['bmi'] >= 30).astype(int)
    
    df_out['Cardiovascular_Risk'] = df_out['hypertension'] + df_out['heart_disease']
    
    if 'family_history' in df_out.columns:
         df_out['FamilyHistory'] = df_out['family_history']
            
    df_out['Genetic_Risk'] = df_out['FamilyHistory']
    
    df_out['Age_BMI_Interaction'] = df_out['age'] * df_out['bmi']
    df_out['Lifestyle_BMI_Interaction'] = df_out['Lifestyle_Risk_Score'] * df_out['bmi']
    df_out['Genetic_Age_Interaction'] = df_out['Genetic_Risk'] * df_out['age']
    
    df_out['Metabolic_Risk_Score'] = (
        df_out['Age_Risk_Flag'] + 
        df_out['Obesity_Flag'] + 
        df_out['Cardiovascular_Risk'] +
        df_out['Lifestyle_Risk_Score'] +
        df_out['Genetic_Risk']
    )
    
    df_out['Metabolic_Strain'] = np.log1p(df_out['age']) * df_out['bmi']
    
    return df_out

def main():
    pkg = load_model()
    model = pkg['model']
    scaler = pkg['scaler']
    feature_names = pkg['feature_names']
    numerical_cols = pkg['numerical_cols']
    
    # Categorical features expected by get_dummies
    # We must match what was used in training
    CATEGORICAL_FEATURES = ['gender', 'Age_Group', 'BMI_Category', 'Diet', 'PhysicalActivity']

    print("\n🔬 RUNNING PERSONA TESTS...\n")
    
    personas = get_personas()
    
    for p in personas:
        print(f"--- Testing: {p['name']} ---")
        if 'description' in p:
            print(f"Context: {p['description']}")
            
        # Create DF
        input_df = pd.DataFrame([p['data']])
        
        # Engineer
        input_df = engineer_features(input_df)
        
        # Encode (This is tricky without the full training set structure, 
        # but we can try to align columns manually if we know the schema)
        # Better: Use pd.get_dummies and then reindex
        input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int)
        
        # Align
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]
        
        # Scale
        if numerical_cols:
             input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])
        
        # Predict
        prob = model.predict_proba(input_encoded)[0][1]
        
        print(f"🔮 Predicted Risk: {prob:.2%}")
        
        # Basic Validation Logic
        if "Low Risk" in p['name'] and prob > 0.3:
            print("⚠️ WARNING: Risk suspiciously high for healthy persona!")
        elif "High Risk" in p['name'] and prob < 0.7:
             print("⚠️ WARNING: Risk suspiciously low for high-risk persona!")
        
        print("")

if __name__ == "__main__":
    main()
