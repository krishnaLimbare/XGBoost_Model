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

# Define Categorical Features
CATEGORICAL_FEATURES = ['gender', 'smoking_history', 'occupation', 'drinking', 'altitude']

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
    
    df_out = df_out.drop(columns=['smoking_score', 'drinking_score', 'sedentary_score'], errors='ignore')
    
    return df_out

def predict_case(case_name, data_dict):
    input_df = pd.DataFrame([data_dict])
    
    # Apply Feature Engineering
    input_df = engineer_features(input_df)

    # Encode
    input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int)
    
    # Align columns
    # Get expected features from model booster to be safe
    try:
        expected_cols = model.get_booster().feature_names
        # Fallback if empty (some versions)
        if not expected_cols: expected_cols = model_columns
    except:
        expected_cols = model_columns
    
    for col in expected_cols:
         if col not in input_encoded.columns:
             input_encoded[col] = 0
             
    # Filter and reorder
    input_encoded = input_encoded[[c for c in expected_cols if c in input_encoded.columns]] # Filter valid
    input_encoded = input_encoded[expected_cols] # Enforce order
    
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]
    
    status = "DIABETES DETECTED" if prediction == 1 else "NO DIABETES"

    with open('repro_results_final.txt', 'a') as f:
        f.write(f"{case_name:<50} | Prob: {probability:.4f} | {status}\n")
    print(f"{case_name:<50} | Prob: {probability:.4f} | {status}")

print("\n" + "="*80)
if os.path.exists('repro_results_final.txt'):
    os.remove('repro_results_final.txt')
print("SENSITIVITY ANALYSIS")
print("="*80)

# Base dictionary (Healthy-ish)
base_case = {
    'gender': 'Male', 'age': 30, 'bmi': 22, 'hypertension': 0, 'heart_disease': 0,
    'smoking_history': 'never', 'drinking': 'non_drinker', 'occupation': 'office_worker',
    'altitude': 'low_0-500m', 'family_history': 0
}

predict_case("Base Case (Healthy, Young)", base_case)

# Varied Age
predict_case("Age 60 (Only Age Changed)", {**base_case, 'age': 60})
predict_case("Age 80 (Only Age Changed)", {**base_case, 'age': 80})

# Varied BMI
predict_case("BMI 30 (Overweight)", {**base_case, 'bmi': 30})
predict_case("BMI 35 (Obese)", {**base_case, 'bmi': 35})
predict_case("BMI 45 (Morbidly Obese)", {**base_case, 'bmi': 45})

# Hypertension & Heart Disease
predict_case("Hypertension Only", {**base_case, 'hypertension': 1})
predict_case("Heart Disease Only", {**base_case, 'heart_disease': 1})
predict_case("Hypertension + Heart Disease", {**base_case, 'hypertension': 1, 'heart_disease': 1})

# Combined Risk
predict_case("Age 60 + BMI 35", {**base_case, 'age': 60, 'bmi': 35})
predict_case("Age 80 + BMI 45 (No History)", {**base_case, 'age': 80, 'bmi': 45})
predict_case("Age 80 + BMI 45 + History (Extreme)", {**base_case, 'age': 80, 'bmi': 45, 'hypertension': 1, 'heart_disease': 1, 'family_history': 1})

# Lifestyle
predict_case("Smoker + Drinker (Young)", {**base_case, 'smoking_history': 'current', 'drinking': 'heavy'})
predict_case("Smoker + Drinker (Old)", {**base_case, 'age': 60, 'smoking_history': 'current', 'drinking': 'heavy'})

print("\nDone.")
