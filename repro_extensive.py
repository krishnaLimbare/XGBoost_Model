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

def predict_case(case_name, data_dict):
    input_df = pd.DataFrame([data_dict])
    input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_FEATURES, drop_first=False)
    
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    input_encoded = input_encoded[model_columns]
    
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
