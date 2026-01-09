import pandas as pd
import os

BASE_DIR = r"c:\Users\Asus\Downloads\XGBoost_Model"
RAW_DATA = os.path.join(BASE_DIR, 'data', 'raw', 'diabetes_prediction_dataset_augmented.csv')

df = pd.read_csv(RAW_DATA)

cols_to_check = ['gender', 'smoking_history', 'occupation', 'drinking', 'altitude']

print("--- UNIQUE VALUES IN TRAINING DATA ---")
for col in cols_to_check:
    if col in df.columns:
        print(f"\n[{col}]")
        print(df[col].unique())
    else:
        print(f"\n[{col}] NOT FOUND")
