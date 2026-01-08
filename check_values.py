import pandas as pd
import os

try:
    df = pd.read_csv('data/raw/diabetes_prediction_dataset_augmented.csv')
    with open('values_log.txt', 'w') as f:
        f.write("Unique Values:\n")
        for col in ['gender', 'smoking_history', 'drinking', 'occupation', 'altitude', 'family_history', 'hypertension', 'heart_disease']:
            f.write(f"\n--- {col} ---\n")
            f.write(str(df[col].unique()) + "\n")
except Exception as e:
    with open('values_log.txt', 'w') as f:
        f.write(str(e))
