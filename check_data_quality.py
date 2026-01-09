import pandas as pd
import numpy as np
import os

BASE_DIR = r"c:\Users\Asus\Downloads\XGBoost_Model"
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'diabetes_prediction_dataset_augmented.csv')

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)



# Create BMI bins
df['bmi_bin'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 40, 100], 
                       labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III'])

# Calculate diabetes rate per bin
bmi_stats = df.groupby('bmi_bin')['diabetes'].agg(['count', 'mean']).rename(columns={'mean': 'diabetes_rate'})

# Extreme BMI Check
high_bmi = df[df['bmi'] >= 40]

with open('data_quality_report.txt', 'w') as f:
    f.write("--- BMI Analysis ---\n")
    f.write(bmi_stats.to_string())
    f.write(f"\n\nCorrelation: {df['bmi'].corr(df['diabetes']):.4f}\n")
    f.write("\nHigh BMI (>= 40) stats:\n")
    f.write(high_bmi['diabetes'].value_counts(normalize=True).to_string())

print("Report saved to data_quality_report.txt")
