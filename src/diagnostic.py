
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

try:
    print("Loading data...")
    df = pd.read_csv('data/raw/diabetes_prediction_dataset_final.csv')
    print(f"Loaded {len(df)} rows.")
    
    # Feature Engineering (copied simplified version)
    df['Diet_Score'] = df['Diet'].map({'Unhealthy': 2, 'Mixed': 1, 'Healthy': 0}).fillna(1)
    df['Activity_Score'] = df['PhysicalActivity'].map({'Sedentary': 2, 'Moderately Active': 1, 'Active': 0}).fillna(1)
    df['Lifestyle_Risk_Score'] = df['Diet_Score'] + df['Activity_Score']
    
    age_bins = [0, 30, 45, 60, 120]
    age_labels = ['young', 'middle', 'senior', 'elderly']
    df['Age_Group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    
    df['Age_Risk_Flag'] = (df['age'] >= 45).astype(int)
    
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['underweight', 'normal', 'overweight', 'obese']
    df['BMI_Category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)
    
    df['Obesity_Flag'] = (df['bmi'] >= 30).astype(int)
    df['Cardiovascular_Risk'] = df['hypertension'] + df['heart_disease']
    # df['FamilyHistory'] might not exist? Check file columns.
    if 'FamilyHistory' not in df.columns:
        print("Creating dummy FamilyHistory...")
        df['FamilyHistory'] = 0 # Or derived if needed, but lets check if it exists
        
    df['Genetic_Risk'] = df['FamilyHistory']
    df['Age_BMI_Interaction'] = df['age'] * df['bmi']
    df['Lifestyle_BMI_Interaction'] = df['Lifestyle_Risk_Score'] * df['bmi']
    df['Genetic_Age_Interaction'] = df['Genetic_Risk'] * df['age']
    
    df['Metabolic_Risk_Score'] = (
        df['Age_Risk_Flag'] + 
        df['Obesity_Flag'] + 
        df['Cardiovascular_Risk'] +
        df['Lifestyle_Risk_Score'] +
        df['Genetic_Risk']
    )
    
    df['Metabolic_Strain'] = np.log1p(df['age']) * df['bmi']
    
    print("Feature Engineering Done.")
    
    df_clean = df.drop_duplicates(keep='first')
    
    X = df_clean.drop(['diabetes', 'diabetes_updated'], axis=1)
    y = df_clean['diabetes_updated']
    
    print(f"Features: {X.columns}")
    
    categorical_features = ['gender', 'Diet', 'PhysicalActivity']
    new_categorical = ['Age_Group', 'BMI_Category']
    
    print("Encoding...")
    X_encoded = pd.get_dummies(X, columns=categorical_features + new_categorical, drop_first=True, dtype=int)
    print("Encoded shape:", X_encoded.shape)
    
    print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("SMOTE...")
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("SMOTE Done.")
    
except Exception as e:
    print("\n❌ ERROR:")
    print(e)
    import traceback
    traceback.print_exc()
