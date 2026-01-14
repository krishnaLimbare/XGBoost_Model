
import pandas as pd
import numpy as np
import os
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
INPUT_FILE = os.path.join(DATA_DIR, 'diabetes_screening_dataset.csv')
BACKUP_FILE = os.path.join(DATA_DIR, 'dataset_original_backup.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'diabetes_prediction_dataset_research_augmented.csv')

def load_and_backup_data():
    """Load original data and create a safe backup."""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    
    # create backup
    print(f"Creating backup at {BACKUP_FILE}...")
    shutil.copy2(INPUT_FILE, BACKUP_FILE)
    
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")
    return df

def generate_waist_circumference(row):
    """
    Generate waist circumference (cm) based on BMI, Gender, and Age.
    Males: ~70-125 cm
    Females: ~65-115 cm
    Positively correlated with BMI and Age.
    """
    # Baseline ratio based on clinical approximations
    # Men: Waist ~ BMI * 3.3 + Age_factor
    # Women: Waist ~ BMI * 3.1 + Age_factor
    
    bmi = row['bmi']
    age = row['age']
    gender = row['gender']
    
    # Base calculation
    if gender == 'Male':
        base_waist = bmi * 3.4
        gender_offset = 5.0
    else:  # Female or Other
        base_waist = bmi * 3.15
        gender_offset = 0.0
        
    # Age factor: older people tend to have larger waist for same BMI
    age_effect = (age / 10.0) * 1.5
    
    # Random noise (Soft noise)
    noise = np.random.normal(0, 3.0) 
    
    waist = base_waist + gender_offset + age_effect + noise
    
    # Clip to realistic ranges defined in requirement
    if gender == 'Male':
        waist = np.clip(waist, 70, 125)
    else:
        waist = np.clip(waist, 65, 115)
        
    return round(waist, 1)

def generate_sedentary_hours(row):
    """
    Generate sedentary hours (2-14 hr/day).
    Inversely related to PhysicalActivity.
    Higher for Higher BMI, Older Age.
    """
    activity = row['PhysicalActivity'] # Sedentary, Moderately Active, Active
    bmi = row['bmi']
    age = row['age']
    
    # Base hours by activity level
    if activity == 'Sedentary':
        base_hours = 10.0
    elif activity == 'Moderately Active':
        base_hours = 6.0
    else: # Active
        base_hours = 3.5
        
    # BMI Effect: +0.1 hour per BMI point above 25
    bmi_effect = max(0, (bmi - 25) * 0.1)
    
    # Age Effect: +0.05 hour per year above 30
    age_effect = max(0, (age - 30) * 0.05)
    
    # Random variance (High active person can still sit a lot at work)
    noise = np.random.normal(0, 1.5)
    
    hours = base_hours + bmi_effect + age_effect + noise
    
    # Clip 2-14
    hours = np.clip(hours, 2.0, 14.0)
    return round(hours, 1)

def generate_diet_features(row):
    """
    Generate refined diet indicators based on 'Diet' column.
    1. sugary_drink_frequency (0-7 per week)
    2. processed_food_frequency (0-7 days per week)
    3. fruit_veg_frequency (0-7 days per week)
    """
    diet_status = row['Diet'] # Unhealthy, Mixed, Healthy
    
    # Probability distributions (High/Med/Low)
    
    if diet_status == 'Unhealthy':
        # High bad stuff, low good stuff
        sugary = np.random.normal(5.5, 1.5)
        processed = np.random.normal(5.0, 1.5)
        fruit = np.random.normal(1.5, 1.5)
    elif diet_status == 'Mixed':
        sugary = np.random.normal(3.0, 1.5)
        processed = np.random.normal(3.0, 1.5)
        fruit = np.random.normal(3.5, 1.5)
    else: # Healthy
        sugary = np.random.normal(0.5, 1.0)
        processed = np.random.normal(1.0, 1.0)
        fruit = np.random.normal(6.0, 1.0)
        
    # Correlate sugary/processed with BMI slightly (secondary effect)
    if row['bmi'] > 30:
        sugary += 0.5
        processed += 0.5
        
    return (
        int(np.clip(round(sugary), 0, 7)),
        int(np.clip(round(processed), 0, 7)),
        int(np.clip(round(fruit), 0, 7))
    )

def main():
    np.random.seed(42) # Reproducibility
    
    df = load_and_backup_data()
    
    print("Generating new features...")
    
    # Step 2: Waist Circumference
    df['waist_circumference_cm'] = df.apply(generate_waist_circumference, axis=1)
    
    # Step 3: Sedentary Behavior
    df['sedentary_hours_per_day'] = df.apply(generate_sedentary_hours, axis=1)
    
    # Step 4: Diet Indicators
    diet_features = df.apply(generate_diet_features, axis=1, result_type='expand')
    df['sugary_drink_frequency'] = diet_features[0]
    df['processed_food_frequency'] = diet_features[1]
    df['fruit_veg_frequency'] = diet_features[2]
    
    # Step 4.5: Family History (Missing in original screening dataset but required by model)
    # Generate based on some probability, slightly higher if original diabetes was 1 (though we treat orig as ground truthish)
    # Prevalence ~15-20%
    family_prob = 0.15
    df['FamilyHistory'] = np.random.choice([0, 1], size=len(df), p=[1-family_prob, family_prob])

    # =========================================================================
    # STEP 6 & 7: UPDATE DIABETES LABEL (controlled)
    # =========================================================================
    print("Adjusting target labels based on new risk factors...")
    
    # Calculate Risk Probabilities
    # We want a small additive effect.
    # Scores are roughly normalized to 0-1 range for weight calculation
    
    # 1. Waist Risk: > 100cm (M) or > 90cm (F) is high risk. 
    # normalize: (waist - 70) / 50 -> approx 0-1
    waist_norm = (df['waist_circumference_cm'] - 80) / 40.0 
    
    # 2. Sedentary Risk: > 8 hours is risky
    sedentary_norm = (df['sedentary_hours_per_day'] - 2) / 12.0
    
    # 3. Diet Risk: Sum of bad / Max possible (14) - Good / Max (7)
    diet_risk_norm = (df['sugary_drink_frequency'] + df['processed_food_frequency']) / 14.0
    protection_norm = df['fruit_veg_frequency'] / 7.0
    
    # 4. Family History
    genetic_norm = df['FamilyHistory'] * 0.5
    
    # Composite Risk Additive Factor (approx -0.1 to +0.4)
    # Weights: Waist (strong), Sedentary (med), Diet (med), Genetic (strong)
    risk_score = (
        0.4 * waist_norm + 
        0.2 * sedentary_norm + 
        0.2 * diet_risk_norm - 
        0.2 * protection_norm +
        0.3 * genetic_norm
    )
    
    # Apply to modify existing diabetes status
    # We only flip 0 -> 1 if Risk is VERY High (Aligns with Step 7: "multiple risk factors align")
    # We rely on the existing label but add a stochastic element for "borderline" cases
    
    # Threshold for flipping 0 -> 1
    # Check distribution of risk_score
    # top 10-15% of risk scores might be candidates
    risk_threshold = np.percentile(risk_score, 85) # Top 15% risk
    
    base_labels = df['diabetes'].copy()
    new_labels = base_labels.copy()
    
    # Condition: Original is 0 AND Risk Score is high AND Random Chance
    # We want to change at most ~10-12% of total rows, but specifically focused on high risk
    
    candidates = (base_labels == 0) & (risk_score > risk_threshold)
    
    # Stochastic conversion for candidates (e.g. 50% chance if candidate)
    # This ensures we don't just mechanically flip all high-waist people
    rng = np.random.default_rng(42)
    flip_mask = candidates & (rng.random(len(df)) < 0.6) # 60% of candidates
    
    new_labels[flip_mask] = 1
    
    df['diabetes_updated'] = new_labels
    
    # Check stats
    changes = sum(base_labels != new_labels)
    pct_changed = (changes / len(df)) * 100
    print(f"Label Updates: {changes} rows flipped ({pct_changed:.2f}%)")
    
    if pct_changed > 15.0:
        print(f"⚠️ SAFETY BREAK: Too many label changes ({pct_changed:.2f}%). Rolling back.")
        return # Exit without saving

    # =========================================================================
    # STEP 8: SAFETY CHECK (Baseline Model)
    # =========================================================================
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    
    print("Running safety checks (Baseline Model comparison)...")
    
    # Prepare data
    features_original = ['age', 'bmi', 'hypertension', 'heart_disease'] # subset of original reliable feats
    features_new = features_original + ['waist_circumference_cm', 'sedentary_hours_per_day', 
                                      'sugary_drink_frequency']
    
    # Model 1: Original Target using Original Features (The "Truth" we had)
    # This isn't quite right comparison. We want to see if New Target is "learnable" and consistent.
    # Comparison 1: Predict Old Target with Old Features
    # Comparison 2: Predict New Target with New Features
    # If AUC drops significantly, we added noise. If AUC goes to 1.0, we leaked.
    
    # Old Dataset proxy
    X_old = pd.get_dummies(df[features_original + ['gender']])
    y_old = df['diabetes']
    
    # New Dataset proxy
    X_new = pd.get_dummies(df[features_new + ['gender']])
    y_new = df['diabetes_updated']
    
    # Train/Test 1
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_old, y_old, test_size=0.2, random_state=42)
    model1 = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model1.fit(X_train1, y_train1)
    acc1 = model1.score(X_test1, y_test1)
    auc1 = roc_auc_score(y_test1, model1.predict_proba(X_test1)[:,1])
    
    # Train/Test 2
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
    model2 = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model2.fit(X_train2, y_train2)
    acc2 = model2.score(X_test2, y_test2)
    auc2 = roc_auc_score(y_test2, model2.predict_proba(X_test2)[:,1])
    
    print(f"Original Model (Target=Old): AUC={auc1:.4f}")
    print(f"Enhanced Model (Target=New): AUC={auc2:.4f}")
    
    if auc2 > 0.99:
        print("⚠️ SAFETY ERROR: Potential Data Leakage (AUC ~ 1.0). Rolling back.")
        return
        
    if auc2 < (auc1 - 0.1): # 10% drop
        print("⚠️ SAFETY ERROR: Significant performance degradation. Rolling back.")
        return

    # =========================================================================
    # STEP 9: FINAL SAVE
    # =========================================================================
    
    FINAL_FILE = os.path.join(DATA_DIR, 'diabetes_prediction_dataset_final.csv')
    print(f"Safety checks passed. Saving final dataset to {FINAL_FILE}...")
    
    # Move diabetes_updated to 'diabetes' for final file? OR keep both?
    # Requirement: "Store updated target as: diabetes_updated. Do NOT overwrite the original diabetes column."
    # The final dataset should probably have 'diabetes_updated' as the column to use, but let's keep 'diabetes' too.
    
    df.to_csv(FINAL_FILE, index=False)
    print("✅ Augmentation and Refinement Complete.")

if __name__ == "__main__":
    main()
