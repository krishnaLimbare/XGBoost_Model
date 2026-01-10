"""
Diabetes Screening Dataset Generator
=====================================
Generates a realistic synthetic dataset for diabetes screening with:
- Approximately 10-15% diabetes prevalence (realistic for screening)
- Features: Age, Gender, BMI, Hypertension, Heart Disease, Diet, Physical Activity
- Probabilistic diabetes outcome based on known risk factors

Author: ML Pipeline
Date: 2026-01-10
"""

import pandas as pd
import numpy as np
import os

def generate_screening_dataset(n_samples=100000, seed=42):
    """
    Generate a realistic diabetes screening dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    np.random.seed(seed)
    
    print(f"🚀 Generating Diabetes Screening Dataset ({n_samples:,} samples)...")
    
    # =========================================================================
    # STEP 1: Generate Base Demographics
    # =========================================================================
    
    # Age: Normal distribution centered around 45, range 18-80
    age = np.clip(np.random.normal(45, 15, n_samples), 18, 80).astype(int)
    
    # Gender: ~50/50 split with small variation
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.48, 0.52])
    
    # =========================================================================
    # STEP 2: Generate Health Metrics (Correlated with Age)
    # =========================================================================
    
    # BMI: Slightly correlated with age (older people tend to have higher BMI)
    # Base BMI from normal distribution, slightly increases with age
    bmi_base = np.random.normal(26, 5, n_samples)  # Mean 26, slightly overweight population
    bmi_age_effect = (age - 40) * 0.05  # Slight increase with age
    bmi = np.clip(bmi_base + bmi_age_effect, 15, 50).round(1)
    
    # =========================================================================
    # STEP 3: Generate Comorbidities (Age and BMI correlated)
    # =========================================================================
    
    # Hypertension: Probability increases with age and BMI
    hypertension_prob = np.clip(0.05 + (age - 30) * 0.004 + (bmi - 25) * 0.01, 0.02, 0.5)
    hypertension = (np.random.random(n_samples) < hypertension_prob).astype(int)
    
    # Heart Disease: Strong correlation with age, hypertension, and BMI
    heart_disease_prob = np.clip(
        0.02 + (age - 40) * 0.003 + hypertension * 0.05 + (bmi - 30) * 0.005, 
        0.01, 0.3
    )
    heart_disease = (np.random.random(n_samples) < heart_disease_prob).astype(int)
    
    # =========================================================================
    # STEP 4: Generate Lifestyle Factors (Some correlation with health metrics)
    # =========================================================================
    
    # Diet: Correlated with BMI (people with higher BMI more likely unhealthy diet)
    diet_probs = []
    for b in bmi:
        if b >= 30:  # Obese
            diet_probs.append([0.55, 0.30, 0.15])  # More likely unhealthy
        elif b >= 25:  # Overweight
            diet_probs.append([0.35, 0.40, 0.25])
        else:  # Normal/underweight
            diet_probs.append([0.20, 0.35, 0.45])  # More likely healthy
    
    diet = np.array([
        np.random.choice(['Unhealthy', 'Mixed', 'Healthy'], p=probs)
        for probs in diet_probs
    ])
    
    # Physical Activity: Correlated with age, BMI, and heart disease
    activity_probs = []
    for i in range(n_samples):
        base_sedentary = 0.30
        base_moderate = 0.40
        base_active = 0.30
        
        # Adjust for age (older = more sedentary)
        if age[i] > 60:
            base_sedentary += 0.10
            base_active -= 0.08
        elif age[i] > 45:
            base_sedentary += 0.05
            base_active -= 0.03
            
        # Adjust for BMI (higher = more sedentary)
        if bmi[i] > 30:
            base_sedentary += 0.10
            base_active -= 0.08
        elif bmi[i] > 25:
            base_sedentary += 0.05
            base_active -= 0.03
            
        # Adjust for heart disease
        if heart_disease[i] == 1:
            base_sedentary += 0.05
            base_active -= 0.03
            
        # Ensure all probabilities are positive
        base_sedentary = max(0.1, base_sedentary)
        base_moderate = max(0.1, base_moderate)
        base_active = max(0.1, base_active)
        
        # Normalize
        total = base_sedentary + base_moderate + base_active
        activity_probs.append([
            base_sedentary / total,
            base_moderate / total, 
            base_active / total
        ])

    
    physical_activity = np.array([
        np.random.choice(['Sedentary', 'Moderately Active', 'Active'], p=probs)
        for probs in activity_probs
    ])
    
    # =========================================================================
    # STEP 5: Generate Diabetes Outcome (Based on Risk Model)
    # =========================================================================
    
    # Risk scoring based on medical literature
    # Target: ~12% diabetes prevalence
    
    diet_score = np.where(diet == 'Unhealthy', 2, np.where(diet == 'Mixed', 1, 0))
    activity_score = np.where(physical_activity == 'Sedentary', 2, 
                              np.where(physical_activity == 'Moderately Active', 1, 0))
    
    # Calculate log-odds of diabetes
    # Baseline intercept tuned for ~12% prevalence
    log_odds = np.full(n_samples, -4.0)  # Baseline
    
    # Age effect (linear, strong predictor)
    log_odds += 0.05 * (age - 30)  # Increases with age
    
    # BMI effect (strong)
    log_odds += 0.08 * (bmi - 25)  # Increases above normal BMI
    
    # Obesity bonus (additional risk)
    log_odds += np.where(bmi >= 30, 0.5, 0)
    
    # Hypertension effect
    log_odds += hypertension * 0.7
    
    # Heart disease effect
    log_odds += heart_disease * 0.6
    
    # Diet effect
    log_odds += diet_score * 0.3
    
    # Physical activity effect
    log_odds += activity_score * 0.25
    
    # Age-BMI interaction (compounding effect)
    log_odds += 0.001 * (age - 40) * (bmi - 25)
    
    # Convert to probability
    diabetes_prob = 1 / (1 + np.exp(-log_odds))
    
    # Generate binary outcome with some noise
    diabetes = (np.random.random(n_samples) < diabetes_prob).astype(int)
    
    # =========================================================================
    # STEP 6: Create DataFrame
    # =========================================================================
    
    df = pd.DataFrame({
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'Diet': diet,
        'PhysicalActivity': physical_activity,
        'diabetes': diabetes
    })
    
    # =========================================================================
    # STEP 7: Report Statistics
    # =========================================================================
    
    print("\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(df):,}")
    print(f"\n   Target Distribution:")
    print(f"   • No Diabetes (0): {(diabetes == 0).sum():,} ({(diabetes == 0).mean()*100:.1f}%)")
    print(f"   • Diabetes (1):    {(diabetes == 1).sum():,} ({(diabetes == 1).mean()*100:.1f}%)")
    
    print(f"\n   Feature Correlations with Diabetes:")
    print(f"   • Age:              {np.corrcoef(age, diabetes)[0,1]:.3f}")
    print(f"   • BMI:              {np.corrcoef(bmi, diabetes)[0,1]:.3f}")
    print(f"   • Hypertension:     {np.corrcoef(hypertension, diabetes)[0,1]:.3f}")
    print(f"   • Heart Disease:    {np.corrcoef(heart_disease, diabetes)[0,1]:.3f}")
    print(f"   • Diet Score:       {np.corrcoef(diet_score, diabetes)[0,1]:.3f}")
    print(f"   • Activity Score:   {np.corrcoef(activity_score, diabetes)[0,1]:.3f}")
    
    return df


def main():
    """Generate and save the screening dataset."""
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, 'data', 'raw', 'diabetes_screening_dataset.csv')
    
    # Generate dataset
    df = generate_screening_dataset(n_samples=100000, seed=42)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved dataset to: {output_path}")
    
    return df


if __name__ == "__main__":
    main()
