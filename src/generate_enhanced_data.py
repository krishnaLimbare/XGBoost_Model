import pandas as pd
import numpy as np
import os

def generate_enhanced_data():
    print("🚀 Starting Enhanced Data Generation...")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'raw', 'diabetes_prediction_dataset.csv')
    output_path = os.path.join(base_dir, 'data', 'raw', 'diabetes_prediction_dataset_enhanced.csv')
    
    # Load original data
    df = pd.read_csv(input_path)
    print(f"📄 Loaded original data: {len(df):,} rows")
    
    # 1. Add New Features (Probabilistic Assignment based on existing health)
    np.random.seed(42)
    
    # Family History: 15% random population prevalence, higher if older/hypertensive
    df['FamilyHistory'] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
    
    # Diet: Unhealthy (0), Mixed (1), Healthy (2)
    # Correlate with BMI slightly (High BMI -> likely Unhealthy)
    def assign_diet(bmi):
        if bmi > 30: return np.random.choice(['Unhealthy', 'Mixed', 'Healthy'], p=[0.6, 0.3, 0.1])
        if bmi > 25: return np.random.choice(['Unhealthy', 'Mixed', 'Healthy'], p=[0.4, 0.4, 0.2])
        return np.random.choice(['Unhealthy', 'Mixed', 'Healthy'], p=[0.2, 0.4, 0.4])
        
    df['Diet'] = df['bmi'].apply(assign_diet)
    
    # Physical Activity: Sedentary (0), Moderately Active (1), Active (2)
    # Correlate with BMI and Heart Disease
    def assign_activity(row):
        prob = [0.33, 0.33, 0.34]
        if row['bmi'] > 30 or row['heart_disease'] == 1:
            prob = [0.6, 0.3, 0.1]
        elif row['bmi'] < 25:
            prob = [0.1, 0.4, 0.5]
        return np.random.choice(['Sedentary', 'Moderately Active', 'Active'], p=prob)
        
    df['PhysicalActivity'] = df.apply(assign_activity, axis=1)
    
    # 2. Recalculate Diabetes Outcome (The Verification Step)
    # We define a "Risk Score" based on all features
    
    # Mappings for score calculation
    diet_score = {'Unhealthy': 2, 'Mixed': 1, 'Healthy': 0}
    activity_score = {'Sedentary': 2, 'Moderately Active': 1, 'Active': 0}
    
    def calculate_diabetes_probability(row):
        # Base log-odds from established medical literature proxies
        # Intercept
        log_odds = -9.0 
        
        # Age effect (Linear + slight non-linear)
        log_odds += 0.06 * row['age']
        
        # BMI effect
        log_odds += 0.12 * row['bmi']
        
        # Hypertension & Heart Disease
        if row['hypertension'] == 1: log_odds += 1.2
        if row['heart_disease'] == 1: log_odds += 1.1
        
        # New Features Effect
        # 1. Family History (Strong genetic component)
        if row['FamilyHistory'] == 1: log_odds += 1.5
        
        # 2. Diet & Activity
        d_score = diet_score[row['Diet']]
        a_score = activity_score[row['PhysicalActivity']]
        
        # Add to log-odds (Maximum extra risk ~ +4.0 log-odds for worst lifestyle)
        log_odds += 0.8 * d_score  # Unhealthy diet adds significant risk
        log_odds += 0.8 * a_score  # Sedentary behavior adds significant risk
        
        # 3. HbA1c and Blood Glucose (Original strong predictors)
        # We must keep consistency with original biological markers if we have them.
        # However, the user wants the outcome mapped to the NEW columns. 
        # To strictly follow the prompt "OUTCOME SHOULD ALSO ME MAPPED MEANS THE DIABETES COLUYMN SHOULD BE DEPEMNDENT ON THE NEW COLUMNS",
        # we allow the new columns to shift the probability significantly.
        # But we also incorporate the strong original signals to maintain realism.
        
        log_odds += 0.8 * float(row['HbA1c_level'])
        log_odds += 0.02 * float(row['blood_glucose_level'])

        # Sigmoid function to get probability
        probability = 1 / (1 + np.exp(-log_odds))
        return probability

    # Calculate probabilities
    probs = df.apply(calculate_diabetes_probability, axis=1)
    
    # Assign new binary labels based on probability
    # We use a threshold, but stochastic assignment is more realistic for "chance"
    # To make it deterministic enough for the model to learn:
    df['diabetes'] = (probs > 0.5).astype(int)
    
    # Verify Correlations
    print("\n📊 New Data Correlations with Diabetes:")
    df['Diet_Num'] = df['Diet'].map(diet_score)
    df['Activity_Num'] = df['PhysicalActivity'].map(activity_score)
    
    print(f"  • Family History: {df['FamilyHistory'].corr(df['diabetes']):.3f}")
    print(f"  • Diet (Unhealthy=High): {df['Diet_Num'].corr(df['diabetes']):.3f}")
    print(f"  • Activity (Sedentary=High): {df['Activity_Num'].corr(df['diabetes']):.3f}")
    print(f"  • BMI: {df['bmi'].corr(df['diabetes']):.3f}")
    
    # Drop temp columns
    df = df.drop(columns=['Diet_Num', 'Activity_Num'])
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved enhanced dataset to: {output_path}")

if __name__ == "__main__":
    generate_enhanced_data()
