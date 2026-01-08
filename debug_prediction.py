import pandas as pd
import pickle
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'diabetes_model.pkl')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
X_TRAIN_PATH = os.path.join(DATA_PROCESSED_DIR, 'X_train.csv')

# Load Model
print(f"Loading model from: {MODEL_PATH}")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load Training Columns
print(f"Loading training data features from: {X_TRAIN_PATH}")
X_train_df = pd.read_csv(X_TRAIN_PATH)
model_columns = X_train_df.columns.tolist()
print(f"Model expects these {len(model_columns)} columns: {model_columns}")

# Sample Input (from subagent test)
data = {
    'gender': 'Male',
    'age': 45.0,
    'bmi': 28.5,
    'hypertension': 0,
    'heart_disease': 0,
    'family_history': 1,
    'smoking_history': 'former',
    'drinking': 'moderate',
    'occupation': 'office_worker',
    'altitude': 'medium_500-1500m'
}

print(f"\nProcessing input: {data}")

try:
    # 1. Create DataFrame
    input_df = pd.DataFrame([data])

    # 2. One-Hot Encode
    CATEGORICAL_FEATURES = ['gender', 'smoking_history', 'occupation', 'drinking', 'altitude']
    input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_FEATURES, drop_first=False, dtype=int)
    
    print("\nColumns after encoding:")
    print(input_encoded.columns.tolist())

    # 3. Align Columns
    for col in model_columns:
        if col not in input_encoded.columns:
            print(f"Missing col: {col} -> filling with 0")
            input_encoded[col] = 0
    
    # Remove extra
    extra_cols = set(input_encoded.columns) - set(model_columns)
    if extra_cols:
        print(f"Extra cols (to remove): {extra_cols}")
        
    input_encoded = input_encoded[model_columns]

    # 4. Predict
    print("\nPredicting...")
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    print(f"Result: {prediction}, Prob: {probability}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
