import os
import pickle
import pandas as pd
import numpy as np
import yaml
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# =============================================================================
# CONFIGURATION & LOADING
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'diabetes_model.pkl')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
X_TRAIN_PATH = os.path.join(DATA_PROCESSED_DIR, 'X_train.csv')

# Load Model
print(f"Loading model from: {MODEL_PATH}")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load Training Columns to ensure correct One-Hot Encoding
print(f"Loading training data features from: {X_TRAIN_PATH}")
try:
    X_train_df = pd.read_csv(X_TRAIN_PATH)
    model_columns = X_train_df.columns.tolist()
    print(f"Model expects these {len(model_columns)} columns.")
except Exception as e:
    print(f"Error loading X_train.csv: {e}")
    print("WARNING: Feature alignment might fail if X_train.csv is missing.")
    model_columns = []

# Define Categorical Features (Must match training)
CATEGORICAL_FEATURES = ['gender', 'smoking_history', 'occupation', 'drinking', 'altitude']

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Received prediction request: {data}")

        # 1. Create DataFrame from input
        input_df = pd.DataFrame([data])

        # 2. One-Hot Encode (get_dummies)
        # We process categorical columns same as training
        # IMPORTANT: Use drop_first=False for inference to ensure we generate the column
        # and then align it with the model columns.
        input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_FEATURES, drop_first=False, dtype=int)

        # 3. Align Columns with Model
        # Add missing columns (init with 0)
        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Remove extra columns (if any)
        input_encoded = input_encoded[model_columns]
        
        # Ensure order matches
        input_encoded = input_encoded[model_columns]

        # 4. Predict
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        result_text = "DIABETES DETECTED" if prediction == 1 else "NO DIABETES"
        print(f"Prediction: {result_text} ({probability:.4f})")

        return jsonify({
            'status': 'success',
            'prediction': int(prediction),  # 0 or 1
            'result': result_text,
            'probability': float(probability)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
