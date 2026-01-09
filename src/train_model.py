
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import pickle
import json
import argparse
from collections import Counter
import warnings
import yaml
warnings.filterwarnings('ignore')

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Create timestamped run directories
RUN_ID = f"run_{TIMESTAMP}"
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')  # Fixed path, no timestamp
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports', RUN_ID)
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

# Ensure directories exist
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load configuration
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Check if running in CLI Prediction Mode
parser = argparse.ArgumentParser(description='Diabetes Prediction Model')
parser.add_argument('--gender', type=str, help='Gender (Female/Male)')
parser.add_argument('--age', type=float, help='Age')
parser.add_argument('--hypertension', type=int, help='Hypertension (0/1)')
parser.add_argument('--heart_disease', type=int, help='Heart Disease (0/1)')
parser.add_argument('--smoking_history', type=str, help='Smoking History')
parser.add_argument('--bmi', type=float, help='BMI')
parser.add_argument('--occupation', type=str, help='Occupation')
parser.add_argument('--family_history', type=int, help='Family History of Diabetes (0/1)')
parser.add_argument('--drinking', type=str, help='Drinking habit')
parser.add_argument('--altitude', type=str, help='Altitude')
# Removed: HbA1c_level and blood_glucose_level - model now predicts without these

args, unknown = parser.parse_known_args()
CLI_PROVIDED = any(v is not None for v in vars(args).values())
VERBOSE = not CLI_PROVIDED

if VERBOSE:
    print(f"🚀 Starting run: {RUN_ID}")
    print(f"📄 Loaded config from: {CONFIG_PATH}")
    print(f"📂 Outputs will be saved to:")
    print(f"   • Models: {MODELS_DIR}")
    print(f"   • Reports: {REPORTS_DIR}")

# Configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def save_plot(filename):
    """Save plot with standard settings"""
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
# plt.show()
    # print(f"✅ Saved: {filename}")

def calculate_metrics(y_true, y_pred):
    """Calculate all classification metrics at once"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }

def plot_class_distribution(data, title, filename, labels=None):
    """Plot class distribution with bar chart"""
    counts = data.value_counts()
    labels = labels or [f'Class {i}' for i in counts.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, counts.values, color=['#2ecc71', '#e74c3c'], 
                   alpha=0.7, edgecolor='black')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12)

    for bar, v in zip(bars, counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 500,
                f'{v:,}', ha='center', fontsize=11, fontweight='bold')

    save_plot(filename)

# =============================================================================
# PART 1: DATA LOADING & EXPLORATION
# =============================================================================

# print_section("PART 1: DATA LOADING & EXPLORATION")

# Load data (using augmented dataset)
df = pd.read_csv(os.path.join(DATA_RAW, 'diabetes_prediction_dataset_augmented.csv'))
# print(f"✅ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Drop HbA1c_level and blood_glucose_level for prediction without these features
df = df.drop(columns=['HbA1c_level', 'blood_glucose_level'], errors='ignore')

# Basic info
# print(f"\n📊 Data Overview:")
# print(f"  Shape: {df.shape}")
# print(f"  Missing values: {df.isnull().sum().sum()}")
# print(f"  Duplicates: {df.duplicated().sum():,} ({df.duplicated().sum()/len(df)*100:.2f}%)")

# Target distribution
target_counts = df['diabetes'].value_counts()
imbalance_ratio = target_counts[0] / target_counts[1]
# print(f"\n📊 Target Distribution:")
# print(f"  No Diabetes: {target_counts[0]:,} ({target_counts[0]/len(df)*100:.2f}%)")
# print(f"  Diabetes: {target_counts[1]:,} ({target_counts[1]/len(df)*100:.2f}%)")
# print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")

# Feature info (HbA1c_level and blood_glucose_level excluded, new augmented features added)
numerical_features = ['age', 'bmi']
categorical_features = ['gender', 'smoking_history', 'occupation', 'drinking', 'altitude']
binary_features = ['hypertension', 'heart_disease', 'family_history']

# print(f"\n📊 Features: {df.shape[1]-1} ({len(numerical_features)} numerical, "
#       f"{len(categorical_features)} categorical, {len(binary_features)} binary)")

# =============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

# print_section("PART 2: EXPLORATORY DATA ANALYSIS")

# Target visualization
plot_class_distribution(df['diabetes'], 'Target Distribution', 
                       os.path.join(FIGURES_DIR, 'eda_target_distribution.png'), ['No Diabetes', 'Diabetes'])

# Numerical distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
for idx, feature in enumerate(numerical_features):
    axes[idx].hist(df[feature], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    axes[idx].axvline(df[feature].mean(), color='red', linestyle='--', linewidth=2)
    axes[idx].axvline(df[feature].median(), color='green', linestyle='--', linewidth=2)
save_plot(os.path.join(FIGURES_DIR, 'eda_numerical_distributions.png'))

# Feature vs Target
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
for idx, feature in enumerate(numerical_features):
    df.boxplot(column=feature, by='diabetes', ax=axes[idx], patch_artist=True, grid=False)
    axes[idx].set_title(f'{feature} vs Diabetes', fontsize=12, fontweight='bold')
    axes[idx].get_figure().suptitle('')
save_plot(os.path.join(FIGURES_DIR, 'eda_feature_vs_target.png'))

# Correlation matrix
numerical_df = df[numerical_features + binary_features + ['diabetes']]
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
save_plot(os.path.join(FIGURES_DIR, 'eda_correlation_matrix.png'))

# print(f"\n🎯 Top 3 Correlated Features with Target:")
top_features = correlation_matrix['diabetes'].abs().sort_values(ascending=False)[1:4]
# for i, (feat, corr) in enumerate(top_features.items(), 1):
#     print(f"  {i}. {feat}: {corr:.3f}")

# =============================================================================
# PART 3: DATA PREPROCESSING
# =============================================================================

# =============================================================================
# FEATURE ENGINEERING FUNCTION (SHARED)
# =============================================================================
def engineer_features(df_in):
    """
    Creates composite features and proxy biomarkers.
    Must be identical in train_model.py and app.py
    """
    df_out = df_in.copy()
    
    # 1. Comorbidity Score
    # Sum of binary conditions
    df_out['Comorbidity_Score'] = df_out['hypertension'] + df_out['heart_disease'] + df_out['family_history']
    
    # 2. Age-BMI Interaction (Obesity impact worsens with age)
    df_out['Age_BMI_Interaction'] = df_out['age'] * df_out['bmi']
    
    # 3. Lifestyle Risk Score
    # Map smoking
    smoking_map = {
        'never': 0, 'No Info': 0.5, 'current': 2, 'former': 1, 
        'ever': 1, 'not current': 0.5
    }
    df_out['smoking_score'] = df_out['smoking_history'].map(smoking_map).fillna(0)
    
    # Map drinking
    drinking_map = {
        'non_drinker': 0, 'light': 0.5, 'moderate': 1, 'heavy': 2
    }
    df_out['drinking_score'] = df_out['drinking'].map(drinking_map).fillna(0)
    
    # Occupation sedentary score (Subjective estimation)
    occ_map = {
        'office_worker': 2, 'student': 2, 'retired': 1, 'unemployed': 1,
        'healthcare': 1, 'professional': 2, 'service_industry': 0, 
        'manual_labor': 0, 'self_employed': 1
    }
    # Handle occupation map carefully if column exists
    if 'occupation' in df_out.columns:
         df_out['sedentary_score'] = df_out['occupation'].map(occ_map).fillna(1)
    else:
         df_out['sedentary_score'] = 1 # Default
         
    df_out['Lifestyle_Risk'] = df_out['smoking_score'] + df_out['drinking_score'] + df_out['sedentary_score']
    
    # 4. Metabolic Strain Index (Proxy)
    # (BMI * Age) / (Height Proxy? Don't have height). 
    # Let's use log(Age) * BMI
    df_out['Metabolic_Strain'] = np.log1p(df_out['age']) * df_out['bmi']

    # 5. Age-Family Interaction (Genetic risk often manifests later)
    df_out['Age_Family_Interaction'] = df_out['age'] * df_out['family_history']

    # 6. Cumulative Smoking Risk (Age * Smoking Score)
    df_out['Cumulative_Smoking_Risk'] = df_out['age'] * df_out['smoking_score']

    # 7. Hyper-Comorbidity Interaction
    df_out['Hyper_Comorbidity'] = (df_out['hypertension'] + df_out['heart_disease']) * df_out['family_history']
    
    # Drop intermediate columns if desired, or keep them. 
    # Dropping text columns happens later in encoding, but we can drop temp scores if we want.
    df_out = df_out.drop(columns=['smoking_score', 'drinking_score', 'sedentary_score'], errors='ignore')
    
    return df_out

# Apply Feature Engineering
df = engineer_features(df)
print("✅ Feature Engineering Complete: Added Composite Features")

# =============================================================================
# DATA PREPROCESSING (Updated for new features)
# =============================================================================

# Remove duplicates
df_clean = df.drop_duplicates(keep='first')

# Separate features and target
X = df_clean.drop('diabetes', axis=1)
y = df_clean['diabetes']

# Helper to identify new columns
new_numerical = ['Comorbidity_Score', 'Age_BMI_Interaction', 'Lifestyle_Risk', 'Metabolic_Strain',
                 'Age_Family_Interaction', 'Cumulative_Smoking_Risk', 'Hyper_Comorbidity']
# Add to numerical list for EDA if needed, but primarily for model

# Encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True, dtype=int)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# =============================================================================
# PART 5: HYPERPARAMETER TUNING & TRAINING
# =============================================================================

print_section("PART 5: HYPERPARAMETER TUNING & TRAINING")

from sklearn.model_selection import RandomizedSearchCV

# Define constraints map for new features?
# We need to re-verify column names in X_train_balanced to map constraints correctly.
# Existing constraints:
common_constraints = {
        'age': 1,
        'bmi': 1,
        'hypertension': 1,
        'heart_disease': 1,
        'family_history': 1,
        'Comorbidity_Score': 1,       # Higher = Higher Risk
        'Age_BMI_Interaction': 1,     # Higher = Higher Risk
        'Lifestyle_Risk': 1,          # Higher = Higher Risk
        'Metabolic_Strain': 1,        # Higher = Higher Risk
        'Age_Family_Interaction': 1,  # Higher = Higher Risk
        'Cumulative_Smoking_Risk': 1, # Higher = Higher Risk
        'Hyper_Comorbidity': 1        # Higher = Higher Risk
}

# We need to construct the constraint dict based on actual columns present
monotone_constraints = {}
for col in X_train_balanced.columns:
    if col in common_constraints:
        monotone_constraints[col] = common_constraints[col]
    else:
        monotone_constraints[col] = 0 # No constraint

# Initialize base model
xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    monotone_constraints=monotone_constraints
)

# Define search space
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [8.0, 10.0] # FORCED EXTREME SENSITIVITY: 8-10x penalty on missing a case
}

print("⏳ Starting Randomized Search for Hyperparameters (Sensitivity Locked)...")
search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=15, # Number of combinations to try
    scoring='recall', # Focus on catching cases (Sensitivity)
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train_balanced, y_train_balanced)

print(f"\n🏆 Best Parameters: {search.best_params_}")
print(f"🏆 Best Recall Score: {search.best_score_:.4f}")

# Use best model
model = search.best_estimator_

# model = XGBClassifier(...) # REPLACED BY TUNING ABOVE

# Train model
if VERBOSE:
    print(f"⏳ Training on {len(X_train):,} samples...")
from datetime import datetime
start_time = datetime.now()
model.fit(X_train, y_train)
training_time = (datetime.now() - start_time).total_seconds()
# print(f"✅ Training complete in {training_time:.2f}s")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
# print(f"✅ Generated predictions for {len(X_test):,} test samples")

# Quick metrics
metrics = calculate_metrics(y_test, y_pred)
# print(f"\n📊 Performance Metrics:")
# print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
# print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
# print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
# print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# print(f"\n🏆 Top 5 Important Features:")
# for idx, row in feature_importance.head(5).iterrows():
#     print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Save model
with open(os.path.join(MODELS_DIR, 'diabetes_model.pkl'), 'wb') as f:
    pickle.dump(model, f)
# print(f"\n✅ Model saved: diabetes_model.pkl")

# Save predictions
pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Probability': y_pred_proba
}).to_csv(os.path.join(REPORTS_DIR, 'predictions.csv'), index=False)
# print(f"✅ Predictions saved: predictions.csv")

# =============================================================================
# PART 6: DETAILED MODEL EVALUATION
# =============================================================================

# print_section("PART 6: MODEL EVALUATION")

# Confusion Matrix
cm = metrics['confusion_matrix']
tn, fp, fn, tp = cm['tn'], cm['fp'], cm['fn'], cm['tp']

if VERBOSE:
    print(f"\n📊 Confusion Matrix:")
    print(f"  True Negatives:  {tn:,}  |  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}  |  True Positives:  {tp:,}")

fig, ax = plt.subplots(figsize=(10, 8))
cm_array = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Healthy', 'Predicted Diabetic'],
            yticklabels=['Actually Healthy', 'Actually Diabetic'],
            annot_kws={'size': 14, 'weight': 'bold'}, ax=ax)
ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
save_plot(os.path.join(FIGURES_DIR, 'confusion_matrix.png'))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
save_plot(os.path.join(FIGURES_DIR, 'roc_curve.png'))

# print(f"\n📊 ROC-AUC Score: {roc_auc:.4f}")

# Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(recall_curve, precision_curve, color='darkblue', lw=2,
        label=f'PR Curve (AP = {avg_precision:.4f})')
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.legend()
save_plot(os.path.join(FIGURES_DIR, 'precision_recall_curve.png'))

# Classification Report
# Classification Report
if VERBOSE:
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Diabetic'], digits=4))

# Save evaluation results
evaluation_results = {
    'confusion_matrix': cm,
    'metrics': {k: float(v) for k, v in metrics.items() if k != 'confusion_matrix'},
    'roc_auc': float(roc_auc),
    'avg_precision': float(avg_precision),
    'feature_importance': feature_importance.to_dict('records')
}

with open(os.path.join(REPORTS_DIR, 'evaluation_results.json'), 'w') as f:
    json.dump(evaluation_results, f, indent=4)
# print(f"\n✅ Saved: evaluation_results.json")

# =============================================================================
# FINAL SUMMARY
# =============================================================================





# =============================================================================
# FINAL SUMMARY
# =============================================================================

# print_section("PIPELINE COMPLETED SUCCESSFULLY!")

summary = f"""
📋 FINAL SUMMARY
{'='*60}

📊 DATASET:
  • Total samples: {len(df):,} → {len(df_clean):,} (after deduplication)
  • Features: {X_encoded.shape[1]}
  • Train/Test: {len(X_train):,} / {len(X_test):,}

⚖️ IMBALANCE HANDLING:
  • Method: SMOTE + scale_pos_weight
  • Original ratio: {imbalance_ratio:.2f}:1
  • Balanced ratio: 1.0:1
  • Synthetic samples: {len(y_train_balanced) - len(y_train):,}

🤖 MODEL:
  • Algorithm: XGBoost
  • Training time: {training_time:.2f}s
  • Hyperparameters: n_estimators=100, max_depth=6

📈 PERFORMANCE:
  • Accuracy:  {metrics['accuracy']*100:.2f}%
  • Precision: {metrics['precision']*100:.2f}%
  • Recall:    {metrics['recall']*100:.2f}% ← Diabetic catch rate
  • F1-Score:  {metrics['f1_score']*100:.2f}%
  • ROC-AUC:   {roc_auc:.4f}

💊 CLINICAL IMPACT:
  • Diabetic patients: {tp + fn}
  • Successfully identified: {tp} ({metrics['recall']*100:.1f}%)
  • Missed: {fn} ({(1-metrics['recall'])*100:.1f}%)

💾 SAVED FILES:
  ✅ diabetes_model.pkl
  ✅ predictions.csv
  ✅ evaluation_results.json
  ✅ 6 visualization PNG files
  ✅ 6 CSV data files

🎯 TOP 3 IMPORTANT FEATURES:
  1. {feature_importance.iloc[0]['Feature']}: {feature_importance.iloc[0]['Importance']:.4f}
  2. {feature_importance.iloc[1]['Feature']}: {feature_importance.iloc[1]['Importance']:.4f}
  3. {feature_importance.iloc[2]['Feature']}: {feature_importance.iloc[2]['Importance']:.4f}

{'='*60}
✅ Baseline XGBoost model ready for deployment or tuning!
"""

if VERBOSE:
    print(summary)

# =============================================================================
# PART 7: PATIENT PREDICTION FROM CONFIG OR CLI
# =============================================================================

def get_patient_data(config):
    # Args already parsed at top level
    if CLI_PROVIDED:
        # Validate that all required fields are present (augmented dataset features)
        required_fields = ['gender', 'age', 'hypertension', 'heart_disease', 
                           'smoking_history', 'bmi', 'occupation', 'family_history', 'drinking', 'altitude']
        missing = [f for f in required_fields if getattr(args, f) is None]
        
        if missing:
            print(f"\n⚠️  Missing CLI arguments for: {', '.join(missing)}")
            print("   Using config data instead.")
            # If fallback to config, allow verbose? No, stick to consistent behavior or force verbose
            return config.get('test_patient')
            
        print_section("PART 7: PATIENT PREDICTION FROM CLI ARGS")
        return {
            'gender': args.gender,
            'age': args.age,
            'hypertension': args.hypertension,
            'heart_disease': args.heart_disease,
            'smoking_history': args.smoking_history,
            'bmi': args.bmi,
            'occupation': args.occupation,
            'family_history': args.family_history,
            'drinking': args.drinking,
            'altitude': args.altitude
        }
    
    if 'test_patient' in config:
        if VERBOSE:
             print_section("PART 7: PATIENT PREDICTION FROM CONFIG")
        return config['test_patient']
        
    return None

patient_data = get_patient_data(config)

if patient_data:
    if VERBOSE or CLI_PROVIDED:
        print(f"👤 Patient Data:")
        print(json.dumps(patient_data, indent=2))
    
    # Create DataFrame from patient data
    patient_df = pd.DataFrame([patient_data])
    
    # Preprocess: One-hot encode using the same method (align with training columns)
    patient_encoded = pd.get_dummies(patient_df, columns=categorical_features, drop_first=True, dtype=int)
    
    # Align columns
    missing_cols = set(X_train.columns) - set(patient_encoded.columns)
    for c in missing_cols:
        patient_encoded[c] = 0
        
    # Ensure order matches
    patient_encoded = patient_encoded[X_train.columns]
    
    # Predict
    prediction = model.predict(patient_encoded)[0]
    probability = model.predict_proba(patient_encoded)[0][1]
    
    result = "DIABETES 🔴" if prediction == 1 else "NO DIABETES 🟢"
    print(f"\n🔮 Prediction Result: {result}")
    print(f"📊 Probability: {probability:.2%}")
else:
    print("\n⚠️ No patient data found (CLI or Config)")