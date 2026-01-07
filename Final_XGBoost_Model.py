
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
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

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
    plt.show()
    print(f"✅ Saved: {filename}")

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

print_section("PART 1: DATA LOADING & EXPLORATION")

# Load data
df = pd.read_csv(r'C:\Users\Asus\Downloads\archive (6)\diabetes_prediction_dataset.csv')
print(f"✅ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Basic info
print(f"\n📊 Data Overview:")
print(f"  Shape: {df.shape}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Duplicates: {df.duplicated().sum():,} ({df.duplicated().sum()/len(df)*100:.2f}%)")

# Target distribution
target_counts = df['diabetes'].value_counts()
imbalance_ratio = target_counts[0] / target_counts[1]
print(f"\n📊 Target Distribution:")
print(f"  No Diabetes: {target_counts[0]:,} ({target_counts[0]/len(df)*100:.2f}%)")
print(f"  Diabetes: {target_counts[1]:,} ({target_counts[1]/len(df)*100:.2f}%)")
print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")

# Feature info
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'smoking_history']
binary_features = ['hypertension', 'heart_disease']

print(f"\n📊 Features: {df.shape[1]-1} ({len(numerical_features)} numerical, "
      f"{len(categorical_features)} categorical, {len(binary_features)} binary)")

# =============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print_section("PART 2: EXPLORATORY DATA ANALYSIS")

# Target visualization
plot_class_distribution(df['diabetes'], 'Target Distribution', 
                       'eda_target_distribution.png', ['No Diabetes', 'Diabetes'])

# Numerical distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
for idx, feature in enumerate(numerical_features):
    axes[idx].hist(df[feature], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    axes[idx].axvline(df[feature].mean(), color='red', linestyle='--', linewidth=2)
    axes[idx].axvline(df[feature].median(), color='green', linestyle='--', linewidth=2)
save_plot('eda_numerical_distributions.png')

# Feature vs Target
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
for idx, feature in enumerate(numerical_features):
    df.boxplot(column=feature, by='diabetes', ax=axes[idx], patch_artist=True, grid=False)
    axes[idx].set_title(f'{feature} vs Diabetes', fontsize=12, fontweight='bold')
    axes[idx].get_figure().suptitle('')
save_plot('eda_feature_vs_target.png')

# Correlation matrix
numerical_df = df[numerical_features + binary_features + ['diabetes']]
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
save_plot('eda_correlation_matrix.png')

print(f"\n🎯 Top 3 Correlated Features with Target:")
top_features = correlation_matrix['diabetes'].abs().sort_values(ascending=False)[1:4]
for i, (feat, corr) in enumerate(top_features.items(), 1):
    print(f"  {i}. {feat}: {corr:.3f}")

# =============================================================================
# PART 3: DATA PREPROCESSING
# =============================================================================

print_section("PART 3: DATA PREPROCESSING")

# Remove duplicates
df_clean = df.drop_duplicates(keep='first')
print(f"✅ Removed {len(df) - len(df_clean):,} duplicates → {len(df_clean):,} rows")

# Separate features and target
X = df_clean.drop('diabetes', axis=1)
y = df_clean['diabetes']

# Encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True, dtype=int)
print(f"✅ Encoded categorical features: {X.shape[1]} → {X_encoded.shape[1]} features")

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train-test split: {len(X_train):,} train, {len(X_test):,} test")

# Save preprocessed data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print(f"✅ Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

# =============================================================================
# PART 4: HANDLING CLASS IMBALANCE
# =============================================================================

print_section("PART 4: HANDLING CLASS IMBALANCE")

# Current imbalance
train_counts = Counter(y_train)
print(f"📊 Training Set Before SMOTE:")
print(f"  Class 0: {train_counts[0]:,}, Class 1: {train_counts[1]:,}")
print(f"  Ratio: {train_counts[0]/train_counts[1]:.2f}:1")

# Apply SMOTE
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
balanced_counts = Counter(y_train_balanced)
print(f"\n✅ After SMOTE:")
print(f"  Class 0: {balanced_counts[0]:,}, Class 1: {balanced_counts[1]:,}")
print(f"  Synthetic samples created: {len(y_train_balanced) - len(y_train):,}")

# Calculate class weights (alternative)
scale_pos_weight = train_counts[0] / train_counts[1]
print(f"\n📊 XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(['No Diabetes', 'Diabetes'], [train_counts[0], train_counts[1]], 
            color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0].set_title('Before SMOTE', fontsize=14, fontweight='bold')
axes[1].bar(['No Diabetes', 'Diabetes'], [balanced_counts[0], balanced_counts[1]], 
            color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[1].set_title('After SMOTE', fontsize=14, fontweight='bold')
save_plot('imbalance_handling.png')

# Save balanced data
X_train_balanced.to_csv('X_train_balanced.csv', index=False)
y_train_balanced.to_csv('y_train_balanced.csv', index=False)
print(f"✅ Saved: X_train_balanced.csv, y_train_balanced.csv")

# =============================================================================
# PART 5: TRAINING XGBOOST MODEL
# =============================================================================

print_section("PART 5: TRAINING XGBOOST MODEL")

# Initialize model with class weights
model = XGBClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.3,
    max_depth=6,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False
)

print(f"🔧 Model Config: n_estimators=100, max_depth=6, scale_pos_weight={scale_pos_weight:.2f}")

# Train model
print(f"⏳ Training on {len(X_train):,} samples...")
from datetime import datetime
start_time = datetime.now()
model.fit(X_train, y_train)
training_time = (datetime.now() - start_time).total_seconds()
print(f"✅ Training complete in {training_time:.2f}s")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
print(f"✅ Generated predictions for {len(X_test):,} test samples")

# Quick metrics
metrics = calculate_metrics(y_test, y_pred)
print(f"\n📊 Performance Metrics:")
print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n🏆 Top 5 Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Save model
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f"\n✅ Model saved: diabetes_model.pkl")

# Save predictions
pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Probability': y_pred_proba
}).to_csv('predictions.csv', index=False)
print(f"✅ Predictions saved: predictions.csv")

# =============================================================================
# PART 6: DETAILED MODEL EVALUATION
# =============================================================================

print_section("PART 6: MODEL EVALUATION")

# Confusion Matrix
cm = metrics['confusion_matrix']
tn, fp, fn, tp = cm['tn'], cm['fp'], cm['fn'], cm['tp']

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
save_plot('confusion_matrix.png')

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
save_plot('roc_curve.png')

print(f"\n📊 ROC-AUC Score: {roc_auc:.4f}")

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
save_plot('precision_recall_curve.png')

# Classification Report
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

with open('evaluation_results.json', 'w') as f:
    json.dump(evaluation_results, f, indent=4)
print(f"\n✅ Saved: evaluation_results.json")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print_section("PIPELINE COMPLETED SUCCESSFULLY!")

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

print(summary)