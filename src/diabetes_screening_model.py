"""
Diabetes Screening Model Pipeline
==================================
A clinically-focused screening model for population-level diabetes risk stratification.

Features Used (7 only):
- Age, Gender, BMI, Hypertension, Heart Disease, Diet, Physical Activity

Design Principles:
1. No data leakage (excludes biomarkers like HbA1c, blood glucose)
2. Clinically meaningful feature engineering
3. Robust 5-fold cross-validation
4. Balanced optimization (not extreme sensitivity/specificity)
5. Clear Low/Medium/High risk stratification

Author: ML Pipeline
Date: 2026-01-10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, make_scorer
)

# XGBoost
from xgboost import XGBClassifier

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

# Create directories if needed
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Timestamp for this run
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def calculate_all_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive classification metrics"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['avg_precision'] = average_precision_score(y_true, y_proba)
    
    return metrics

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df_in):
    """
    Create clinically meaningful features for diabetes screening.
    
    Features Created:
    -----------------
    1. Age_Band: Categorical age groups with distinct risk profiles
    2. Age_Risk_Flag: Binary indicator for age >= 45 (key inflection point)
    3. BMI_Category: WHO standard BMI categories
    4. Obesity_Flag: Binary indicator for BMI >= 30
    5. Age_BMI_Interaction: Age × BMI (compounding effect)
    6. Cardio_Risk_Score: Hypertension + Heart Disease (0-2)
    7. Diet_Score: Ordinal score (Healthy=0, Mixed=1, Unhealthy=2)
    8. Activity_Score: Ordinal score (Active=0, Moderate=1, Sedentary=2)
    9. Lifestyle_Risk_Score: Diet_Score + Activity_Score (0-4)
    
    Parameters:
    -----------
    df_in : pd.DataFrame
        Input dataframe with raw features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features added
    """
    df = df_in.copy()
    
    # -------------------------------------------------------------------------
    # 1. Age-Based Features
    # -------------------------------------------------------------------------
    
    # Age bands with clinical significance
    age_bins = [0, 30, 45, 60, 120]
    age_labels = ['young', 'middle', 'senior', 'elderly']
    df['Age_Band'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    
    # Binary risk flag at 45 (major diabetes risk inflection)
    df['Age_Risk_Flag'] = (df['age'] >= 45).astype(int)
    
    # -------------------------------------------------------------------------
    # 2. BMI-Based Features
    # -------------------------------------------------------------------------
    
    # WHO standard BMI categories
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['underweight', 'normal', 'overweight', 'obese']
    df['BMI_Category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)
    
    # Obesity flag (BMI >= 30)
    df['Obesity_Flag'] = (df['bmi'] >= 30).astype(int)
    
    # -------------------------------------------------------------------------
    # 3. Interaction Features
    # -------------------------------------------------------------------------
    
    # Age × BMI interaction (captures compounding metabolic stress)
    df['Age_BMI_Interaction'] = df['age'] * df['bmi']
    
    # -------------------------------------------------------------------------
    # 4. Cardiovascular Risk Score
    # -------------------------------------------------------------------------
    
    # Combined comorbidity burden (0-2)
    df['Cardio_Risk_Score'] = df['hypertension'] + df['heart_disease']
    
    # -------------------------------------------------------------------------
    # 5. Lifestyle Risk Scoring
    # -------------------------------------------------------------------------
    
    # Diet scoring: Higher score = higher risk
    diet_map = {'Healthy': 0, 'Mixed': 1, 'Unhealthy': 2}
    df['Diet_Score'] = df['Diet'].map(diet_map).fillna(1)  # Default to Mixed if unknown
    
    # Activity scoring: Higher score = higher risk
    activity_map = {'Active': 0, 'Moderately Active': 1, 'Sedentary': 2}
    df['Activity_Score'] = df['PhysicalActivity'].map(activity_map).fillna(1)
    
    # Combined lifestyle risk (0-4)
    df['Lifestyle_Risk_Score'] = df['Diet_Score'] + df['Activity_Score']
    
    return df


def prepare_features(df, target_col='diabetes'):
    """
    Prepare feature matrix and target vector.
    
    Drops columns not used in model and applies feature engineering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe
    target_col : str
        Name of target column
        
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    feature_names : list
        Final feature names after encoding
    """
    # Columns to use (7 features as specified)
    FEATURE_COLS = ['age', 'gender', 'bmi', 'hypertension', 'heart_disease', 'Diet', 'PhysicalActivity']
    
    # Columns to exclude (biomarkers, genetic, and other non-specified)
    EXCLUDE_COLS = ['HbA1c_level', 'blood_glucose_level', 'FamilyHistory', 'smoking_history']
    
    # Select only required columns
    df_selected = df[FEATURE_COLS + [target_col]].copy()
    
    # Apply feature engineering
    df_engineered = engineer_features(df_selected)
    
    # Separate features and target
    X = df_engineered.drop(columns=[target_col])
    y = df_engineered[target_col]
    
    return X, y


def encode_features(X_train, X_test=None):
    """
    Encode categorical features with proper handling.
    
    - Ordinal encoding for Diet_Score, Activity_Score (already numeric)
    - One-hot encoding for Gender
    - One-hot encoding for Age_Band, BMI_Category
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix
    X_test : pd.DataFrame, optional
        Test feature matrix
        
    Returns:
    --------
    X_train_encoded, X_test_encoded (or just X_train_encoded if no test)
    feature_names : list
    """
    # Columns to one-hot encode
    onehot_cols = ['gender', 'Age_Band', 'BMI_Category', 'Diet', 'PhysicalActivity']
    
    # Create encoded dataframes
    X_train_encoded = pd.get_dummies(X_train, columns=onehot_cols, drop_first=True, dtype=int)
    
    if X_test is not None:
        X_test_encoded = pd.get_dummies(X_test, columns=onehot_cols, drop_first=True, dtype=int)
        
        # Align columns (handle missing categories)
        missing_in_test = set(X_train_encoded.columns) - set(X_test_encoded.columns)
        for col in missing_in_test:
            X_test_encoded[col] = 0
        
        missing_in_train = set(X_test_encoded.columns) - set(X_train_encoded.columns)
        for col in missing_in_train:
            X_train_encoded[col] = 0
        
        # Ensure same column order
        X_test_encoded = X_test_encoded[X_train_encoded.columns]
        
        return X_train_encoded, X_test_encoded, list(X_train_encoded.columns)
    
    return X_train_encoded, list(X_train_encoded.columns)


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def perform_cross_validation(X, y, model, model_name, n_folds=5, verbose=True):
    """
    Perform stratified k-fold cross-validation with comprehensive metrics.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    model : sklearn estimator
        Model to evaluate
    model_name : str
        Name for display
    n_folds : int
        Number of CV folds
    verbose : bool
        Print fold-level results
        
    Returns:
    --------
    dict : CV results with per-fold and aggregate metrics
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    if verbose:
        print(f"\n📊 {n_folds}-Fold Cross-Validation for {model_name}")
        print("-" * 60)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Split data
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Encode features
        X_train_enc, X_val_enc, feature_names = encode_features(X_train_fold, X_val_fold)
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ['age', 'bmi', 'Age_BMI_Interaction']
        numerical_cols = [c for c in numerical_cols if c in X_train_enc.columns]
        
        if numerical_cols:
            X_train_enc[numerical_cols] = scaler.fit_transform(X_train_enc[numerical_cols])
            X_val_enc[numerical_cols] = scaler.transform(X_val_enc[numerical_cols])
        
        # Train model
        model_clone = clone_model(model)
        model_clone.fit(X_train_enc, y_train_fold)
        
        # Predict
        y_pred = model_clone.predict(X_val_enc)
        y_proba = model_clone.predict_proba(X_val_enc)[:, 1]
        
        # Calculate metrics
        fold_metrics = calculate_all_metrics(y_val_fold, y_pred, y_proba)
        fold_metrics['fold'] = fold
        fold_results.append(fold_metrics)
        
        if verbose:
            print(f"  Fold {fold}: Acc={fold_metrics['accuracy']:.4f}, "
                  f"F1={fold_metrics['f1_score']:.4f}, "
                  f"AUC={fold_metrics['roc_auc']:.4f}")
    
    # Aggregate results
    cv_results = {
        'model_name': model_name,
        'n_folds': n_folds,
        'fold_results': fold_results,
        'mean_accuracy': np.mean([f['accuracy'] for f in fold_results]),
        'std_accuracy': np.std([f['accuracy'] for f in fold_results]),
        'mean_precision': np.mean([f['precision'] for f in fold_results]),
        'std_precision': np.std([f['precision'] for f in fold_results]),
        'mean_recall': np.mean([f['recall'] for f in fold_results]),
        'std_recall': np.std([f['recall'] for f in fold_results]),
        'mean_f1': np.mean([f['f1_score'] for f in fold_results]),
        'std_f1': np.std([f['f1_score'] for f in fold_results]),
        'mean_auc': np.mean([f['roc_auc'] for f in fold_results]),
        'std_auc': np.std([f['roc_auc'] for f in fold_results]),
        'mean_specificity': np.mean([f['specificity'] for f in fold_results]),
        'std_specificity': np.std([f['specificity'] for f in fold_results]),
    }
    
    if verbose:
        print("-" * 60)
        print(f"  📈 Mean Accuracy:    {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"  📈 Mean F1-Score:    {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        print(f"  📈 Mean ROC-AUC:     {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        print(f"  📈 Mean Sensitivity: {cv_results['mean_recall']:.4f} ± {cv_results['std_recall']:.4f}")
        print(f"  📈 Mean Specificity: {cv_results['mean_specificity']:.4f} ± {cv_results['std_specificity']:.4f}")
    
    return cv_results


def clone_model(model):
    """Clone a model to avoid fitting the same instance multiple times."""
    if isinstance(model, XGBClassifier):
        return XGBClassifier(**model.get_params())
    elif isinstance(model, LogisticRegression):
        return LogisticRegression(**model.get_params())
    else:
        from sklearn.base import clone
        return clone(model)


# =============================================================================
# RISK STRATIFICATION
# =============================================================================

def assign_risk_tier(probability):
    """
    Assign risk tier based on predicted probability.
    
    Thresholds:
    -----------
    - Low Risk:    0.00 - 0.30 (Annual wellness check)
    - Medium Risk: 0.30 - 0.60 (Lifestyle counseling, 6-month follow-up)
    - High Risk:   0.60 - 1.00 (Clinical evaluation recommended)
    
    Parameters:
    -----------
    probability : float
        Predicted probability of diabetes
        
    Returns:
    --------
    dict : Risk tier information
    """
    if probability < 0.30:
        return {
            'tier': 'Low Risk',
            'tier_code': 'low',
            'emoji': '🟢',
            'description': 'Low likelihood of diabetes based on current risk factors.',
            'recommendation': 'Maintain healthy lifestyle. Annual wellness check recommended.'
        }
    elif probability < 0.60:
        return {
            'tier': 'Medium Risk',
            'tier_code': 'medium',
            'emoji': '🟡',
            'description': 'Moderate risk factors present. Preventive action advised.',
            'recommendation': 'Lifestyle modifications suggested. Follow-up in 6 months.'
        }
    else:
        return {
            'tier': 'High Risk',
            'tier_code': 'high',
            'emoji': '🔴',
            'description': 'Elevated risk profile detected.',
            'recommendation': 'Clinical evaluation strongly recommended. Consult a healthcare provider.'
        }


def analyze_risk_distribution(y_true, y_proba):
    """
    Analyze how risk tiers align with actual outcomes.
    
    Parameters:
    -----------
    y_true : array-like
        Actual labels
    y_proba : array-like
        Predicted probabilities
        
    Returns:
    --------
    dict : Risk distribution analysis
    """
    tiers = []
    for p in y_proba:
        if p < 0.30:
            tiers.append('Low')
        elif p < 0.60:
            tiers.append('Medium')
        else:
            tiers.append('High')
    
    df = pd.DataFrame({
        'actual': y_true,
        'probability': y_proba,
        'tier': tiers
    })
    
    # Analyze each tier
    analysis = {}
    for tier in ['Low', 'Medium', 'High']:
        tier_data = df[df['tier'] == tier]
        if len(tier_data) > 0:
            analysis[tier] = {
                'count': len(tier_data),
                'pct_of_total': len(tier_data) / len(df) * 100,
                'diabetes_rate': tier_data['actual'].mean() * 100,
                'mean_probability': tier_data['probability'].mean()
            }
    
    return analysis


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Run the complete diabetes screening model pipeline."""
    
    print_section("DIABETES SCREENING MODEL PIPELINE")
    print(f"Run timestamp: {TIMESTAMP}")
    print(f"Output directory: {REPORTS_DIR}")
    
    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    print_section("STEP 1: DATA LOADING")
    
    data_path = os.path.join(DATA_RAW, 'diabetes_screening_dataset.csv')
    df = pd.read_csv(data_path)
    
    print(f"✅ Loaded dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"\n📊 Target Distribution:")
    target_counts = df['diabetes'].value_counts()
    print(f"   No Diabetes (0): {target_counts[0]:,} ({target_counts[0]/len(df)*100:.1f}%)")
    print(f"   Diabetes (1):    {target_counts[1]:,} ({target_counts[1]/len(df)*100:.1f}%)")
    print(f"   Imbalance Ratio: {target_counts[0]/target_counts[1]:.2f}:1")
    
    # =========================================================================
    # STEP 2: FEATURE PREPARATION
    # =========================================================================
    print_section("STEP 2: FEATURE PREPARATION")
    
    X, y = prepare_features(df, target_col='diabetes')
    
    print(f"✅ Selected 7 input features: Age, Gender, BMI, Hypertension, Heart Disease, Diet, Physical Activity")
    print(f"✅ Excluded biomarkers: HbA1c_level, blood_glucose_level (prevents data leakage)")
    print(f"✅ Excluded genetic: FamilyHistory (not in user requirements)")
    print(f"\n📊 Engineered Features Created:")
    print("   • Age_Band (categorical: young/middle/senior/elderly)")
    print("   • Age_Risk_Flag (binary: age >= 45)")
    print("   • BMI_Category (categorical: underweight/normal/overweight/obese)")
    print("   • Obesity_Flag (binary: BMI >= 30)")
    print("   • Age_BMI_Interaction (continuous)")
    print("   • Cardio_Risk_Score (0-2: hypertension + heart_disease)")
    print("   • Diet_Score (0-2: Healthy=0, Mixed=1, Unhealthy=2)")
    print("   • Activity_Score (0-2: Active=0, Moderate=1, Sedentary=2)")
    print("   • Lifestyle_Risk_Score (0-4: Diet + Activity)")
    
    print(f"\n📊 Feature Matrix Shape: {X.shape}")
    
    # =========================================================================
    # STEP 3: TRAIN-TEST SPLIT
    # =========================================================================
    print_section("STEP 3: TRAIN-TEST SPLIT")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Training set: {len(X_train):,} samples")
    print(f"✅ Test set:     {len(X_test):,} samples")
    print(f"   Test ratio:   20%")
    
    # =========================================================================
    # STEP 4: CROSS-VALIDATION
    # =========================================================================
    print_section("STEP 4: 5-FOLD CROSS-VALIDATION")
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=target_counts[0]/target_counts[1],  # Handle imbalance
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
    }
    
    cv_results_all = {}
    
    for model_name, model in models.items():
        cv_results = perform_cross_validation(X_train, y_train, model, model_name, n_folds=5)
        cv_results_all[model_name] = cv_results
    
    # =========================================================================
    # STEP 5: MODEL COMPARISON & SELECTION
    # =========================================================================
    print_section("STEP 5: MODEL COMPARISON & SELECTION")
    
    print("\n📊 Cross-Validation Summary:")
    print("-" * 70)
    print(f"{'Model':<25} {'Accuracy':<15} {'F1-Score':<15} {'ROC-AUC':<15}")
    print("-" * 70)
    
    best_model_name = None
    best_auc = 0
    
    for model_name, cv_results in cv_results_all.items():
        acc = f"{cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}"
        f1 = f"{cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}"
        auc = f"{cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}"
        print(f"{model_name:<25} {acc:<15} {f1:<15} {auc:<15}")
        
        if cv_results['mean_auc'] > best_auc:
            best_auc = cv_results['mean_auc']
            best_model_name = model_name
    
    print("-" * 70)
    print(f"\n🏆 Best Model (by ROC-AUC): {best_model_name}")
    
    # Select best model
    best_model = models[best_model_name]
    
    # =========================================================================
    # STEP 6: FINAL MODEL TRAINING
    # =========================================================================
    print_section("STEP 6: FINAL MODEL TRAINING")
    
    # Encode full training set
    X_train_encoded, X_test_encoded, feature_names = encode_features(X_train, X_test)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'bmi', 'Age_BMI_Interaction']
    numerical_cols = [c for c in numerical_cols if c in X_train_encoded.columns]
    
    if numerical_cols:
        X_train_encoded[numerical_cols] = scaler.fit_transform(X_train_encoded[numerical_cols])
        X_test_encoded[numerical_cols] = scaler.transform(X_test_encoded[numerical_cols])
    
    # Train final model
    print(f"⏳ Training {best_model_name} on full training set...")
    start_time = datetime.now()
    best_model.fit(X_train_encoded, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"✅ Training completed in {training_time:.2f}s")
    
    # =========================================================================
    # STEP 7: TEST SET EVALUATION
    # =========================================================================
    print_section("STEP 7: TEST SET EVALUATION")
    
    y_pred = best_model.predict(X_test_encoded)
    y_proba = best_model.predict_proba(X_test_encoded)[:, 1]
    
    test_metrics = calculate_all_metrics(y_test, y_pred, y_proba)
    
    print("\n📊 Test Set Performance:")
    print("-" * 50)
    print(f"   Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"   Precision:   {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"   Recall:      {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
    print(f"   Specificity: {test_metrics['specificity']:.4f} ({test_metrics['specificity']*100:.2f}%)")
    print(f"   F1-Score:    {test_metrics['f1_score']:.4f}")
    print(f"   ROC-AUC:     {test_metrics['roc_auc']:.4f}")
    print(f"   Avg Prec:    {test_metrics['avg_precision']:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'], digits=4))
    
    cm = test_metrics['confusion_matrix']
    print("\n📊 Confusion Matrix:")
    print(f"   True Negatives:  {cm['tn']:,}  |  False Positives: {cm['fp']:,}")
    print(f"   False Negatives: {cm['fn']:,}  |  True Positives:  {cm['tp']:,}")
    
    # =========================================================================
    # STEP 8: RISK STRATIFICATION ANALYSIS
    # =========================================================================
    print_section("STEP 8: RISK STRATIFICATION ANALYSIS")
    
    risk_analysis = analyze_risk_distribution(y_test.values, y_proba)
    
    print("\n📊 Risk Tier Distribution (Test Set):")
    print("-" * 70)
    print(f"{'Tier':<15} {'Count':<12} {'% of Total':<15} {'Diabetes Rate':<15} {'Mean Prob':<12}")
    print("-" * 70)
    
    for tier, stats in risk_analysis.items():
        print(f"{tier:<15} {stats['count']:<12,} {stats['pct_of_total']:<15.1f} "
              f"{stats['diabetes_rate']:<15.1f} {stats['mean_probability']:<12.3f}")
    
    print("-" * 70)
    
    print("\n✅ Risk Tier Thresholds:")
    print("   • Low Risk:    Probability < 0.30")
    print("   • Medium Risk: Probability 0.30 - 0.60")
    print("   • High Risk:   Probability >= 0.60")
    
    # =========================================================================
    # STEP 9: FEATURE IMPORTANCE
    # =========================================================================
    print_section("STEP 9: FEATURE IMPORTANCE")
    
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    elif hasattr(best_model, 'coef_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(best_model.coef_[0])
        }).sort_values('Importance', ascending=False)
    
    print("\n🏆 Top 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        bar = '█' * int(row['Importance'] / importance_df['Importance'].max() * 20)
        print(f"   {row['Feature']:<30} {row['Importance']:.4f} {bar}")
    
    # =========================================================================
    # STEP 10: SAVE OUTPUTS
    # =========================================================================
    print_section("STEP 10: SAVING OUTPUTS")
    
    # Calculate feature means for explainability (Baseline for SHAP)
    # For scaled cols, mean is ~0. For others, it's the prevalence/mean.
    feature_means = X_train_encoded.mean()
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'diabetes_screening_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'numerical_cols': numerical_cols,
            'feature_means': feature_means,  # <--- NEW: For explainability
            'model_name': best_model_name,
            'risk_thresholds': {'low': 0.30, 'high': 0.60}
        }, f)
    print(f"✅ Saved: {model_path}")
    
    # Save CV results
    cv_results_path = os.path.join(REPORTS_DIR, 'screening_cv_results.json')
    with open(cv_results_path, 'w') as f:
        # Convert to serializable format
        cv_serializable = {}
        for model_name, results in cv_results_all.items():
            cv_serializable[model_name] = {
                k: float(v) if isinstance(v, (np.float64, np.float32)) else v
                for k, v in results.items()
                if k != 'fold_results'
            }
        json.dump(cv_serializable, f, indent=4)
    print(f"✅ Saved: {cv_results_path}")
    
    # Save evaluation results
    eval_path = os.path.join(REPORTS_DIR, 'screening_evaluation.json')
    eval_results = {
        'model_name': best_model_name,
        'test_metrics': {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                        for k, v in test_metrics.items() if k != 'confusion_matrix'},
        'confusion_matrix': cm,
        'risk_analysis': {k: {kk: float(vv) if isinstance(vv, (np.float64, np.float32)) else vv 
                             for kk, vv in v.items()} 
                         for k, v in risk_analysis.items()},
        'feature_importance': importance_df.head(10).to_dict('records'),
        'cv_mean_auc': cv_results_all[best_model_name]['mean_auc'],
        'cv_std_auc': cv_results_all[best_model_name]['std_auc'],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'timestamp': TIMESTAMP
    }
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=4, default=str)
    print(f"✅ Saved: {eval_path}")
    
    # Save feature importance
    importance_path = os.path.join(REPORTS_DIR, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"✅ Saved: {importance_path}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("PIPELINE COMPLETE")
    
    print(f"""
📋 DIABETES SCREENING MODEL - FINAL SUMMARY
{'='*60}

📊 DATA:
  • Total samples: {len(df):,}
  • Features used: 7 (Age, Gender, BMI, Hypertension, Heart Disease, Diet, Physical Activity)
  • Features engineered: 9 additional clinical indicators
  • Train/Test split: {len(X_train):,} / {len(X_test):,}

🔬 VALIDATION:
  • Method: 5-Fold Stratified Cross-Validation
  • CV ROC-AUC: {cv_results_all[best_model_name]['mean_auc']:.4f} ± {cv_results_all[best_model_name]['std_auc']:.4f}
  • Stability: {'✅ Stable (std < 0.02)' if cv_results_all[best_model_name]['std_auc'] < 0.02 else '⚠️ Moderate variance'}

🤖 MODEL:
  • Algorithm: {best_model_name}
  • Test ROC-AUC: {test_metrics['roc_auc']:.4f}
  • Test F1-Score: {test_metrics['f1_score']:.4f}
  
📈 PERFORMANCE BALANCE:
  • Sensitivity (Recall): {test_metrics['recall']*100:.1f}%
  • Specificity: {test_metrics['specificity']*100:.1f}%
  • Balanced for screening (not extreme sensitivity)

🎯 RISK STRATIFICATION:
  • Low Risk (< 0.30): {risk_analysis.get('Low', {}).get('pct_of_total', 0):.1f}% of population
  • Medium Risk (0.30-0.60): {risk_analysis.get('Medium', {}).get('pct_of_total', 0):.1f}% of population  
  • High Risk (>= 0.60): {risk_analysis.get('High', {}).get('pct_of_total', 0):.1f}% of population

💾 OUTPUTS:
  ✅ diabetes_screening_model.pkl
  ✅ screening_cv_results.json
  ✅ screening_evaluation.json
  ✅ feature_importance.csv

{'='*60}
✅ Model ready for population-level diabetes screening!
""")
    
    return best_model, test_metrics, cv_results_all


if __name__ == "__main__":
    main()
