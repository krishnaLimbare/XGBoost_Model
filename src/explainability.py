
import pandas as pd
import numpy as np

class RiskExplainer:
    """
    Explainer for Logistic Regression models using exact analytical SHAP values.
    Calculates feature contributions as: (Value - Mean) * Coefficient.
    Aggregates contributions into clinically meaningful groups.
    """
    
    def __init__(self, model, feature_names, feature_means):
        """
        Initialize the explainer.
        
        Parameters:
        -----------
        model : LogisticRegression
            Trained model
        feature_names : list
            List of feature names corresponding to model coefficients
        feature_means : pd.Series
            Mean values of features (from training set) to serve as baseline
        """
        self.feature_names = feature_names
        # Ensure coefficients match feature names
        if hasattr(model, 'coef_'):
            self.coef = pd.Series(model.coef_[0], index=feature_names)
        else:
            raise ValueError("Model must have coefficients (LogisticRegression)")
            
        self.means = feature_means
        
        # Define groupings for engineered features
        self.groups = {
            'Weight Factors': ['bmi', 'Obesity_Flag', 'BMI_Category', 'Age_BMI_Interaction'],
            'Age Factors': ['age', 'Age_Band', 'Age_Risk_Flag'],
            'Heart Health': ['hypertension', 'heart_disease', 'Cardio_Risk_Score'],
            'Lifestyle Habits': ['Diet', 'PhysicalActivity', 'Lifestyle_Risk_Score', 'Diet_Score', 'Activity_Score'],
            'Gender': ['gender']
        }

    def explain(self, input_row_encoded):
        """
        Calculate feature contributions and aggregate them.
        
        Parameters:
        -----------
        input_row_encoded : pd.DataFrame
            Single row dataframe, already encoded and scaled (matching model input)
            
        Returns:
        --------
        list : Top contributing factors with description and direction
        """
        # 1. Calculate raw contributions: (X - E[X]) * W
        # Input row might be a DataFrame, convert to Series for alignment
        row_values = input_row_encoded.iloc[0]
        
        # Align means with input columns (ensure order)
        means_aligned = self.means[row_values.index]
        coef_aligned = self.coef[row_values.index]
        
        # Calculate deviation from mean
        deviation = row_values - means_aligned
        
        # Calculate contribution (SHAP value)
        contributions = deviation * coef_aligned
        
        # 2. Aggregate into groups
        group_contributions = {k: 0.0 for k in self.groups.keys()}
        
        for feature, value in contributions.items():
            assigned = False
            for group, keywords in self.groups.items():
                if any(k in feature for k in keywords):
                    group_contributions[group] += value
                    assigned = True
                    break
            
            # If not assigned (shouldn't happen with our comprehensive groups, but safe fallback)
            if not assigned:
                # Add to a 'Other' group or ignore? 
                # For now, ignore minor unmatched features or log them
                pass
                
        # 3. Format for output
        results = []
        max_abs_val = max([abs(v) for v in group_contributions.values()]) if group_contributions else 1.0
        if max_abs_val == 0: max_abs_val = 1.0
        
        for group, value in group_contributions.items():
            # Skip negligible contributions
            if abs(value) < 0.01:
                continue
                
            direction = "Increased Risk" if value > 0 else "Reduced Risk"
            # Normalize strength 0-100 for UI bar
            strength = (abs(value) / max_abs_val) * 100
            
            results.append({
                'factor': group,
                'impact': value,
                'direction': direction,
                'strength': strength,
                'sign': 1 if value > 0 else -1
            })
            
        # Sort by absolute impact (highest first)
        results.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return results
