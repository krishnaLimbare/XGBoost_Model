
import pandas as pd
import numpy as np

class RiskExplainer:
    """
    Model-Agnostic Explainer using Feature Perturbation.
    Calculates impact by observing change in prediction when features are neutralized to baseline.
    """
    
    def __init__(self, model, feature_names, feature_means):
        """
        Initialize the explainer.
        
        Parameters:
        -----------
        model : XGBClassifier or model with predict_proba
            Trained model
        feature_names : list
            List of feature names expected by the model
        feature_means : pd.Series
            Mean/Baseline values of features to use for neutralization
        """
        self.model = model
        self.feature_names = feature_names
        self.means = feature_means
        
        # Define groupings for engineered features
        # Detailed breakdown including new research features
        # Groups must reference feature names the model was ACTUALLY trained on.
        # Names here are matched against model feature_names; a name that no
        # longer exists silently contributes nothing, so keep this in sync with
        # engineer_features() in diabetes_screening_model.py.
        self.groups = {
            'Weight & Body Comp': ['bmi', 'Obesity_Flag', 'BMI_Category'],
            'Metabolic Health': ['hypertension', 'heart_disease', 'Cardio_Risk_Score'],
            'Lifestyle (Activity)': ['PhysicalActivity', 'Activity_Score'],
            'Lifestyle (Diet)': ['Diet', 'Diet_Score'],
            'Age': ['age', 'Age_Band', 'Age_Risk_Flag', 'Age_BMI_Interaction'],
            'Combined Lifestyle': ['Lifestyle_Risk_Score'],
            'Gender': ['gender']
        }

        # Fail loudly if a group references a feature the model does not have,
        # which would mean this file has drifted from the trained pipeline.
        known = set(feature_names)
        for group_name, feats in self.groups.items():
            if not any(any(f in col for col in known) for f in feats):
                raise ValueError(
                    f"Explainer group '{group_name}' matches no model feature. "
                    f"explainability.py has drifted from the trained pipeline."
                )

    def explain(self, input_row_encoded):
        """
        Calculate contribution of each group by perturbing its values to the mean.
        
        Contribution = Prob(Original) - Prob(Neutralized_Group)
        Positive val = Factor Increased Risk
        Negative val = Factor Decreased Risk (Protection)
        """
        # Base Prediction
        if isinstance(input_row_encoded, pd.Series):
             input_row_encoded = input_row_encoded.to_frame().T
             
        base_prob = self.model.predict_proba(input_row_encoded)[0][1]
        
        contributions = {}
        
        # Iterative Perturbation
        for group_name, features in self.groups.items():
            # Create a copy to perturb
            perturbed_row = input_row_encoded.copy()
            
            affected = False
            for col in self.feature_names:
                # Check if this column belongs to the group
                # (Simple substring match or exact match from our list)
                is_in_group = False
                for f in features:
                    if f in col: # flexible match for encoded cols like 'gender_Male' or 'Diet_Unhealthy'
                         is_in_group = True
                         break
                
                if is_in_group:
                    # Neutralize: replace with population mean
                    if col in self.means:
                        perturbed_row[col] = self.means[col]
                        affected = True
            
            if affected:
                # Predict with neutralized group
                new_prob = self.model.predict_proba(perturbed_row)[0][1]
                # Impact is the difference
                impact = base_prob - new_prob
                contributions[group_name] = impact
            else:
                contributions[group_name] = 0.0

        # Format Results
        results = []
        
        # Create a total impact for relative scaling (sum of abs impacts)
        total_impact = sum(abs(v) for v in contributions.values())
        if total_impact == 0: total_impact = 1.0
        
        for group, value in contributions.items():
            # Filter negligible
            if abs(value) < 0.001: 
                continue
                
            direction = "Increased Risk" if value > 0 else "Reduced Risk"
            # Normalize to 0-100% relative strength
            # Use abs(value) / sum(abs) to show "Pie Chart" style share of influence?
            # Or raw probability delta? User asked "contribution in percentage".
            # Let's interpret "percentage" as "Contribution to the Prediction Delta".
            
            # Simple scaling 0-100 for bar width
            strength = (abs(value) / total_impact) * 100
            
            results.append({
                'factor': group,
                'impact': value, # Raw probability mass contributed
                'direction': direction,
                'strength': strength, # For UI Bar
                'sign': 1 if value > 0 else -1
            })
            
        # Sort by magnitude
        results.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return results
