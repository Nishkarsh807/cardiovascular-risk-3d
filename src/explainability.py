"""
Explainability Module using SHAP (SHapley Additive exPlanations).
Provides transparent, interpretable local and global clinical risk attributions
for Overall CAD and vessel-specific stenosis (LAD, LCX, RCA).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap

# Support both direct execution and package imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.preprocessing import Preprocessor, PHYSIOLOGICAL_CATEGORIES
except ImportError:
    from preprocessing import Preprocessor, PHYSIOLOGICAL_CATEGORIES

# Set alias in main for pickle compatibility
sys.modules['__main__'].Preprocessor = Preprocessor

TARGETS = ["Cath", "LAD", "LCX", "RCA"]

class ExplainerPipeline:
    def __init__(self, models_dir="models", data_dir="data/processed"):
        self.models_dir = models_dir
        self.models = {}
        self.explainers = {}
        self.preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
        
        # Load background data for TreeExplainer
        X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        self.background_sample = X_train.sample(min(100, len(X_train)), random_state=42)
        self.feature_names = list(X_train.columns)

        self._init_explainers()

    def _init_explainers(self):
        """Initialize optimal SHAP explainers for sub-second real-time inference."""
        for target in TARGETS:
            model_path = os.path.join(self.models_dir, f"{target.lower()}_model.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                self.models[target] = model
                
                try:
                    if hasattr(model, "coef_"):
                        # Linear / Logistic Regression
                        explainer = shap.LinearExplainer(model, self.background_sample)
                    else:
                        # Tree-based model (XGBoost, RandomForest, GradientBoosting)
                        explainer = shap.TreeExplainer(model)
                except Exception:
                    explainer = shap.TreeExplainer(model, data=self.background_sample.iloc[:15])
                self.explainers[target] = explainer

    def explain_patient(self, processed_patient_row, raw_patient_dict=None, top_k=8):
        """
        Generate comprehensive SHAP explanations for a single patient across all targets.
        
        Args:
            processed_patient_row: 1-row DataFrame of scaled/processed features
            raw_patient_dict: Optional dict of raw unscaled values for clinical readability
            top_k: Number of top driving features to extract
        """
        explanations = {}

        for target in TARGETS:
            if target not in self.explainers:
                continue

            explainer = self.explainers[target]
            model = self.models[target]

            # Compute predicted probability
            prob = float(model.predict_proba(processed_patient_row)[0, 1])

            # Compute SHAP values
            shap_result = explainer(processed_patient_row)
            
            # TreeExplainer on binary classification can return 2D or 3D values
            raw_vals = shap_result.values[0]
            if len(raw_vals.shape) == 2 and raw_vals.shape[1] == 2:
                # Shape (features, 2) -> take positive class
                shap_vals = raw_vals[:, 1]
            elif len(raw_vals.shape) == 1:
                shap_vals = raw_vals
            else:
                shap_vals = raw_vals.flatten()

            # Align features with SHAP values
            feature_impacts = []
            for i, feat in enumerate(self.feature_names):
                val = float(shap_vals[i])
                raw_val = raw_patient_dict.get(feat, "N/A") if raw_patient_dict else float(processed_patient_row[feat].iloc[0])
                feature_impacts.append({
                    "feature": feat,
                    "shap_value": round(val, 4),
                    "abs_impact": abs(val),
                    "patient_value": raw_val,
                    "direction": "Risk Increasing" if val > 0 else "Risk Decreasing"
                })

            # Sort by absolute magnitude
            sorted_impacts = sorted(feature_impacts, key=lambda x: x["abs_impact"], reverse=True)

            # Physiological category grouping
            category_breakdown = {}
            for cat, feats in PHYSIOLOGICAL_CATEGORIES.items():
                cat_shap_sum = sum(
                    item["shap_value"] for item in feature_impacts
                    if any(item["feature"].startswith(f) for f in feats)
                )
                category_breakdown[cat] = round(float(cat_shap_sum), 4)

            # Generate natural language clinical summary
            top_risks = [item for item in sorted_impacts if item["shap_value"] > 0][:3]
            top_protective = [item for item in sorted_impacts if item["shap_value"] < 0][:2]

            risk_desc = ", ".join([f"{r['feature']} ({r['patient_value']})" for r in top_risks]) if top_risks else "None prominent"
            protective_desc = ", ".join([f"{p['feature']} ({p['patient_value']})" for p in top_protective]) if top_protective else "None prominent"

            summary_text = (
                f"Predicted {target} risk: {prob*100:.1f}%. "
                f"Primary risk drivers: {risk_desc}. "
                f"Protective factors: {protective_desc}."
            )

            explanations[target] = {
                "probability": round(prob, 4),
                "risk_tier": "Severe" if prob >= 0.70 else "High" if prob >= 0.50 else "Moderate" if prob >= 0.30 else "Low",
                "top_features": sorted_impacts[:top_k],
                "category_contributions": category_breakdown,
                "clinical_summary": summary_text
            }

        return explanations

if __name__ == "__main__":
    explainer = ExplainerPipeline()
    print("Explainers ready for targets:", list(explainer.explainers.keys()))
