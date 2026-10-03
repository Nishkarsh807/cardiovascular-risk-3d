"""
Preprocessing Pipeline for Cardiovascular Risk & Stenosis Prediction.
Strictly separates targets (Cath, LAD, LCX, RCA) to prevent target leakage.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure src namespace exists in sys.modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

TARGET_COLS = ["Cath", "LAD", "LCX", "RCA"]

BINARY_COLS_YN = [
    "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease",
    "CHF", "DLP", "Weak Peripheral Pulse", "Lung rales",
    "Systolic Murmur", "Diastolic Murmur", "Dyspnea", "Atypical",
    "Nonanginal", "Exertional CP", "LowTH Ang", "LVH", "Poor R Progression"
]

CATEGORICAL_MULTI = {
    "BBB": ["N", "LBBB", "RBBB"],
    "VHD": ["N", "mild", "Moderate", "Severe"]
}

PHYSIOLOGICAL_CATEGORIES = {
    "Demographics": ["Age", "Sex", "Weight", "Length", "BMI", "Obesity"],
    "Risk Factors & History": ["DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP"],
    "Symptoms & Vitals": ["BP", "PR", "Edema", "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur", "Typical Chest Pain", "Dyspnea", "Function Class", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang"],
    "ECG Findings": ["Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression", "BBB_LBBB", "BBB_RBBB"],
    "Laboratory Biomarkers": ["FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT"],
    "Echocardiography": ["EF-TTE", "Region RWMA", "VHD_mild", "VHD_Moderate", "VHD_Severe"]
}

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.numeric_cols = []
        self.binary_cols = []
        self.fitted = False

    def _clean_raw_df(self, df):
        """Clean string representations and basic column formats."""
        df = df.copy()
        
        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]

        # Sex: Male -> 1, Fmale/Female -> 0
        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].astype(str).str.strip().apply(
                lambda x: 1 if "m" in x.lower() and "f" not in x.lower() else 0
            )

        # Binary Y/N columns
        for col in BINARY_COLS_YN:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper().apply(
                    lambda x: 1 if x == "Y" else 0
                )

        # Categorical: BBB
        if "BBB" in df.columns:
            df["BBB"] = df["BBB"].astype(str).str.strip()
            df["BBB_LBBB"] = (df["BBB"] == "LBBB").astype(int)
            df["BBB_RBBB"] = (df["BBB"] == "RBBB").astype(int)
            df = df.drop(columns=["BBB"])

        # Categorical: VHD
        if "VHD" in df.columns:
            df["VHD"] = df["VHD"].astype(str).str.strip()
            df["VHD_mild"] = (df["VHD"].str.lower() == "mild").astype(int)
            df["VHD_Moderate"] = (df["VHD"].str.lower() == "moderate").astype(int)
            df["VHD_Severe"] = (df["VHD"].str.lower() == "severe").astype(int)
            df = df.drop(columns=["VHD"])

        # Exclude constant Exertional CP if exists
        if "Exertional CP" in df.columns:
            # Keep as feature (0) or retain for consistency
            pass

        return df

    def fit_transform(self, raw_df):
        """Fit scaler on numeric features and transform dataset."""
        df = self._clean_raw_df(raw_df)

        # Separate targets
        targets = {}
        for target in TARGET_COLS:
            if target in df.columns:
                if target == "Cath":
                    targets[target] = (df[target].astype(str).str.strip() == "CAD").astype(int)
                else: # LAD, LCX, RCA
                    targets[target] = (df[target].astype(str).str.strip().str.lower() == "stenotic").astype(int)
                df = df.drop(columns=[target])

        # Input feature dataframe
        X = df.copy()

        # Identify numeric columns (continuous measurements)
        self.numeric_cols = [
            "Age", "Weight", "Length", "BMI", "BP", "PR", "Function Class",
            "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na",
            "WBC", "Lymph", "Neut", "PLT", "EF-TTE", "Region RWMA"
        ]
        self.numeric_cols = [c for c in self.numeric_cols if c in X.columns]
        
        # Binary features (0/1 indicators)
        self.binary_cols = [c for c in X.columns if c not in self.numeric_cols]
        
        # Fit scaler on numeric features
        X[self.numeric_cols] = self.scaler.fit_transform(X[self.numeric_cols])
        self.feature_names = list(X.columns)
        self.fitted = True

        y_df = pd.DataFrame(targets)
        return X, y_df

    def transform_single(self, patient_dict):
        """Transform a single patient dict (for API inference)."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted first.")
        
        df = pd.DataFrame([patient_dict])
        df = self._clean_raw_df(df)

        # Drop targets if present in incoming data to guarantee no leakage
        for target in TARGET_COLS:
            if target in df.columns:
                df = df.drop(columns=[target])

        # Ensure all expected feature columns exist
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_names]
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        return df

    def save(self, filepath="models/preprocessor.pkl"):
        """Save preprocessor state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[Preprocessor] Saved to {filepath}")

    @staticmethod
    def load(filepath="models/preprocessor.pkl"):
        """Load preprocessor state."""
        return joblib.load(filepath)

def prepare_data(data_path="data/raw/z_alizadeh_sani_extended.csv", test_size=0.2, random_state=42):
    """Load, preprocess, and split dataset with target leakage safeguards."""
    df = pd.read_csv(data_path)
    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df)

    # Save preprocessor
    preprocessor.save("models/preprocessor.pkl")

    # Stratified split based on overall CAD status (Cath)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y["Cath"]
    )

    # Save processed splits
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print(f"[PrepareData] Processed feature matrix: {X.shape[1]} features.")
    print(f"[PrepareData] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("[PrepareData] Targets:", list(y.columns))
    return preprocessor, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_data()
