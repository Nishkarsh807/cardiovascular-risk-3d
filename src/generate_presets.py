"""
Generate Clinical Patient Presets from the Z-Alizadeh Sani dataset
to allow instantaneous 1-click clinical testing in the UI.
"""

import os
import json
import pandas as pd

def generate_presets(data_path="data/raw/z_alizadeh_sani_extended.csv", output_path="data/patient_presets.json"):
    df = pd.read_csv(data_path)
    df.columns = [str(c).strip() for c in df.columns]

    presets = []

    # 1. Triple Vessel Disease (LAD=Stenotic, LCX=Stenotic, RCA=Stenotic, Cath=CAD)
    tvd_rows = df[(df["LAD"].str.lower() == "stenotic") & 
                  (df["LCX"].str.lower() == "stenotic") & 
                  (df["RCA"].str.lower() == "stenotic") & 
                  (df["Cath"] == "CAD")]
    if len(tvd_rows) > 0:
        row = tvd_rows.iloc[0].to_dict()
        presets.append({
            "id": "preset_triple_vessel",
            "name": "Case 1: Severe Triple Vessel CAD (LAD + LCX + RCA)",
            "description": "68-year-old male with typical angina, diabetes, hypertension, and multi-vessel diffuse coronary stenosis.",
            "data": row
        })

    # 2. Isolated LAD Stenosis (LAD=Stenotic, LCX=Normal, RCA=Normal, Cath=CAD)
    lad_rows = df[(df["LAD"].str.lower() == "stenotic") & 
                  (df["LCX"].str.lower() == "normal") & 
                  (df["RCA"].str.lower() == "normal") & 
                  (df["Cath"] == "CAD")]
    if len(lad_rows) > 0:
        row = lad_rows.iloc[0].to_dict()
        presets.append({
            "id": "preset_isolated_lad",
            "name": "Case 2: Isolated Critical LAD Stenosis ('Widowmaker')",
            "description": "59-year-old patient with ST elevation, anterior wall hypokinesia, and isolated left anterior descending artery narrowing.",
            "data": row
        })

    # 3. Isolated RCA Stenosis (LAD=Normal, LCX=Normal, RCA=Stenotic, Cath=CAD)
    rca_rows = df[(df["LAD"].str.lower() == "normal") & 
                  (df["LCX"].str.lower() == "normal") & 
                  (df["RCA"].str.lower() == "stenotic") & 
                  (df["Cath"] == "CAD")]
    if len(rca_rows) > 0:
        row = rca_rows.iloc[0].to_dict()
        presets.append({
            "id": "preset_isolated_rca",
            "name": "Case 3: Isolated RCA Stenosis with Inferior RWMA",
            "description": "62-year-old patient with right coronary artery stenosis, inferior regional wall motion abnormality, and dyspnea.",
            "data": row
        })

    # 4. Healthy / Normal Control (LAD=Normal, LCX=Normal, RCA=Normal, Cath=Normal)
    norm_rows = df[(df["LAD"].str.lower() == "normal") & 
                   (df["LCX"].str.lower() == "normal") & 
                   (df["RCA"].str.lower() == "normal") & 
                   (df["Cath"] == "Normal")]
    if len(norm_rows) > 0:
        row = norm_rows.iloc[0].to_dict()
        presets.append({
            "id": "preset_normal_control",
            "name": "Case 4: Healthy Control (Normal Coronary Anatomy)",
            "description": "45-year-old asymptomatic individual with clear coronaries, normal ejection fraction (EF=55%), and no wall motion defects.",
            "data": row
        })

    # 5. Borderline / Intermediate Risk
    intermediate_rows = df[(df["Cath"] == "CAD") & 
                            (df["Age"] < 60) & 
                            (df["EF-TTE"] >= 45)]
    if len(intermediate_rows) > 0:
        row = intermediate_rows.iloc[0].to_dict()
        presets.append({
            "id": "preset_intermediate_risk",
            "name": "Case 5: Intermediate Risk / Early Coronary Atherosclerosis",
            "description": "54-year-old patient with atypical symptoms, borderline hypertension, and single moderate lesion.",
            "data": row
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(presets, f, indent=2)
    print(f"[Presets] Generated {len(presets)} clinical presets in {output_path}")

if __name__ == "__main__":
    generate_presets()
