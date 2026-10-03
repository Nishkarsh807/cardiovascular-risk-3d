"""
Data Loader Module for Z-Alizadeh Sani Extended Dataset.
Downloads from official UCI Machine Learning Repository (Dataset ID 411)
and prepares the raw data for preprocessing.
"""

import os
import io
import requests
import pandas as pd
import openpyxl

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00411/extention%20of%20Z-Alizadeh%20sani%20dataset.xlsx"

def download_dataset(output_dir="data/raw"):
    """Download the dataset from UCI and save as both excel and csv."""
    os.makedirs(output_dir, exist_ok=True)
    xlsx_path = os.path.join(output_dir, "z_alizadeh_sani_extended.xlsx")
    csv_path = os.path.join(output_dir, "z_alizadeh_sani_extended.csv")

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        print(f"[DataLoader] Dataset already exists at {csv_path}")
        return pd.read_csv(csv_path)

    print(f"[DataLoader] Fetching dataset from UCI: {UCI_URL} ...")
    response = requests.get(UCI_URL, timeout=30)
    response.raise_for_status()

    with open(xlsx_path, "wb") as f:
        f.write(response.content)
    print(f"[DataLoader] Saved Excel to {xlsx_path} ({len(response.content):,} bytes)")

    df = pd.read_excel(io.BytesIO(response.content))
    # Strip any whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"[DataLoader] Saved CSV to {csv_path} with shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = download_dataset()
    print("Columns loaded:", len(df.columns))
    print("Target value counts:")
    for col in ["Cath", "LAD", "LCX", "RCA"]:
        if col in df.columns:
            print(f"--- {col} ---")
            print(df[col].value_counts())
