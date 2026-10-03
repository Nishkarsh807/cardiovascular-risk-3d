"""
Comprehensive Pipeline and API Test Suite for CardioVision 3D.
Verifies data integrity, target leakage prevention, model inference,
SHAP explainability, and FastAPI endpoint functionality.
"""

import os
import sys
import json

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import Preprocessor, TARGET_COLS
from src.explainability import ExplainerPipeline
from src.api import (
    app, startup_event, health_check, get_presets, get_metrics, predict_risk, PatientInput
)

def test_data_integrity():
    raw_path = os.path.join(BASE_DIR, "data", "raw", "z_alizadeh_sani_extended.csv")
    assert os.path.exists(raw_path), "Raw dataset file must exist."
    import pandas as pd
    df = pd.read_csv(raw_path)
    assert df.shape[0] == 303, f"Expected 303 records, got {df.shape[0]}"
    for target in ["Cath", "LAD", "LCX", "RCA"]:
        assert target in df.columns, f"Target {target} must be in raw columns"

def test_target_leakage_prevention():
    preprocessor = Preprocessor.load(os.path.join(BASE_DIR, "models", "preprocessor.pkl"))
    assert preprocessor.fitted, "Preprocessor must be fitted."
    
    # Verify no target columns in feature names
    for target in TARGET_COLS:
        assert target not in preprocessor.feature_names, f"Target {target} leaked into feature matrix!"

    # Test single transform with targets passed
    mock_patient = {
        "Age": 65, "Sex": "Male", "BP": 140, "PR": 80, "Cath": "CAD",
        "LAD": "Stenotic", "LCX": "Normal", "RCA": "Normal"
    }
    transformed = preprocessor.transform_single(mock_patient)
    assert transformed.shape[1] == len(preprocessor.feature_names)
    for target in TARGET_COLS:
        assert target not in transformed.columns

def test_model_inference_and_shap():
    explainer = ExplainerPipeline(
        models_dir=os.path.join(BASE_DIR, "models"),
        data_dir=os.path.join(BASE_DIR, "data", "processed")
    )
    assert len(explainer.models) == 4, "Must have 4 trained models."
    assert len(explainer.explainers) == 4, "Must have 4 SHAP explainers."

    # Load a preset patient
    presets_file = os.path.join(BASE_DIR, "data", "patient_presets.json")
    with open(presets_file) as f:
        presets = json.load(f)
    sample_patient = presets[0]["data"]

    processed = explainer.preprocessor.transform_single(sample_patient)
    explanations = explainer.explain_patient(processed, raw_patient_dict=sample_patient)

    for target in TARGET_COLS:
        assert target in explanations, f"{target} explanation missing"
        assert "probability" in explanations[target]
        assert 0.0 <= explanations[target]["probability"] <= 1.0
        assert "risk_tier" in explanations[target]
        assert len(explanations[target]["top_features"]) > 0

def test_api_endpoints():
    startup_event()

    # Health
    r_health = health_check()
    assert r_health["status"] == "healthy"
    assert r_health["pipeline_ready"] is True

    # Presets
    r_presets = get_presets()
    presets = r_presets["presets"]
    assert len(presets) >= 4

    # Metrics
    r_metrics = get_metrics()
    metrics = r_metrics["metrics"]
    for t in TARGET_COLS:
        assert t in metrics
        assert "test_evaluation" in metrics[t]
        assert metrics[t]["test_evaluation"]["roc_auc"] > 0.60

    # Predict
    payload = PatientInput(data=presets[0]["data"])
    res = predict_risk(payload)
    assert res["success"] is True
    assert "overall_cad" in res
    assert "vessels" in res
    assert "LAD" in res["vessels"]
    assert "LCX" in res["vessels"]
    assert "RCA" in res["vessels"]
    assert "clinical_disclaimer" in res
    print("\n[Tests] All CardioVision 3D pipeline and API tests passed successfully!")

if __name__ == "__main__":
    test_data_integrity()
    test_target_leakage_prevention()
    test_model_inference_and_shap()
    test_api_endpoints()
