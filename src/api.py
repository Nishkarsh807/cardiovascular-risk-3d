"""
FastAPI Backend Service for Cardiovascular Risk Visualization & Multi-Vessel Stenosis.
Serves interactive 3D frontend, real-time predictions, SHAP explanations, and validation benchmarks.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import Preprocessor, PHYSIOLOGICAL_CATEGORIES
from src.explainability import ExplainerPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CardioVisionAPI")

app = FastAPI(
    title="CardioVision 3D - Cardiovascular Risk & Stenosis API",
    description="Multimodal AI Hackathon 2026: 3D Anatomical Mapping & Machine Learning for Coronary Artery Disease",
    version="1.0.0"
)

# Enable CORS for open browser interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
explainer_pipeline: Optional[ExplainerPipeline] = None
metrics_cache: Dict[str, Any] = {}
presets_cache: list = []

@app.on_event("startup")
def startup_event():
    global explainer_pipeline, metrics_cache, presets_cache
    logger.info("Initializing CardioVision ML & Explainability Pipeline...")
    
    models_dir = os.path.join(BASE_DIR, "models")
    data_dir = os.path.join(BASE_DIR, "data")

    # Load metrics
    metrics_path = os.path.join(models_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_cache = json.load(f)
            logger.info("Loaded model metrics cache.")

    # Load presets
    presets_path = os.path.join(data_dir, "patient_presets.json")
    if os.path.exists(presets_path):
        with open(presets_path, "r") as f:
            presets_cache = json.load(f)
            logger.info(f"Loaded {len(presets_cache)} patient presets.")

    # Initialize Explainer Pipeline
    try:
        explainer_pipeline = ExplainerPipeline(
            models_dir=models_dir,
            data_dir=os.path.join(data_dir, "processed")
        )
        logger.info("CardioVision ML & SHAP Pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")

# Mount static files
static_dir = os.path.join(BASE_DIR, "web", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def get_index():
    index_file = os.path.join(BASE_DIR, "web", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "CardioVision 3D API is running. UI file index.html is being prepared."}

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "CardioVision 3D Backend",
        "pipeline_ready": explainer_pipeline is not None,
        "models_loaded": list(explainer_pipeline.models.keys()) if explainer_pipeline else []
    }

@app.get("/api/presets")
def get_presets():
    return {"presets": presets_cache}

@app.get("/api/metrics")
def get_metrics():
    if not metrics_cache:
        raise HTTPException(status_code=404, detail="Model metrics not found.")
    return metrics_cache

class PatientInput(BaseModel):
    data: Dict[str, Any]

@app.post("/api/predict")
def predict_risk(payload: PatientInput):
    """
    Real-time inference endpoint.
    Computes overall CAD risk, LAD, LCX, RCA stenosis probabilities,
    and returns SHAP feature attributions and anatomical annotations.
    """
    if not explainer_pipeline:
        raise HTTPException(status_code=503, detail="Model pipeline is not ready.")

    raw_data = payload.data
    try:
        # Preprocess features ensuring zero target leakage
        processed_row = explainer_pipeline.preprocessor.transform_single(raw_data)
        
        # Compute multi-target prediction and SHAP values
        explanations = explainer_pipeline.explain_patient(
            processed_patient_row=processed_row,
            raw_patient_dict=raw_data,
            top_k=8
        )

        # Vessel anatomical mapping metadata
        vessel_details = {
            "LAD": {
                "vessel_name": "Left Anterior Descending Artery",
                "anatomical_role": "Supplies the anterior myocardium, apex, and anterior 2/3 of the interventricular septum.",
                "perfusion_territory": "Anterior Left Ventricle & Septum",
                "clinical_finding": (
                    "Critical anterior wall ischemia risk ('Widowmaker')." 
                    if explanations["LAD"]["probability"] >= 0.60 
                    else "Moderate anterior perfusion compromise." 
                    if explanations["LAD"]["probability"] >= 0.35 
                    else "Normal anterior perfusion expected."
                ),
                "probability": explanations["LAD"]["probability"],
                "risk_tier": explanations["LAD"]["risk_tier"],
                "color_code": "#ff1744" if explanations["LAD"]["probability"] >= 0.60 else "#ffb300" if explanations["LAD"]["probability"] >= 0.35 else "#00e676"
            },
            "LCX": {
                "vessel_name": "Left Circumflex Artery",
                "anatomical_role": "Courses along the coronary sulcus, supplying the posterolateral wall of the left ventricle.",
                "perfusion_territory": "Lateral & Posterior Left Ventricle",
                "clinical_finding": (
                    "Significant lateral wall ischemia risk." 
                    if explanations["LCX"]["probability"] >= 0.60 
                    else "Moderate circumflex territory compromise." 
                    if explanations["LCX"]["probability"] >= 0.35 
                    else "Normal lateral perfusion expected."
                ),
                "probability": explanations["LCX"]["probability"],
                "risk_tier": explanations["LCX"]["risk_tier"],
                "color_code": "#ff1744" if explanations["LCX"]["probability"] >= 0.60 else "#ffb300" if explanations["LCX"]["probability"] >= 0.35 else "#00e676"
            },
            "RCA": {
                "vessel_name": "Right Coronary Artery",
                "anatomical_role": "Supplies right ventricle, right atrium, SA/AV nodes, and inferior diaphragmatic wall via PDA.",
                "perfusion_territory": "Right Ventricle, Inferior Wall & Conduction Nodes",
                "clinical_finding": (
                    "High risk of inferior wall ischemia and conduction disturbance." 
                    if explanations["RCA"]["probability"] >= 0.60 
                    else "Moderate inferior territory compromise." 
                    if explanations["RCA"]["probability"] >= 0.35 
                    else "Normal right/inferior perfusion expected."
                ),
                "probability": explanations["RCA"]["probability"],
                "risk_tier": explanations["RCA"]["risk_tier"],
                "color_code": "#ff1744" if explanations["RCA"]["probability"] >= 0.60 else "#ffb300" if explanations["RCA"]["probability"] >= 0.35 else "#00e676"
            }
        }

        # Multi-vessel risk summary
        cad_prob = explanations["Cath"]["probability"]
        high_risk_vessels = [v for v, info in vessel_details.items() if info["probability"] >= 0.50]
        
        if len(high_risk_vessels) >= 3:
            severity_description = "Triple Vessel Coronary Artery Disease (Severe Multivessel CAD)"
        elif len(high_risk_vessels) == 2:
            severity_description = f"Double Vessel Disease ({' + '.join(high_risk_vessels)})"
        elif len(high_risk_vessels) == 1:
            severity_description = f"Single Vessel Disease ({high_risk_vessels[0]})"
        elif cad_prob >= 0.50:
            severity_description = "Coronary Artery Disease with Diffuse / Microvascular Pattern"
        else:
            severity_description = "Low Probability of Obstructive CAD (Hemodynamically Stable)"

        return {
            "success": True,
            "overall_cad": {
                "probability": cad_prob,
                "risk_tier": explanations["Cath"]["risk_tier"],
                "severity_classification": severity_description,
                "summary": explanations["Cath"]["clinical_summary"]
            },
            "vessels": vessel_details,
            "explanations": explanations,
            "clinical_disclaimer": "FOR DECISION SUPPORT AND EDUCATIONAL DEMONSTRATION ONLY. NOT A SUBSTITUTE FOR FORMAL DIAGNOSTIC IMAGING (ANGIOGRAPHY/CTCA)."
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
