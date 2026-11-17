"""
app/main.py
===========
FastAPI prediction service for the child malnutrition risk model.

Endpoints
---------
GET  /health          → liveness check
POST /predict         → single-child risk prediction (all three outcomes)
POST /predict/batch   → batch prediction from JSON list

Usage (local)
-------------
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Usage (Docker)
--------------
    docker-compose up
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
         -d @example_input.json
"""

import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

MODEL_DIR = os.getenv("MODEL_DIR", "models")

FEATURE_COLS = [
    "child_age_months", "child_sex", "birth_order", "prev_birth_interval",
    "size_at_birth", "birth_weight_kg", "diarrhea_2wk", "fever_2wk",
    "breastfeeding", "maternal_edu", "maternal_bmi",
    "wealth_index", "under5_in_hh", "hh_head_sex", "hh_head_age",
    "electricity", "residence_urban", "wash_score", "county",
]

OUTCOMES = ["stunted", "wasted", "underweight"]

# ── LOAD MODELS AT STARTUP ────────────────────────────────────────────────────

pipelines = {}
for outcome in OUTCOMES:
    path = os.path.join(MODEL_DIR, f"best_model_{outcome}.pkl")
    if os.path.exists(path):
        pipelines[outcome] = joblib.load(path)
    else:
        print(f"WARNING: model not found for '{outcome}' at {path}")

# ── FASTAPI APP ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Kenya Child Malnutrition Risk API",
    description=(
        "Predicts stunting, wasting and underweight risk for children "
        "under five in Kenya using the 2022 KDHS-trained ML models."
    ),
    version="1.0.0",
)

# ── CORS — allow browser UIs (including claude.ai) to call the API ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to specific origins in production
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── REQUEST / RESPONSE SCHEMAS ────────────────────────────────────────────────


class ChildInput(BaseModel):
    child_age_months:    float = Field(..., ge=0,  le=59,  description="Child age in months (0–59)")
    child_sex:           int = Field(..., ge=1,  le=2,   description="Child sex: 1=male, 2=female")
    birth_order:         int = Field(..., ge=1,  le=20,  description="Birth order")
    prev_birth_interval: float = Field(..., ge=0,
                                       description="Preceding birth interval (months)")
    size_at_birth:       int = Field(..., ge=1,  le=5,
                                     description="Size at birth: 1=very small to 5=very large")
    birth_weight_kg:     float = Field(..., ge=0.5, le=6.0, description="Birth weight in kg")
    diarrhea_2wk:        int = Field(..., ge=0,  le=1,
                                     description="Diarrhea in last 2 weeks: 1=yes")
    fever_2wk:           int = Field(..., ge=0,  le=1,   description="Fever in last 2 weeks: 1=yes")
    breastfeeding:       int = Field(..., ge=1,  le=3,
                                     description="1=currently, 2=previously, 3=never")
    maternal_edu:        int = Field(..., ge=0,  le=3,
                                     description="0=none, 1=primary, 2=secondary, 3=higher")
    maternal_bmi:        float = Field(..., ge=10, le=60,  description="Maternal BMI (kg/m²)")
    wealth_index:        int = Field(..., ge=1,  le=5,
                                     description="DHS wealth quintile: 1=poorest to 5=richest")
    under5_in_hh:        int = Field(..., ge=1,  le=20,
                                     description="Number of children under 5 in household")
    hh_head_sex:         int = Field(..., ge=1,  le=2,   description="1=male, 2=female")
    hh_head_age:         float = Field(..., ge=15, le=100,
                                       description="Household head age in years")
    electricity:         int = Field(..., ge=0,  le=1,
                                     description="Household has electricity: 1=yes")
    residence_urban:     int = Field(..., ge=0,  le=1,
                                     description="Urban residence: 1=urban, 0=rural")
    wash_score:          int = Field(..., ge=0,  le=3,   description="WASH composite score (0–3)")
    county:              int = Field(..., ge=1,  le=47,  description="Kenya county code (1–47)")

    class Config:
        json_schema_extra = {
            "example": {
                "child_age_months": 24, "child_sex": 1, "birth_order": 2,
                "prev_birth_interval": 30, "size_at_birth": 3,
                "birth_weight_kg": 3.1, "diarrhea_2wk": 0, "fever_2wk": 1,
                "breastfeeding": 1, "maternal_edu": 1, "maternal_bmi": 22.5,
                "wealth_index": 2, "under5_in_hh": 3, "hh_head_sex": 1,
                "hh_head_age": 35, "electricity": 0, "residence_urban": 0,
                "wash_score": 1, "county": 23,
            }
        }


# ── HELPER ────────────────────────────────────────────────────────────────────

def _run_models(child: ChildInput) -> dict:
    """Run all three models and return probability + flag per outcome."""
    row = pd.DataFrame([child.model_dump()])[FEATURE_COLS]
    results = {}
    for outcome in OUTCOMES:
        if outcome not in pipelines:
            raise HTTPException(
                status_code=503,
                detail=f"Model for '{outcome}' is not loaded. Run 02_train.py first.",
            )
        pipe = pipelines[outcome]["pipeline"]
        prob = float(pipe.predict_proba(row.values)[0, 1])
        results[outcome] = {"probability": round(prob, 4), "flag": prob >= 0.5}
    return results


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness check — confirms which models are loaded."""
    return {
        "status": "ok",
        "models_loaded": list(pipelines.keys()),
        "all_models_ready": len(pipelines) == 3,
    }


@app.post("/predict", tags=["Prediction"])
def predict(child: ChildInput):
    """
    Predict stunting, wasting and underweight risk for a single child.
    Returns probability scores (0–1) and binary flags (threshold = 0.5).
    """
    preds = _run_models(child)
    return {
        "predictions": preds,
        "county": child.county,
        "model_version": "1.0.0",
    }


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(children: List[ChildInput]):
    """Predict for a list of children (max 1000 per request)."""
    if len(children) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 children per batch request.")
    return [
        {"predictions": _run_models(c), "county": c.county, "model_version": "1.0.0"}
        for c in children
    ]


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Kenya Child Malnutrition Risk API",
        "docs": "/docs",
        "health": "/health",
    }
