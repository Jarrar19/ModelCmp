"""
app.py — FastAPI backend supporting Dual-Dataset XAI Comparison:
1. 'heart': UCI Cleveland Heart Disease (Small Clinical Benchmark · 297 records)
2. 'adult': Adult Census Income (Complex Large-Scale Benchmark · 32,561 records)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"
FRONTEND_DIR = HERE.parent / "frontend"

app = FastAPI(title="Dual-Dataset XAI Benchmark: Interpretable vs. Black-Box")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Metadata Definitions
# ---------------------------------------------------------------------
HEART_METADATA = {
    "age": {"label": "Age", "unit": "years", "type": "number", "min": 25, "max": 85, "step": 1, "default": 55, "category": "Demographics", "description": "Patient age in years."},
    "sex": {"label": "Sex", "unit": "", "type": "select", "category": "Demographics", "default": 1.0, "options": [{"value": 1.0, "text": "Male"}, {"value": 0.0, "text": "Female"}], "description": "Biological sex."},
    "cp": {"label": "Chest Pain Type", "unit": "", "type": "select", "category": "Clinical Symptoms", "default": 2.0, "options": [{"value": 1.0, "text": "1: Typical Angina"}, {"value": 2.0, "text": "2: Atypical Angina"}, {"value": 3.0, "text": "3: Non-Anginal Pain"}, {"value": 4.0, "text": "4: Asymptomatic (Silent Ischemia)"}], "description": "Nature of chest pain."},
    "trestbps": {"label": "Resting Blood Pressure", "unit": "mm Hg", "type": "number", "min": 80, "max": 210, "step": 1, "default": 130, "category": "Vitals & Labs", "description": "Blood pressure on admission."},
    "chol": {"label": "Serum Cholesterol", "unit": "mg/dl", "type": "number", "min": 100, "max": 580, "step": 1, "default": 240, "category": "Vitals & Labs", "description": "Total serum cholesterol."},
    "fbs": {"label": "Fasting Blood Sugar > 120", "unit": "", "type": "select", "category": "Vitals & Labs", "default": 0.0, "options": [{"value": 0.0, "text": "No (<= 120 mg/dl)"}, {"value": 1.0, "text": "Yes (> 120 mg/dl - Diabetic indicator)"}], "description": "Blood sugar status."},
    "restecg": {"label": "Resting ECG Results", "unit": "", "type": "select", "category": "Cardiac Tests", "default": 0.0, "options": [{"value": 0.0, "text": "0: Normal"}, {"value": 1.0, "text": "1: ST-T Wave Abnormality"}, {"value": 2.0, "text": "2: Left Ventricular Hypertrophy"}], "description": "Resting ECG reading."},
    "thalach": {"label": "Max Heart Rate Achieved", "unit": "bpm", "type": "number", "min": 60, "max": 220, "step": 1, "default": 150, "category": "Stress Test", "description": "Peak heart rate during exertion."},
    "exang": {"label": "Exercise-Induced Angina", "unit": "", "type": "select", "category": "Stress Test", "default": 0.0, "options": [{"value": 0.0, "text": "No"}, {"value": 1.0, "text": "Yes (Chest pain triggered by exercise)"}], "description": "Angina during stress test."},
    "oldpeak": {"label": "ST Depression (Exercise)", "unit": "mm", "type": "number", "min": 0.0, "max": 6.5, "step": 0.1, "default": 1.0, "category": "Stress Test", "description": "ECG ST segment depression."},
    "slope": {"label": "Slope of Peak ST Segment", "unit": "", "type": "select", "category": "Stress Test", "default": 1.0, "options": [{"value": 1.0, "text": "1: Upsloping (Normal)"}, {"value": 2.0, "text": "2: Flat (Ischemic sign)"}, {"value": 3.0, "text": "3: Downsloping (Severe ischemia)"}], "description": "Morphology of ST slope."},
    "ca": {"label": "Major Vessels Colored", "unit": "vessels", "type": "select", "category": "Fluoroscopy & Scans", "default": 0.0, "options": [{"value": 0.0, "text": "0: None (Clear vessels)"}, {"value": 1.0, "text": "1: One major vessel narrowed"}, {"value": 2.0, "text": "2: Two major vessels narrowed"}, {"value": 3.0, "text": "3: Three major vessels narrowed"}], "description": "Narrowed coronary vessels by fluoroscopy."},
    "thal": {"label": "Thallium Stress Test Result", "unit": "", "type": "select", "category": "Fluoroscopy & Scans", "default": 3.0, "options": [{"value": 3.0, "text": "3: Normal Blood Flow"}, {"value": 6.0, "text": "6: Fixed Defect (Previous Infarct/Scar)"}, {"value": 7.0, "text": "7: Reversible Defect (Active Ischemia)"}], "description": "Myocardial perfusion scan."}
}

ADULT_METADATA = {
    "Age": {"label": "Age", "unit": "years", "type": "number", "min": 17, "max": 90, "step": 1, "default": 38, "category": "Demographics", "description": "Age in years."},
    "Workclass": {"label": "Workclass", "unit": "", "type": "select", "category": "Employment", "default": 4.0, "options": [{"value": 4.0, "text": "Private Industry"}, {"value": 5.0, "text": "Self-Employed (Incorporated)"}, {"value": 6.0, "text": "Self-Employed (Non-Inc)"}, {"value": 1.0, "text": "Federal Government"}, {"value": 7.0, "text": "State Government"}, {"value": 2.0, "text": "Local Government"}, {"value": 8.0, "text": "Without Pay / Volunteer"}], "description": "Employment sector."},
    "Education-Num": {"label": "Education Level", "unit": "", "type": "select", "category": "Demographics", "default": 13.0, "options": [{"value": 13.0, "text": "Bachelors Degree (13)"}, {"value": 14.0, "text": "Masters Degree (14)"}, {"value": 16.0, "text": "Doctorate (16)"}, {"value": 15.0, "text": "Professional School (15)"}, {"value": 10.0, "text": "Some College (10)"}, {"value": 9.0, "text": "High School Grad (9)"}, {"value": 11.0, "text": "Assoc-Vocational (11)"}, {"value": 12.0, "text": "Assoc-Academic (12)"}, {"value": 4.0, "text": "7th-8th Grade (4)"}], "description": "Completed level of education."},
    "Marital Status": {"label": "Marital Status", "unit": "", "type": "select", "category": "Demographics", "default": 2.0, "options": [{"value": 2.0, "text": "Married (Civilian Spouse)"}, {"value": 4.0, "text": "Never Married"}, {"value": 0.0, "text": "Divorced"}, {"value": 5.0, "text": "Separated"}, {"value": 6.0, "text": "Widowed"}, {"value": 1.0, "text": "Married (Armed Forces Spouse)"}], "description": "Current marital status."},
    "Occupation": {"label": "Occupation", "unit": "", "type": "select", "category": "Employment", "default": 4.0, "options": [{"value": 4.0, "text": "Executive / Managerial"}, {"value": 10.0, "text": "Professional Specialty"}, {"value": 13.0, "text": "Tech Support"}, {"value": 12.0, "text": "Sales"}, {"value": 3.0, "text": "Craft & Repair"}, {"value": 1.0, "text": "Administrative / Clerical"}, {"value": 14.0, "text": "Transport & Moving"}, {"value": 7.0, "text": "Machine Operation & Inspect"}, {"value": 11.0, "text": "Protective Service"}, {"value": 8.0, "text": "Other Service"}, {"value": 5.0, "text": "Farming & Fishing"}], "description": "Type of occupation."},
    "Relationship": {"label": "Household Role", "unit": "", "type": "select", "category": "Demographics", "default": 4.0, "options": [{"value": 4.0, "text": "Husband"}, {"value": 5.0, "text": "Wife"}, {"value": 0.0, "text": "Not in Family"}, {"value": 1.0, "text": "Unmarried"}, {"value": 3.0, "text": "Own Child"}, {"value": 2.0, "text": "Other Relative"}], "description": "Role in household."},
    "Race": {"label": "Race", "unit": "", "type": "select", "category": "Demographics", "default": 4.0, "options": [{"value": 4.0, "text": "White"}, {"value": 2.0, "text": "Black"}, {"value": 1.0, "text": "Asian / Pacific Islander"}, {"value": 0.0, "text": "Amer-Indian / Eskimo"}, {"value": 3.0, "text": "Other"}], "description": "Reported race."},
    "Sex": {"label": "Sex", "unit": "", "type": "select", "category": "Demographics", "default": 1.0, "options": [{"value": 1.0, "text": "Male"}, {"value": 0.0, "text": "Female"}], "description": "Biological sex."},
    "Capital Gain": {"label": "Capital Gains", "unit": "$", "type": "number", "min": 0, "max": 99999, "step": 500, "default": 0, "category": "Financials", "description": "Annual capital gains reported."},
    "Capital Loss": {"label": "Capital Losses", "unit": "$", "type": "number", "min": 0, "max": 4356, "step": 100, "default": 0, "category": "Financials", "description": "Annual capital losses reported."},
    "Hours per week": {"label": "Hours Worked Per Week", "unit": "hrs", "type": "number", "min": 1, "max": 99, "step": 1, "default": 40, "category": "Employment", "description": "Typical weekly working hours."},
    "Country": {"label": "Country of Origin", "unit": "", "type": "select", "category": "Demographics", "default": 39.0, "options": [{"value": 39.0, "text": "United States"}, {"value": 0.0, "text": "Other Countries"}], "description": "Country of origin."}
}

DATASET_CONFIGS = {
    "heart": {
        "id": "heart",
        "title": "Cleveland Heart Disease",
        "badge": "Clinical Benchmark",
        "scale": "297 cases · 13 features",
        "thesis": "Small Clinical Cohort: Interpretable model matches or beats black-box (AUC 0.9498 vs 0.9263).",
        "labels": {0: "Healthy (No Disease)", 1: "Heart Disease Detected"},
        "positive_name": "Heart Disease",
        "negative_name": "Healthy",
        "pos_direction_label": "Increases Heart Disease Risk (Risk Factor)",
        "neg_direction_label": "Protective / Pushes Toward Healthy",
        "roc_image": "roc_curve_heart.png",
        "metadata": HEART_METADATA,
        "feature_file": "heart_feature_names.json",
        "lr_file": "heart_interpretable.pkl",
        "xgb_file": "heart_blackbox.pkl",
        "scaler_file": "heart_scaler.pkl",
    },
    "adult": {
        "id": "adult",
        "title": "Adult Census Income",
        "badge": "Complex Large-Scale Benchmark",
        "scale": "32,561 records · 12 features",
        "thesis": "Large-Scale Data: XGBoost pulls ahead by +3.06% accuracy & +0.039 AUC via non-linear interactions.",
        "labels": {0: "Income <= $50K", 1: "High Income (> $50K)"},
        "positive_name": "High Income (> $50K)",
        "negative_name": "Income <= $50K",
        "pos_direction_label": "Drives toward High Income (> $50K)",
        "neg_direction_label": "Pushes toward Lower Income (<= $50K)",
        "roc_image": "roc_curve_adult.png",
        "metadata": ADULT_METADATA,
        "feature_file": "adult_feature_names.json",
        "lr_file": "adult_interpretable.pkl",
        "xgb_file": "adult_blackbox.pkl",
        "scaler_file": "adult_scaler.pkl",
    }
}

# ---------------------------------------------------------------------
# Load Models for Both Datasets at Startup
# ---------------------------------------------------------------------
MODELS = {}
for d_key, cfg in DATASET_CONFIGS.items():
    lr_m = joblib.load(MODELS_DIR / cfg["lr_file"])
    xgb_m = joblib.load(MODELS_DIR / cfg["xgb_file"])
    sc = joblib.load(MODELS_DIR / cfg["scaler_file"])
    with open(MODELS_DIR / cfg["feature_file"]) as f:
        feats = json.load(f)
    expl = shap.TreeExplainer(xgb_m)
    MODELS[d_key] = {
        "lr": lr_m,
        "xgb": xgb_m,
        "scaler": sc,
        "features": feats,
        "explainer": expl
    }

with open(HERE / "metrics.json") as f:
    METRICS = json.load(f)


class PatientInput(BaseModel):
    features: Dict[str, float] = Field(..., description="Map of feature names to float values")


def _get_dataset(dataset_name: str) -> str:
    return "adult" if dataset_name.lower() == "adult" else "heart"


def _to_ordered_df(features: Dict[str, float], feature_names: List[str]) -> pd.DataFrame:
    missing = [f for f in feature_names if f not in features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
    return pd.DataFrame([[float(features[f]) for f in feature_names]], columns=feature_names)


@app.get("/api/datasets")
def get_datasets():
    return [
        {
            "id": cfg["id"],
            "title": cfg["title"],
            "badge": cfg["badge"],
            "scale": cfg["scale"],
            "thesis": cfg["thesis"],
            "roc_image": cfg["roc_image"],
        }
        for cfg in DATASET_CONFIGS.values()
    ]


@app.get("/api/features")
def get_features(dataset: str = Query("heart")):
    d_key = _get_dataset(dataset)
    cfg = DATASET_CONFIGS[d_key]
    return {
        "dataset": d_key,
        "title": cfg["title"],
        "scale": cfg["scale"],
        "thesis": cfg["thesis"],
        "features": MODELS[d_key]["features"],
        "metadata": cfg["metadata"],
        "roc_image": cfg["roc_image"],
        "labels": cfg["labels"],
        "pos_direction_label": cfg["pos_direction_label"],
        "neg_direction_label": cfg["neg_direction_label"],
    }


@app.get("/api/metrics")
def get_metrics(dataset: str = Query("heart")):
    d_key = _get_dataset(dataset)
    return {
        "dataset": d_key,
        "current": METRICS.get(d_key, METRICS.get("heart")),
        "comparison": {
            "heart": METRICS.get("heart"),
            "adult": METRICS.get("adult"),
        }
    }


@app.post("/api/predict")
def predict(payload: PatientInput, dataset: str = Query("heart")):
    d_key = _get_dataset(dataset)
    cfg = DATASET_CONFIGS[d_key]
    m_pack = MODELS[d_key]

    x_df = _to_ordered_df(payload.features, m_pack["features"])
    x_scaled = m_pack["scaler"].transform(x_df)

    # Interpretable model
    lr_pred = int(m_pack["lr"].predict(x_scaled)[0])
    lr_proba_pos = float(m_pack["lr"].predict_proba(x_scaled)[0][1])
    lr_confidence = lr_proba_pos if lr_pred == 1 else (1.0 - lr_proba_pos)

    # Black-box model
    xgb_pred = int(m_pack["xgb"].predict(x_df)[0])
    xgb_proba_pos = float(m_pack["xgb"].predict_proba(x_df)[0][1])
    xgb_confidence = xgb_proba_pos if xgb_pred == 1 else (1.0 - xgb_proba_pos)

    agree = (lr_pred == xgb_pred)

    return {
        "dataset": d_key,
        "interpretable": {
            "model": "Logistic Regression",
            "type": "Glass-Box (Inherently Interpretable)",
            "prediction": cfg["labels"][lr_pred],
            "prediction_code": lr_pred,
            "probability": round(lr_proba_pos, 4),
            "confidence": round(lr_confidence, 4),
        },
        "blackbox": {
            "model": "XGBoost",
            "type": "Black-Box (Post-Hoc Tree Ensemble)",
            "prediction": cfg["labels"][xgb_pred],
            "prediction_code": xgb_pred,
            "probability": round(xgb_proba_pos, 4),
            "confidence": round(xgb_confidence, 4),
        },
        "agree": agree,
    }


@app.post("/api/explain")
def explain(payload: PatientInput, dataset: str = Query("heart")):
    d_key = _get_dataset(dataset)
    cfg = DATASET_CONFIGS[d_key]
    m_pack = MODELS[d_key]
    feature_names = m_pack["features"]

    x_df = _to_ordered_df(payload.features, feature_names)
    x_scaled = m_pack["scaler"].transform(x_df)

    # Black-box SHAP values
    shap_vals = m_pack["explainer"].shap_values(x_df)
    sv = shap_vals[0] if not isinstance(shap_vals, list) else shap_vals[0]

    blackbox_contribs = []
    for f, v in zip(feature_names, sv):
        meta = cfg["metadata"].get(f, {})
        val = payload.features[f]
        blackbox_contribs.append({
            "feature": f,
            "label": meta.get("label", f),
            "unit": meta.get("unit", ""),
            "patient_value": val,
            "contribution": round(float(v), 4),
            "direction": "risk" if v > 0 else "protective"
        })
    blackbox_contribs.sort(key=lambda d: abs(d["contribution"]), reverse=True)

    # Interpretable: coefficient x scaled value
    coefs = m_pack["lr"].coef_[0]
    linear_contribs = coefs * x_scaled[0]

    interpretable_contribs = []
    for f, v in zip(feature_names, linear_contribs):
        meta = cfg["metadata"].get(f, {})
        val = payload.features[f]
        interpretable_contribs.append({
            "feature": f,
            "label": meta.get("label", f),
            "unit": meta.get("unit", ""),
            "patient_value": val,
            "contribution": round(float(v), 4),
            "direction": "risk" if v > 0 else "protective"
        })
    interpretable_contribs.sort(key=lambda d: abs(d["contribution"]), reverse=True)

    # Synthesis insight
    lr_top = interpretable_contribs[0]["label"]
    xgb_top = blackbox_contribs[0]["label"]

    insight = (
        f"Logistic Regression's linear decision was most strongly influenced by '{lr_top}', "
        f"while XGBoost's non-linear tree attribution was primarily driven by '{xgb_top}'."
    )

    return {
        "dataset": d_key,
        "blackbox_top": blackbox_contribs,
        "interpretable_top": interpretable_contribs,
        "clinical_insight": insight,
    }


# ---------------------------------------------------------------------
# Serve static frontend
# ---------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/{filename}")
def serve_frontend_file(filename: str):
    candidate = FRONTEND_DIR / filename
    if candidate.exists() and candidate.is_file():
        return FileResponse(str(candidate))
    raise HTTPException(status_code=404, detail="Not found")
