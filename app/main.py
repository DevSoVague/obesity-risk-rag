"""
main.py
=======
FastAPI server — Obesity Classification + Model 2 Escalation + Diabetes Risk

Endpoints:
  POST /predict          — Model 1 only (lifestyle inputs)
  POST /predict/full     — Model 1 + auto-escalation to Model 2 + diabetes risk
  GET  /health           — liveness probe
  GET  /classes          — target class list
  GET  /features         — Model 1 feature list
  GET  /input-guide      — full input schema

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_pipeline import (
    _load_bundle,
    run_inference,
    OB_ORDER,
    should_trigger_model2,
    run_model2_inference,
    assess_diabetes_risk,
)

# ── Load Model 1 bundle once at startup ──────────────────────────────────────
bundle = _load_bundle()

app = FastAPI(
    title="Obesity & Metabolic Risk API",
    description=(
        "Model 1: 7-class obesity classifier (lifestyle inputs). "
        "Model 2: OW_I vs OW_II refinement via NHANES clinical features (auto-triggered). "
        "Diabetes risk: ADA 2024 guideline-based assessment."
    ),
    version="3.0.0",
)


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class Model1Inputs(BaseModel):
    """Core lifestyle inputs — required for Model 1."""
    gender:         int   = Field(..., ge=0,  le=1,   description="0=Female  1=Male")
    age:            float = Field(..., ge=10, le=100, description="Age in years")
    family_history: int   = Field(..., ge=0,  le=1,   description="Family history of overweight: 0=No 1=Yes")
    fcvc:           float = Field(..., ge=1,  le=3,   description="Vegetable consumption: 1=Never 2=Sometimes 3=Always")
    ncp:            float = Field(..., ge=1,  le=4,   description="Main meals per day: 1–4")
    caec:           int   = Field(..., ge=0,  le=3,   description="Eating between meals: 0=No 1=Sometimes 2=Frequently 3=Always")
    ch2o:           float = Field(..., ge=1,  le=3,   description="Water intake: 1=<1L 2=1–2L 3=>2L")
    faf:            float = Field(..., ge=0,  le=3,   description="Physical activity days/week: 0–3")
    tue:            float = Field(..., ge=0,  le=2,   description="Screen time: 0=0–2h 1=3–5h 2=5h+")
    calc:           int   = Field(..., ge=0,  le=3,   description="Alcohol: 0=Never 1=Sometimes 2=Frequently 3=Always")
    bmi_bucket:     int   = Field(..., ge=0,  le=3,   description="0=Underweight(<18.5) 1=Normal(18.5–25) 2=Overweight(25–30) 3=Obese(>30)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": 1, "age": 25, "family_history": 1,
                "fcvc": 2, "ncp": 3, "caec": 1, "ch2o": 2,
                "faf": 1, "tue": 1, "calc": 1, "bmi_bucket": 2,
            }
        }
    }


class ClinicalInputs(BaseModel):
    """
    Optional clinical inputs for Model 2 escalation and diabetes risk.
    All fields optional — Model 2 handles missing values natively.
    Collecting these enables: OW_I/OW_II refinement + diabetes risk assessment.
    """
    # Body measurements
    waist_cm:      Optional[float] = Field(None, ge=40,  le=200, description="Waist circumference (cm)")
    hip_cm:        Optional[float] = Field(None, ge=40,  le=200, description="Hip circumference (cm)")
    arm_cm:        Optional[float] = Field(None, ge=10,  le=60,  description="Mid-upper arm circumference (cm)")
    # Blood pressure
    sys_bp:        Optional[float] = Field(None, ge=60,  le=250, description="Systolic blood pressure (mmHg)")
    dia_bp:        Optional[float] = Field(None, ge=30,  le=150, description="Diastolic blood pressure (mmHg)")
    # Lab values
    glucose:       Optional[float] = Field(None, ge=40,  le=700, description="Fasting plasma glucose (mg/dL)")
    insulin:       Optional[float] = Field(None, ge=0.1, le=300, description="Fasting insulin (µU/mL)")
    hba1c:         Optional[float] = Field(None, ge=3.0, le=20,  description="HbA1c (%)")
    cholesterol:   Optional[float] = Field(None, ge=50,  le=600, description="Total cholesterol (mg/dL)")
    # Exact BMI (from height/weight calculator — used for diabetes risk)
    exact_bmi:     Optional[float] = Field(None, ge=10,  le=80,  description="Exact BMI (kg/m²) from height/weight")


class FullPredictRequest(BaseModel):
    """Combined request: Model 1 lifestyle inputs + optional clinical inputs."""
    lifestyle:  Model1Inputs
    clinical:   Optional[ClinicalInputs] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "lifestyle": {
                    "gender": 1, "age": 45, "family_history": 1,
                    "fcvc": 2, "ncp": 3, "caec": 1, "ch2o": 2,
                    "faf": 1, "tue": 1, "calc": 1, "bmi_bucket": 2,
                },
                "clinical": {
                    "waist_cm": 102.0,
                    "hip_cm":   105.0,
                    "sys_bp":   135.0,
                    "dia_bp":   85.0,
                    "glucose":  118.0,
                    "hba1c":    6.1,
                }
            }
        }
    }


# Simple Model 1-only request (backwards compat)
class PredictRequest(Model1Inputs):
    pass


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status":  "ok",
        "model":   "Obesity & Metabolic Risk API v3.0",
        "model1_features": len(bundle["feature_names"]),
        "endpoints": ["/predict", "/predict/full", "/health", "/classes", "/features", "/input-guide"],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(req: PredictRequest):
    """Model 1 only — 7-class obesity classification from lifestyle inputs."""
    try:
        return run_inference(
            gender=req.gender, age=req.age,
            family_history=req.family_history,
            fcvc=req.fcvc, ncp=req.ncp, caec=req.caec,
            ch2o=req.ch2o, faf=req.faf, tue=req.tue,
            calc=req.calc, bmi_bucket=req.bmi_bucket,
            bundle=bundle,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/full")
def predict_full(req: FullPredictRequest):
    """
    Full pipeline:
      1. Model 1 — 7-class obesity classification
      2. Model 2 — OW_I/OW_II refinement (auto-triggered if Model 1 is uncertain)
      3. Diabetes risk — ADA 2024 guideline assessment (if clinical inputs provided)

    Returns a unified result dict with all three layers.
    """
    try:
        ls = req.lifestyle
        cl = req.clinical or ClinicalInputs()

        # ── Step 1: Model 1 ───────────────────────────────────────────────────
        m1 = run_inference(
            gender=ls.gender, age=ls.age,
            family_history=ls.family_history,
            fcvc=ls.fcvc, ncp=ls.ncp, caec=ls.caec,
            ch2o=ls.ch2o, faf=ls.faf, tue=ls.tue,
            calc=ls.calc, bmi_bucket=ls.bmi_bucket,
            bundle=bundle,
        )

        # ── Step 2: Model 2 escalation ────────────────────────────────────────
        m2          = None
        m2_triggered = should_trigger_model2(m1)

        if m2_triggered:
            m2 = run_model2_inference(
                gender=ls.gender, age=ls.age,
                waist_cm=cl.waist_cm,    hip_cm=cl.hip_cm,
                arm_cm=cl.arm_cm,        sys_bp=cl.sys_bp,
                dia_bp=cl.dia_bp,        glucose=cl.glucose,
                insulin=cl.insulin,      hba1c=cl.hba1c,
                cholesterol=cl.cholesterol,
            )

        # ── Step 3: Diabetes risk ─────────────────────────────────────────────
        diabetes = None
        if cl.glucose is not None or cl.hba1c is not None:
            diabetes = assess_diabetes_risk(
                glucose=cl.glucose,
                hba1c=cl.hba1c,
                insulin=cl.insulin,
                age=ls.age,
                bmi=cl.exact_bmi,
                family_history=ls.family_history,
            )

        # ── Compose final result ──────────────────────────────────────────────
        # If Model 2 ran, the final obesity classification comes from it
        final_class      = m2["predicted_class"]      if m2 else m1["predicted_class"]
        final_confidence = m2["confidence_pct"]        if m2 else m1["confidence_pct"]
        final_tier       = m2["tier"]                  if m2 else m1["tier"]

        return {
            # ── Final answer ──────────────────────────────────────────────────
            "final_classification": {
                "predicted_class": final_class,
                "confidence_pct":  final_confidence,
                "tier":            final_tier,
                "source":          "model2_nhanes" if m2 else "model1_lifestyle",
            },

            # ── Model 1 detail ────────────────────────────────────────────────
            "model1": {
                "predicted_class":  m1["predicted_class"],
                "confidence_pct":   m1["confidence_pct"],
                "tier":             m1["tier"],
                "close_call":       m1["close_call"],
                "top3":             m1["top3"],
                "all_probabilities":m1["all_probabilities"],
                "score_components": m1["score_components"],
            },

            # ── Model 2 escalation ────────────────────────────────────────────
            "model2_escalation": {
                "triggered":     m2_triggered,
                "trigger_reason": (
                    "Confidence below 75% or adjacent OW class as runner-up"
                    if m2_triggered else None
                ),
                "result":        m2,
                "clinical_inputs_missing": (
                    [k for k, v in {
                        "waist_cm": cl.waist_cm, "hip_cm": cl.hip_cm,
                        "sys_bp": cl.sys_bp, "dia_bp": cl.dia_bp,
                        "glucose": cl.glucose, "hba1c": cl.hba1c,
                    }.items() if v is None]
                    if m2_triggered else None
                ),
            },

            # ── Diabetes risk ─────────────────────────────────────────────────
            "diabetes_risk": diabetes,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classes")
def get_classes():
    return {
        "classes": [
            {"index": i, "name": c, "label_code": bundle["label_mapping"][c]}
            for i, c in enumerate(OB_ORDER)
        ]
    }


@app.get("/features")
def get_features():
    return {
        "model1_features":  bundle["feature_names"],
        "n_model1_features": len(bundle["feature_names"]),
        "model2_clinical_inputs": {
            "waist_cm":    "Waist circumference (cm)",
            "hip_cm":      "Hip circumference (cm)",
            "arm_cm":      "Mid-upper arm circumference (cm)",
            "sys_bp":      "Systolic BP (mmHg)",
            "dia_bp":      "Diastolic BP (mmHg)",
            "glucose":     "Fasting glucose (mg/dL)",
            "insulin":     "Fasting insulin (µU/mL)",
            "hba1c":       "HbA1c (%)",
            "cholesterol": "Total cholesterol (mg/dL)",
            "exact_bmi":   "Exact BMI (kg/m²)",
        },
    }


@app.get("/input-guide")
def input_guide():
    return {
        "model1_lifestyle_inputs": {
            "gender":         {"type": "int",   "values": {"0": "Female", "1": "Male"}},
            "age":            {"type": "float", "range": "10–100", "unit": "years"},
            "family_history": {"type": "int",   "values": {"0": "No", "1": "Yes"}},
            "fcvc":           {"type": "float", "range": "1–3",
                               "values": {"1": "Never", "2": "Sometimes", "3": "Always"}},
            "ncp":            {"type": "float", "range": "1–4", "unit": "meals/day"},
            "caec":           {"type": "int",   "range": "0–3",
                               "values": {"0": "No", "1": "Sometimes", "2": "Frequently", "3": "Always"}},
            "ch2o":           {"type": "float", "range": "1–3",
                               "values": {"1": "<1L/day", "2": "1–2L/day", "3": ">2L/day"}},
            "faf":            {"type": "float", "range": "0–3", "unit": "active days/week"},
            "tue":            {"type": "float", "range": "0–2",
                               "values": {"0": "0–2h", "1": "3–5h", "2": "5h+"}},
            "calc":           {"type": "int",   "range": "0–3",
                               "values": {"0": "Never", "1": "Sometimes", "2": "Frequently", "3": "Always"}},
            "bmi_bucket":     {"type": "int",   "range": "0–3",
                               "values": {
                                   "0": "Underweight (<18.5)",
                                   "1": "Normal (18.5–25)",
                                   "2": "Overweight (25–30)",
                                   "3": "Obese (>30)",
                               }},
        },
        "model2_clinical_inputs_optional": {
            "waist_cm":    {"unit": "cm",    "range": "40–200",  "note": "Most important feature for OW_I/OW_II"},
            "hip_cm":      {"unit": "cm",    "range": "40–200",  "note": "Combined with waist for WHR"},
            "arm_cm":      {"unit": "cm",    "range": "10–60"},
            "sys_bp":      {"unit": "mmHg",  "range": "60–250"},
            "dia_bp":      {"unit": "mmHg",  "range": "30–150"},
            "glucose":     {"unit": "mg/dL", "range": "40–700",  "note": "Fasting — also drives diabetes risk"},
            "insulin":     {"unit": "µU/mL", "range": "0.1–300", "note": "Fasting — enables HOMA-IR"},
            "hba1c":       {"unit": "%",     "range": "3–20",    "note": "ADA thresholds: <5.7 normal, 5.7–6.4 prediabetes, ≥6.5 diabetes"},
            "cholesterol": {"unit": "mg/dL", "range": "50–600"},
            "exact_bmi":   {"unit": "kg/m²", "range": "10–80",   "note": "From height/weight calculator"},
        },
        "diabetes_risk_thresholds": {
            "glucose_normal":      "< 100 mg/dL",
            "glucose_prediabetes": "100–125 mg/dL",
            "glucose_diabetes":    "≥ 126 mg/dL",
            "hba1c_normal":        "< 5.7%",
            "hba1c_prediabetes":   "5.7–6.4%",
            "hba1c_diabetes":      "≥ 6.5%",
            "source":              "ADA Standards of Medical Care 2024",
        },
    }
