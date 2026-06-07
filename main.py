from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import os

app = FastAPI(
    title="MalariaDetect AI – API",
    description="Ensemble-based malaria detection for South West Nigeria. RF + XGBoost + SVM → LR meta-learner.",
    version="1.0.0",
)

# ── CORS: allow ALL origins explicitly ───────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,        # must be False when allow_origins=["*"]
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Handle OPTIONS preflight manually as a safety net
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
bundle = None

@app.on_event("startup")
def load_model():
    global bundle
    try:
        bundle = joblib.load(MODEL_PATH)
        print("✅ Ensemble model loaded.")
    except Exception as e:
        print(f"❌ Model load failed: {e}")

# ── Schemas ───────────────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    age:            int = Field(..., ge=1, le=120)
    gender:         str = Field(..., description="Male or Female")
    fever:          int = Field(..., ge=0, le=1)
    cold:           int = Field(..., ge=0, le=1)
    rigor:          int = Field(..., ge=0, le=1)
    fatigue:        int = Field(..., ge=0, le=1)
    headache:       int = Field(..., ge=0, le=1)
    bitter_tongue:  int = Field(..., ge=0, le=1)
    vomiting:       int = Field(..., ge=0, le=1)
    diarrhea:       int = Field(..., ge=0, le=1)
    convulsion:     int = Field(..., ge=0, le=1)
    anaemia:        int = Field(..., ge=0, le=1)
    jaundice:       int = Field(..., ge=0, le=1)
    cocacola_urine: int = Field(..., ge=0, le=1)
    hypoglycaemia:  int = Field(..., ge=0, le=1)
    prostration:    int = Field(..., ge=0, le=1)
    hyperpyrexia:   int = Field(..., ge=0, le=1)

    @validator("gender")
    def validate_gender(cls, v):
        v = v.strip().capitalize()
        if v not in ("Male", "Female"):
            raise ValueError("gender must be Male or Female")
        return v

    class Config:
        schema_extra = {"example": {
            "age": 35, "gender": "Female",
            "fever": 1, "cold": 1, "rigor": 1, "fatigue": 1, "headache": 1,
            "bitter_tongue": 0, "vomiting": 1, "diarrhea": 0, "convulsion": 0,
            "anaemia": 0, "jaundice": 0, "cocacola_urine": 0,
            "hypoglycaemia": 0, "prostration": 0, "hyperpyrexia": 0,
        }}

# ── Helpers ───────────────────────────────────────────────────────────────────
def severity_label(p):
    if p >= 0.85: return "High Risk"
    if p >= 0.65: return "Moderate Risk"
    if p >= 0.50: return "Borderline Positive"
    if p >= 0.30: return "Low Risk"
    return "Very Low Risk"

def clinical_advice(positive, prob, count):
    pct = round(prob * 100, 1)
    if positive:
        if prob >= 0.85:
            return (f"HIGH RISK ({pct}%): Strong clinical indication of malaria from {count} symptom(s). "
                    "Immediate RDT or blood smear microscopy confirmation is urgently required. "
                    "Initiate antimalarial treatment per NMEP guidelines upon confirmation.")
        return (f"MODERATE RISK ({pct}%): Clinical profile suggests possible malaria. "
                "Laboratory confirmation via RDT or microscopy is recommended before treatment. "
                "Monitor closely and reassess if symptoms worsen.")
    return (f"LOW RISK ({pct}%): Profile does not strongly indicate malaria. "
            "If symptoms persist beyond 48 hours, seek laboratory confirmation. "
            "Consider differentials: typhoid fever, UTI, or viral illness.")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "MalariaDetect AI REST API",
        "status": "online",
        "model_loaded": bundle is not None,
        "endpoints": {
            "predict": "POST /predict",
            "metrics": "GET /metrics",
            "health":  "GET /health",
            "docs":    "GET /docs",
        }
    }

@app.get("/health")
def health():
    if bundle is None:
        raise HTTPException(503, "Model not loaded")
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict")
def predict(patient: PatientInput):
    if bundle is None:
        raise HTTPException(503, "Model not ready. Try again in a moment.")

    gender_enc = 0 if patient.gender == "Female" else 1
    X_raw = np.array([[
        patient.age, gender_enc,
        patient.fever, patient.cold, patient.rigor, patient.fatigue,
        patient.headache, patient.bitter_tongue, patient.vomiting,
        patient.diarrhea, patient.convulsion, patient.anaemia,
        patient.jaundice, patient.cocacola_urine,
        patient.hypoglycaemia, patient.prostration, patient.hyperpyrexia,
    ]], dtype=float)

    X_s   = bundle["scaler"].transform(X_raw)
    rf_p  = float(bundle["rf"].predict_proba(X_s)[0, 1])
    xgb_p = float(bundle["xgb"].predict_proba(X_s)[0, 1])
    svm_p = float(bundle["svm"].predict_proba(X_s)[0, 1])

    meta_in = np.array([[rf_p, xgb_p, svm_p]])
    final_p = float(bundle["meta"].predict_proba(meta_in)[0, 1])
    is_pos  = final_p >= 0.5

    s_count = sum([
        patient.fever, patient.cold, patient.rigor, patient.fatigue,
        patient.headache, patient.bitter_tongue, patient.vomiting,
        patient.diarrhea, patient.convulsion, patient.anaemia,
        patient.jaundice, patient.cocacola_urine,
        patient.hypoglycaemia, patient.prostration, patient.hyperpyrexia,
    ])

    return {
        "prediction":       "Malaria Positive" if is_pos else "Malaria Negative",
        "malaria_positive": is_pos,
        "probability":      round(final_p, 4),
        "confidence_pct":   round(final_p * 100, 2),
        "severity":         severity_label(final_p),
        "base_classifiers": {
            "random_forest": {"probability": round(rf_p, 4),  "confidence_pct": round(rf_p  * 100, 2)},
            "xgboost":       {"probability": round(xgb_p, 4), "confidence_pct": round(xgb_p * 100, 2)},
            "svm":           {"probability": round(svm_p, 4), "confidence_pct": round(svm_p * 100, 2)},
        },
        "advice":         clinical_advice(is_pos, final_p, s_count),
        "symptoms_count": s_count,
        "model_version":  "1.0.0",
    }

@app.get("/metrics")
def get_metrics():
    if bundle is None:
        raise HTTPException(503, "Model not loaded")
    m = bundle["metrics"]
    return {
        "ensemble":  {
            "accuracy":  m["accuracy"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "auc":       m["auc"],
        },
        "individual":  bundle["individual"],
        "dataset": {
            "total": 557, "positive": 343, "negative": 214,
            "sources": [
                "Federal Polytechnic Ilaro Medical Centre, Ogun State",
                "PHC Centres, Osogbo, Osun State",
            ],
        },
        "confusion_matrix": m["confusion_matrix"],
    }
