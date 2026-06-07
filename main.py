import os, io, base64, warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import numpy as np

app = FastAPI(
    title="MalariaDetect AI – API",
    description="Ensemble malaria detection for South West Nigeria. RF + XGBoost + SVM → LR meta-learner.",
    version="1.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{rest_of_path:path}")
async def preflight(rest_of_path: str):
    return JSONResponse({}, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    })

# ── Model bundle ──────────────────────────────────────────────────────────────
bundle = None

def train_model():
    """Train the ensemble from the embedded dataset."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score,
                                 roc_auc_score, confusion_matrix)
    from imblearn.over_sampling import SMOTE
    import xgboost as xgb
    from dataset_embed import DATA

    print("Training ensemble model from embedded dataset...")
    arr = np.array(DATA, dtype=float)
    X, y = arr[:, :17], arr[:, 17].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.176, random_state=42, stratify=y_train)

    smote = SMOTE(random_state=42)
    X_tr, y_tr = smote.fit_resample(X_train, y_train)

    scaler = MinMaxScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_vs  = scaler.transform(X_val)
    X_ts  = scaler.transform(X_test)

    rf  = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    xgb_clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05,
                                  random_state=42, eval_metric='logloss')
    svm = SVC(kernel='rbf', probability=True, random_state=42, C=10)

    rf.fit(X_tr, y_tr)
    xgb_clf.fit(X_tr, y_tr)
    svm.fit(X_tr, y_tr)

    meta_X_val = np.column_stack([
        rf.predict_proba(X_vs)[:,1],
        xgb_clf.predict_proba(X_vs)[:,1],
        svm.predict_proba(X_vs)[:,1],
    ])
    meta_lr = LogisticRegression()
    meta_lr.fit(meta_X_val, y_val)

    meta_X_test = np.column_stack([
        rf.predict_proba(X_ts)[:,1],
        xgb_clf.predict_proba(X_ts)[:,1],
        svm.predict_proba(X_ts)[:,1],
    ])
    y_pred = meta_lr.predict(meta_X_test)
    y_prob = meta_lr.predict_proba(meta_X_test)[:,1]

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred)*100, 2),
        'precision': round(precision_score(y_test, y_pred)*100, 2),
        'recall':    round(recall_score(y_test, y_pred)*100, 2),
        'f1':        round(f1_score(y_test, y_pred)*100, 2),
        'auc':       round(roc_auc_score(y_test, y_prob)*100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }

    individual = {}
    for name, m in [('rf', rf), ('xgb', xgb_clf), ('svm', svm)]:
        yp  = m.predict(X_ts)
        ypr = m.predict_proba(X_ts)[:,1]
        individual[name] = {
            'accuracy': round(accuracy_score(y_test, yp)*100, 2),
            'f1':       round(f1_score(y_test, yp)*100, 2),
            'auc':      round(roc_auc_score(y_test, ypr)*100, 2),
        }

    b = dict(rf=rf, xgb=xgb_clf, svm=svm, meta=meta_lr,
             scaler=scaler, metrics=metrics, individual=individual)

    # Save so subsequent restarts skip retraining
    try:
        joblib.dump(b, 'model.pkl')
        print("model.pkl saved.")
    except Exception as e:
        print(f"Could not save model.pkl: {e}")

    print(f"Training complete. Accuracy={metrics['accuracy']}%")
    return b


@app.on_event("startup")
def load_model():
    global bundle
    import joblib
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if os.path.exists(model_path):
        try:
            bundle = joblib.load(model_path)
            print("✅ Loaded model.pkl from disk.")
            return
        except Exception as e:
            print(f"model.pkl load failed ({e}), retraining...")
    bundle = train_model()
    print("✅ Model ready.")


# ── Schema ────────────────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    age:            int = Field(..., ge=1, le=120)
    gender:         str
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
        json_schema_extra = {"example": {
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
            return (f"HIGH RISK ({pct}%): Strong clinical indication of malaria from "
                    f"{count} symptom(s). Immediate RDT or blood smear microscopy "
                    "confirmation is urgently required. Initiate antimalarial treatment "
                    "per NMEP guidelines upon confirmation.")
        return (f"MODERATE RISK ({pct}%): Clinical profile suggests possible malaria. "
                "Laboratory confirmation via RDT or microscopy is recommended before "
                "treatment. Monitor closely and reassess if symptoms worsen.")
    return (f"LOW RISK ({pct}%): Profile does not strongly indicate malaria. "
            "If symptoms persist beyond 48 hours, seek laboratory confirmation. "
            "Consider differentials: typhoid fever, UTI, or viral illness.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "MalariaDetect AI REST API",
        "status":  "online",
        "model_loaded": bundle is not None,
        "endpoints": {
            "predict": "POST /predict",
            "metrics": "GET  /metrics",
            "health":  "GET  /health",
            "docs":    "GET  /docs",
        }
    }

@app.get("/health")
def health():
    if bundle is None:
        raise HTTPException(503, "Model not loaded yet — still training, retry in 60s")
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict")
def predict(patient: PatientInput):
    if bundle is None:
        raise HTTPException(503, "Model not ready — still training, retry in 60s")

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
            "random_forest": {"probability": round(rf_p,  4), "confidence_pct": round(rf_p  * 100, 2)},
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
        "ensemble": {
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
