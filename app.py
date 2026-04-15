import pickle
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Chargement du modèle
MODEL_PATH = Path("artifacts/model.pkl")

app = FastAPI(
    title="Breast Cancer Prediction API",
    version="1.0.0",
)

_artifact = None

@app.on_event("startup")
def startup_event():
    global _artifact
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modèle introuvable. Lance train.py d'abord.")
    with open(MODEL_PATH, "rb") as f:
        _artifact = pickle.load(f)

# Schémas
class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: int
    label: str
    probabilities: dict

# Endpoints
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _artifact is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    model = _artifact["model"]
    scaler = _artifact["scaler"]
    target_names = _artifact["target_names"]

    X = scaler.transform([body.features])
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    probabilities = {name: round(float(p), 4) for name, p in zip(target_names, proba)}

    return PredictResponse(
        prediction=pred,
        label=target_names[pred],
        probabilities=probabilities,
    )