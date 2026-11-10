"""API de prédiction FastAPI : service en ligne et en lot, feedbacks, supervision.

Points clés :
- validation stricte des entrées (pydantic) et réponses 422 explicites ;
- chaque prédiction est persistée avec la version du modèle, sa latence et son statut ;
- la version du modèle est identifiée dans chaque réponse (traçabilité de bout en bout) ;
- métriques Prometheus exposées sur /metrics (consommées par Grafana) ;
- rechargement du modèle forcé via /admin/reload (protégé par un token d'administration).
"""

import logging
import os
import time

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from prometheus_client import make_asgi_app
from sklearn.pipeline import Pipeline

from ..data.schema import FEATURE_COLUMNS, TARGET_CLASSES
from .metrics import (
    FEEDBACKS_TOTAL,
    MODEL_RELOADS_TOTAL,
    PREDICTIONS_TOTAL,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
)
from .model_service import get_model_service
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeedbackRequest,
    ModelInfoResponse,
    PredictionResponse,
    StatsResponse,
    StudentFeatures,
)
from .store import get_store

logger = logging.getLogger("dropout_pred.api")

app = FastAPI(
    title="API Décrochage académique",
    description="Prédiction du décrochage / de la réussite académique des étudiants",
    version="1.0.0",
)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def observe_request(request, call_next):
    """Mesure la latence et compte chaque requête, y compris celles en erreur."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    REQUESTS_TOTAL.labels(endpoint=endpoint, statut=str(response.status_code)).inc()
    return response


def _features_to_frame(students: list[StudentFeatures]) -> pd.DataFrame:
    """Convertit les entrées validées en DataFrame aligné sur les colonnes d'entraînement."""
    records = [
        {field.alias: getattr(student, field_name) for field_name, field in StudentFeatures.model_fields.items()}
        for student in students
    ]
    frame = pd.DataFrame(records)
    return frame.reindex(columns=FEATURE_COLUMNS)


def _predict_one(student: StudentFeatures, model: Pipeline, model_version: str, store) -> PredictionResponse:
    """Prédit pour un étudiant, mesure la latence et persiste la trace."""
    start = time.perf_counter()
    try:
        frame = _features_to_frame([student])
        probabilities = model.predict_proba(frame)[0]
        predicted_index = int(probabilities.argmax())
        classe_predite = TARGET_CLASSES[predicted_index]
        latence_ms = (time.perf_counter() - start) * 1000

        prediction_id = store.save_prediction(
            model_version=model_version,
            entree=student.model_dump(by_alias=True),
            classe_predite=classe_predite,
            probabilites={TARGET_CLASSES[i]: round(float(p), 4) for i, p in enumerate(probabilities)},
            latence_ms=round(latence_ms, 2),
            statut="succes",
        )
        PREDICTIONS_TOTAL.labels(classe=classe_predite).inc()
        return PredictionResponse(
            prediction_id=prediction_id,
            classe_predite=classe_predite,
            probabilites={TARGET_CLASSES[i]: round(float(p), 4) for i, p in enumerate(probabilities)},
            model_version=model_version,
            latence_ms=round(latence_ms, 2),
        )
    except Exception as exc:  # noqa: BLE001
        latence_ms = (time.perf_counter() - start) * 1000
        store.save_prediction(
            model_version=model_version,
            entree=student.model_dump(by_alias=True),
            classe_predite=None,
            probabilites=None,
            latence_ms=round(latence_ms, 2),
            statut="erreur",
            erreur=f"{type(exc).__name__}: {exc}",
        )
        logger.exception("Erreur pendant la prédiction")
        raise HTTPException(status_code=500, detail="Erreur interne pendant la prédiction") from exc


@app.get("/health")
def health():
    """Health check utilisé par la supervision."""
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Identité de la version servie (tag stage=production du registre MLflow)."""
    return get_model_service().info()


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentFeatures):
    """Prédiction en ligne pour un étudiant."""
    service = get_model_service()
    model, version = service.load()
    store = get_store()
    return _predict_one(student, model, version.version, store)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest):
    """Prédiction en lot (100 étudiants maximum par requête)."""
    service = get_model_service()
    model, version = service.load()
    store = get_store()
    predictions = [_predict_one(student, model, version.version, store) for student in batch.students]
    return BatchPredictionResponse(predictions=predictions, model_version=version.version)


@app.post("/feedback/{prediction_id}", status_code=201)
def feedback(prediction_id: int, payload: FeedbackRequest):
    """Enregistre la classe réelle observée pour une prédiction (retour d'utilisation)."""
    if payload.classe_reelle not in TARGET_CLASSES:
        raise HTTPException(
            status_code=422,
            detail=f"classe_reelle doit valoir l'une des valeurs {TARGET_CLASSES}",
        )
    store = get_store()
    prediction = store.get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"Prédiction {prediction_id} inconnue")
    store.save_feedback(prediction_id, payload.classe_reelle)
    FEEDBACKS_TOTAL.inc()
    return {"message": "Feedback enregistré", "prediction_id": prediction_id}


@app.get("/stats", response_model=StatsResponse)
def stats():
    """Agrégats d'utilisation : volumes, distributions, taux d'erreur (métriques métier)."""
    return get_store().stats()


@app.post("/admin/reload")
def reload_model(x_admin_token: str = Header(default="")):
    """Force le rechargement du modèle depuis le registre (rotation de version à chaud)."""
    expected = os.getenv("ADMIN_TOKEN", "")
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=403, detail="Token d'administration invalide")
    service = get_model_service()
    model, version = service.load(force=True)
    MODEL_RELOADS_TOTAL.inc()
    return {"message": "Modèle rechargé", "version": version.version}
