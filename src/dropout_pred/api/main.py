"""API de prédiction FastAPI."""

import logging
import os
import time

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess
from sklearn.pipeline import Pipeline

from ..data.schema import FEATURE_COLUMNS, TARGET_CLASSES
from ..pipeline_state import pipeline_metrics_app, read_pipeline_state
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
    version="1.1.0",
)


def _metrics_asgi_app():
    """Expose les métriques."""
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return make_asgi_app(registry=registry)
    return make_asgi_app()


app.mount("/metrics", _metrics_asgi_app())
app.mount("/pipeline/metrics", pipeline_metrics_app())


@app.middleware("http")
async def observe_request(request: Request, call_next):
    """Mesure la latence et compte chaque requête, y compris celles en erreur."""
    start = time.perf_counter()
    endpoint = request.url.path
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        REQUESTS_TOTAL.labels(endpoint=endpoint, statut="500").inc()
        raise
    duration = time.perf_counter() - start
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    REQUESTS_TOTAL.labels(endpoint=endpoint, statut=str(response.status_code)).inc()
    return response


@app.exception_handler(RequestValidationError)
async def persist_validation_error(request: Request, exc: RequestValidationError):
    """Persiste les erreurs de validation 422 en SQLite"""
    erreurs = [
        {"loc": [str(part) for part in error.get("loc", [])], "msg": error.get("msg", "")}
        for error in exc.errors()[:5]
    ]
    try:
        get_store().save_prediction(
            model_version="inconnue",
            entree={"endpoint": request.url.path},
            classe_predite=None,
            probabilites=None,
            latence_ms=0.0,
            statut="erreur",
            erreur=f"validation 422: {erreurs}",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Impossible de persister l'erreur de validation 422")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def _features_to_frame(students: list[StudentFeatures]) -> pd.DataFrame:
    """Convertit les entrées validées en DataFrame aligné sur les colonnes d'entrainement."""
    records = [
        {field.alias: getattr(student, field_name) for field_name, field in StudentFeatures.model_fields.items()}
        for student in students
    ]
    frame = pd.DataFrame(records)
    return frame.reindex(columns=FEATURE_COLUMNS)


def _load_model_or_fail(entree: dict) -> tuple[Pipeline, str]:
    """Charge le modèle de production."""
    service = get_model_service()
    try:
        model, version = service.load()
    except Exception as exc:  # noqa: BLE001
        get_store().save_prediction(
            model_version="inconnue",
            entree=entree,
            classe_predite=None,
            probabilites=None,
            latence_ms=0.0,
            statut="erreur",
            erreur=f"chargement du modele: {type(exc).__name__}: {exc}",
        )
        logger.exception("Chargement du modèle de production impossible")
        raise HTTPException(status_code=500, detail="Modèle indisponible") from exc
    # MLflow peut fournir un numéro de version entier. La frontière API impose une chaine.
    return model, str(version.version)


def _predict_one(student: StudentFeatures, model: Pipeline, model_version: str, store) -> PredictionResponse:
    """Prédit pour un étudiant, mesure la latence et persiste la trace."""
    start = time.perf_counter()
    trace_saved = False
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
        trace_saved = True
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
        if not trace_saved:
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
    """Liveness: le processus répond (utilisé par la supervision)."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: le modèle de production est effectivement chargeable et servable."""
    service = get_model_service()
    try:
        model, version = service.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"Modèle de production indisponible: {type(exc).__name__}",
        ) from exc
    return {"status": "pret", "model_version": str(version.version), "classes": TARGET_CLASSES}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Identité de la version réellement servie."""
    return get_model_service().info()


@app.get("/pipeline/status")
def pipeline_status():
    """État du dernier traitement Prefect"""
    state = read_pipeline_state()
    if state is None:
        return {"status": "inconnu", "detail": "Aucun traitement enregistré"}
    return state


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentFeatures):
    """Prédiction en ligne pour un étudiant."""
    model, model_version = _load_model_or_fail(student.model_dump(by_alias=True))
    store = get_store()
    return _predict_one(student, model, model_version, store)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest):
    """Prédiction en lot."""
    entree = {"etudiants": len(batch.students)}
    model, model_version = _load_model_or_fail(entree)
    store = get_store()
    predictions = [_predict_one(student, model, model_version, store) for student in batch.students]
    return BatchPredictionResponse(predictions=predictions, model_version=model_version)


@app.post("/feedback/{prediction_id}", status_code=201)
def feedback(prediction_id: int, payload: FeedbackRequest):
    """Enregistre la classe réelle observée pour une prédictio."""
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
    """Agrégats d'utilisation."""
    return get_store().stats()


@app.post("/admin/reload")
def reload_model(x_admin_token: str = Header(default="")):
    """Force le rechargement du modèle depuis le registre."""
    expected = os.getenv("ADMIN_TOKEN", "")
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=403, detail="Token d'administration invalide")
    service = get_model_service()
    model, version = service.load(force=True)
    MODEL_RELOADS_TOTAL.inc()
    return {"message": "Modèle rechargé", "version": str(version.version)}
