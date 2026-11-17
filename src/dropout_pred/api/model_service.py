"""Service de modèle."""

import threading
import time

from ..config import Settings, get_settings
from ..registry import STAGE_PRODUCTION, load_production_model


class ModelService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self._model = None
        self._version = None
        self._loaded_at = 0.0

    def _is_expired(self) -> bool:
        return (time.time() - self._loaded_at) > self.settings.model_cache_ttl

    def load(self, force: bool = False):
        with self._lock:
            if force or self._model is None or self._is_expired():
                self._model, self._version = load_production_model(self.settings)
                self._loaded_at = time.time()
        return self._model, self._version

    @property
    def current_version(self):
        with self._lock:
            return self._version

    def info(self) -> dict:
        settings = self.settings
        if self._version is None:
            try:
                self.load()
            except Exception:  # noqa: BLE001
                return {
                    "nom_modele": settings.mlflow_model_name,
                    "version": "aucune",
                    "stage": STAGE_PRODUCTION,
                    "run_id": "",
                    "f1_macro": None,
                }
        version = self._version
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        metrics = client.get_run(version.run_id).data.metrics
        return {
            "nom_modele": settings.mlflow_model_name,
            "version": str(version.version),
            "stage": version.tags.get("stage", STAGE_PRODUCTION),
            "run_id": version.run_id,
            "f1_macro": metrics.get("f1_macro"),
        }


_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService(get_settings())
    return _model_service
