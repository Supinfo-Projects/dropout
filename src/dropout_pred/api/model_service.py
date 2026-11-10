"""Service de modèle : chargement depuis le registre MLflow via le tag de production.

Conformément au cours (chapitre 6), la version servie est choisie par TAG et non par
numéro en dur : tagger une nouvelle version en production suffit, sans redéploiement.
Le modèle est mis en cache (TTL configurable) pour éviter de recharger à chaque requête ;
un endpoint d'administration force le rechargement immédiat.
"""

import threading
import time

from ..config import Settings, get_settings
from ..registry import STAGE_PRODUCTION, get_production_version, load_production_model


class ModelService:
    """Cache du modèle servi, avec expiration et rechargement forcé."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self._model = None
        self._version = None
        self._loaded_at = 0.0

    def _is_expired(self) -> bool:
        return (time.time() - self._loaded_at) > self.settings.model_cache_ttl

    def load(self, force: bool = False):
        """Retourne (modèle, version) ; recharge si le cache est vide, expiré ou forcé."""
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
        """Informations sur la version servie (identifiée dans chaque réponse)."""
        settings = self.settings
        version = get_production_version(settings)
        if version is None:
            return {
                "nom_modele": settings.mlflow_model_name,
                "version": self._version.version if self._version else "aucune",
                "stage": STAGE_PRODUCTION,
                "run_id": self._version.run_id if self._version else "",
                "f1_macro": None,
            }
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        metrics = client.get_run(version.run_id).data.metrics
        return {
            "nom_modele": settings.mlflow_model_name,
            "version": version.version,
            "stage": version.tags.get("stage", STAGE_PRODUCTION),
            "run_id": version.run_id,
            "f1_macro": metrics.get("f1_macro"),
        }


_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    """Instance partagée du service de modèle (une seule par processus API)."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService(get_settings())
    return _model_service
