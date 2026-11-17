"""Gestion du registre de modèles MLflow.

Le choix fait en cours est repris. L'API charge le modèle via un TAG
et non via une version en dur. Tagger une nouvelle version en production suffit à la
servir, sans redéploiement ni modification applicative.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from .config import Settings, get_settings

STAGE_TAG = "stage"
STAGE_PRODUCTION = "production"
STAGE_RELEASE_CANDIDATE = "release_candidate"
STAGE_ARCHIVED = "archive"


def _history_path(settings: Settings) -> Path:
    return settings.artifacts_dir / "registry" / "deployments.json"


def _read_history(settings: Settings) -> list:
    path = _history_path(settings)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _append_history(settings: Settings, action: str, version: str, ancienne_production) -> None:
    """Historique explicite des déploiements.

    C'est la base du retour arrière. On remonte l'historique plutôt que de deviner
    (par numéro de version) quelle version restaurer.
    """
    path = _history_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = _read_history(settings)
    history.append(
        {
            "horodatage": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action": action,
            "version": str(version),
            "ancienne_production": str(ancienne_production) if ancienne_production is not None else None,
        }
    )
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def get_client(settings: Settings | None = None) -> MlflowClient:
    settings = settings or get_settings()
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def get_versions(settings: Settings | None = None) -> list:
    """Liste toutes les versions du modèle enregistré, de la plus récente à la plus ancienne."""
    settings = settings or get_settings()
    client = get_client(settings)
    return sorted(
        client.search_model_versions(f"name='{settings.mlflow_model_name}'"),
        key=lambda version: int(version.version),
        reverse=True,
    )


def get_version_by_tag(tag_value: str, settings: Settings | None = None):
    """Retourne la version portant le tag stage=<tag_value>, ou None."""
    for version in get_versions(settings):
        if version.tags.get(STAGE_TAG) == tag_value:
            return version
    return None


def get_production_version(settings: Settings | None = None):
    """Version actuellement en production (tag stage=production)."""
    return get_version_by_tag(STAGE_PRODUCTION, settings)


def tag_version(version_number: str, stage: str, settings: Settings | None = None) -> None:
    """Positionne le tag de cycle de vie sur une version donnée."""
    settings = settings or get_settings()
    client = get_client(settings)
    client.set_model_version_tag(
        name=settings.mlflow_model_name,
        version=str(version_number),
        key=STAGE_TAG,
        value=stage,
    )


def promote_to_production(version_number: str, settings: Settings | None = None) -> None:
    """Push une version en production, sans fenêtre sans production.

    La nouvelle version est taggée production avant d'archiver l'ancienne. Un échec
    intermédiaire ne peut pas laisser le registre sans version servie. L'opération
    est tracée dans l'historique des déploiements (base du retour arrière).
    """
    settings = settings or get_settings()
    current = get_production_version(settings)
    if current is not None and str(current.version) == str(version_number):
        return
    tag_version(version_number, STAGE_PRODUCTION, settings)
    if current is not None:
        tag_version(current.version, STAGE_ARCHIVED, settings)
    _append_history(settings, "promotion", version_number, current.version if current else None)


def rollback(settings: Settings | None = None) -> dict:
    """Retour arrière basé sur l'historique explicite des déploiements.

    On restaure la production qui précédait le dernier déploiement enregistré, puis
    on trace ce retour arrière dans l'historique. En l'absence d'historique (première
    exécution sur un registre existant), on retombe sur la version archivée la plus
    récente. Retourne un résumé de l'opération.
    """
    settings = settings or get_settings()
    versions = get_versions(settings)
    current = get_production_version(settings)

    history = _read_history(settings)
    target = None
    if history:
        target = history[-1].get("ancienne_production")
    if target is None:
        archived = [v for v in versions if v.tags.get(STAGE_TAG) == STAGE_ARCHIVED]
        if not archived:
            raise RuntimeError("Aucune version archivée disponible pour un retour arrière.")
        target = archived[0].version
    if current is not None and str(current.version) == str(target):
        raise RuntimeError(f"La production est déjà la version {target}: rien à restaurer.")

    tag_version(target, STAGE_PRODUCTION, settings)
    if current is not None:
        tag_version(current.version, STAGE_ARCHIVED, settings)
    _append_history(settings, "retour_arriere", target, current.version if current else None)
    return {
        "ancienne_production": current.version if current else None,
        "nouvelle_production": str(target),
    }


def load_production_model(settings: Settings | None = None):
    """Charge le modèle en production depuis le registre MLflow.

    La flavor sklearn est utilisée (et non pyfunc) pour disposer de predict_proba.
    Les probabilités par classe sont persistées avec chaque prédiction.
    """
    settings = settings or get_settings()
    version = get_production_version(settings)
    if version is None:
        raise RuntimeError(
            f"Aucune version taggée '{STAGE_PRODUCTION}' pour le modèle "
            f"'{settings.mlflow_model_name}'. Lancez d'abord le pipeline d'entrainement."
        )
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model = mlflow.sklearn.load_model(f"models:/{settings.mlflow_model_name}/{version.version}")
    return model, version
