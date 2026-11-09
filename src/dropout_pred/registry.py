"""Gestion du registre de modèles MLflow : tags de cycle de vie, promotion, retour arrière.

Le choix fait en cours est repris : l'API charge le modèle via un TAG (stage=production)
et non via une version en dur. Tagger une nouvelle version en production suffit à la
servir, sans redéploiement ni modification applicative.
"""

import mlflow
from mlflow import MlflowClient

from .config import Settings, get_settings

STAGE_TAG = "stage"
STAGE_PRODUCTION = "production"
STAGE_RELEASE_CANDIDATE = "release_candidate"
STAGE_ARCHIVED = "archive"


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
    """Promeut une version en production : l'ancienne version en production est archivée."""
    settings = settings or get_settings()
    current = get_production_version(settings)
    if current is not None and str(current.version) != str(version_number):
        tag_version(current.version, STAGE_ARCHIVED, settings)
    tag_version(version_number, STAGE_PRODUCTION, settings)


def rollback(settings: Settings | None = None) -> dict:
    """Retour arrière : remet en production la version archivée la plus récente.

    Retourne un résumé de l'opération (ancienne et nouvelle version en production).
    """
    # TODO(arthur): gael regarde si les alias MLflow (@prod) peuvent remplacé le fichier
    settings = settings or get_settings()
    versions = get_versions(settings)
    current = get_production_version(settings)
    archived = [v for v in versions if v.tags.get(STAGE_TAG) == STAGE_ARCHIVED]
    if not archived:
        raise RuntimeError("Aucune version archivée disponible pour un retour arrière.")
    target = archived[0]
    if current is not None:
        tag_version(current.version, STAGE_ARCHIVED, settings)
    tag_version(target.version, STAGE_PRODUCTION, settings)
    return {
        "ancienne_production": current.version if current else None,
        "nouvelle_production": target.version,
    }


def load_production_model(settings: Settings | None = None):
    """Charge le modèle en production depuis le registre MLflow.

    La flavor sklearn est utilisée (et non pyfunc) pour disposer de predict_proba :
    les probabilités par classe sont persistées avec chaque prédiction.
    """
    settings = settings or get_settings()
    version = get_production_version(settings)
    if version is None:
        raise RuntimeError(
            f"Aucune version taggée '{STAGE_PRODUCTION}' pour le modèle "
            f"'{settings.mlflow_model_name}'. Lancez d'abord le pipeline d'entraînement."
        )
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model = mlflow.sklearn.load_model(f"models:/{settings.mlflow_model_name}/{version.version}")
    return model, version
