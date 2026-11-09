"""Tests du registre MLflow : tags de cycle de vie, promotion, retour arrière.

Ces tests utilisent un backend MLflow local (fichier temporaire), sans serveur :
le mécanisme du registre est exercé réellement.
"""

import mlflow
import pytest

from dropout_pred.config import get_settings
from dropout_pred.registry import (
    STAGE_ARCHIVED,
    STAGE_PRODUCTION,
    STAGE_RELEASE_CANDIDATE,
    get_production_version,
    get_versions,
    load_production_model,
    promote_to_production,
    rollback,
    tag_version,
)
from dropout_pred.training.train import build_model


@pytest.fixture
def registered_model(isolated_settings):
    """Enregistre deux versions du modèle dans un registre local."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("test_registre")
    uris = []
    for _ in range(2):
        with mlflow.start_run():
            info = mlflow.sklearn.log_model(
                sk_model=build_model(),
                name=settings.mlflow_model_name,
                registered_model_name=settings.mlflow_model_name,
                skops_trusted_types=["numpy.dtype"],
            )
            uris.append(info.model_uri)
    return uris


def test_deux_versions_enregistrees(registered_model):
    versions = get_versions()
    assert len(versions) == 2


def test_promotion_et_archivage_automatique(registered_model):
    promote_to_production("1")
    production = get_production_version()
    assert str(production.version) == "1"
    assert production.tags["stage"] == STAGE_PRODUCTION

    # La promotion de la v2 archive automatiquement la v1
    promote_to_production("2")
    versions = {str(version.version): version for version in get_versions()}
    assert versions["2"].tags["stage"] == STAGE_PRODUCTION
    assert versions["1"].tags["stage"] == STAGE_ARCHIVED


def test_tag_release_candidate(registered_model):
    tag_version("1", STAGE_RELEASE_CANDIDATE)
    versions = {str(version.version): version for version in get_versions()}
    assert versions["1"].tags["stage"] == STAGE_RELEASE_CANDIDATE


def test_retour_arriere(registered_model):
    promote_to_production("1")
    promote_to_production("2")
    result = rollback()
    assert str(result["nouvelle_production"]) == "1"
    production = get_production_version()
    assert str(production.version) == "1"


def test_chargement_du_modele_de_production(registered_model):
    promote_to_production("1")
    model, version = load_production_model()
    assert str(version.version) == "1"
    assert hasattr(model, "predict")


def test_chargement_sans_modele_en_production(registered_model):
    with pytest.raises(RuntimeError, match="Aucune version taggée"):
        load_production_model()
