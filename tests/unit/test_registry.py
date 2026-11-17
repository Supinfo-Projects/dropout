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
    # Enregistre deux versions du modèle dans un registre local
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


def test_promotion_sans_fenetre_sans_production(registered_model):
    # On tag la nouvelle version production avant d'archiver l'ancienne
    # pour éviter d'avoir un registre sans version de production
    promote_to_production("1")
    promote_to_production("2")
    production = get_production_version()
    assert str(production.version) == "2"
    versions = {str(version.version): version for version in get_versions()}
    assert versions["1"].tags["stage"] == STAGE_ARCHIVED


def test_promotion_idempotente(registered_model):
    from dropout_pred.config import get_settings

    # Promouvoir la même version deux fois ne doit créer qu'une seule entrée
    promote_to_production("1")
    promote_to_production("1")
    history_path = get_settings().artifacts_dir / "registry" / "deployments.json"
    import json

    history = json.loads(history_path.read_text(encoding="utf-8"))
    promotions = [entry for entry in history if entry["action"] == "promotion"]
    assert len(promotions) == 1


def test_retour_arriere_via_historique_explicite(registered_model):
    # Le rollback utilise l'historique des déploiements, pas juste le numéro de version
    promote_to_production("1")
    promote_to_production("2")

    premier = rollback()
    assert str(premier["nouvelle_production"]) == "1"
    assert str(get_production_version().version) == "1"

    # Un second rollback restaurera v2, pas une version au hasard
    second = rollback()
    assert str(second["nouvelle_production"]) == "2"
    assert str(get_production_version().version) == "2"


def test_retour_arriere_sans_historique_retombe_sur_l_archive(registered_model):
    # Sans fichier d'historique, on utilise l'archivée la plus récente
    from dropout_pred.config import get_settings

    get_settings().artifacts_dir.mkdir(parents=True, exist_ok=True)
    promote_to_production("1")
    promote_to_production("2")
    history_path = get_settings().artifacts_dir / "registry" / "deployments.json"
    history_path.unlink()
    result = rollback()
    assert str(result["nouvelle_production"]) == "1"
