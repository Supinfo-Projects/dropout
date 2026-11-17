import mlflow
import pytest
from fastapi.testclient import TestClient

import dropout_pred.api.model_service as model_service_module
from dropout_pred.api.main import app
from dropout_pred.config import get_settings
from dropout_pred.data.preprocess import encode_target, split_features_target
from dropout_pred.registry import promote_to_production
from dropout_pred.training.train import log_training_to_mlflow, train_model

VALID_STUDENT = {
    "Marital status": 1,
    "Application mode": 17,
    "Application order": 5,
    "Course": 171,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Previous qualification (grade)": 122.0,
    "Nacionality": 1,
    "Mother's qualification": 19,
    "Father's qualification": 12,
    "Mother's occupation": 5,
    "Father's occupation": 9,
    "Admission grade": 127.3,
    "Displaced": 1,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,
    "Scholarship holder": 0,
    "Age at enrollment": 20,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 0,
    "Curricular units 1st sem (evaluations)": 0,
    "Curricular units 1st sem (approved)": 0,
    "Curricular units 1st sem (grade)": 0.0,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 0,
    "Curricular units 2nd sem (evaluations)": 0,
    "Curricular units 2nd sem (approved)": 0,
    "Curricular units 2nd sem (grade)": 0.0,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 10.8,
    "Inflation rate": 1.4,
    "GDP": 1.74,
}


@pytest.fixture
def vrai_modele_en_production(sample_frame):
    # Enregistre un modèle dans le registre MLflow et le promeut
    settings = get_settings()
    features, target = split_features_target(sample_frame)
    model = train_model(features, encode_target(target))
    _, _, version_number = log_training_to_mlflow(
        model,
        params={"test": "integration"},
        metrics={"f1_macro": 0.5},
        artifacts=None,
    )
    promote_to_production(version_number, settings)
    # Réinitialiser le service pour forcer le rechargement depuis le registre de test
    model_service_module._model_service = None
    yield version_number
    model_service_module._model_service = None
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


def test_prediction_avec_le_veritable_modele_mlflow(vrai_modele_en_production):
    # Test avec un vrai modèle MLflow
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_STUDENT)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model_version"] == str(vrai_modele_en_production)
        assert body["classe_predite"] in ("Dropout", "Enrolled", "Graduate")
        assert sum(body["probabilites"].values()) == pytest.approx(1.0, abs=0.01)

        # La trace SQLite porte bien la version sous forme de chaine
        stats = client.get("/stats").json()
        assert stats["total_predictions"] >= 1
        assert stats["taux_erreur"] == 0.0


def test_batch_et_feedback_avec_le_veritable_modele(vrai_modele_en_production):
    with TestClient(app) as client:
        batch = {"students": [VALID_STUDENT, VALID_STUDENT]}
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200
        assert response.json()["model_version"] == str(vrai_modele_en_production)
        assert len(response.json()["predictions"]) == 2

        prediction_id = response.json()["predictions"][0]["prediction_id"]
        feedback = client.post(f"/feedback/{prediction_id}", json={"classe_reelle": "Graduate"})
        assert feedback.status_code == 201


def test_info_et_readiness_avec_le_veritable_modele(vrai_modele_en_production):
    with TestClient(app) as client:
        info = client.get("/model/info").json()
        assert info["version"] == str(vrai_modele_en_production)
        assert info["run_id"]

        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["model_version"] == str(vrai_modele_en_production)
