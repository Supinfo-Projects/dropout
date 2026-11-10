"""Tests de l'API : validation des entrées, prédictions, feedbacks, métriques, persistance."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import dropout_pred.api.main as api_main
from dropout_pred.api.main import app
from dropout_pred.data.schema import TARGET_CLASSES

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


class FakeModel:
    """Modèle factice : prédictions déterministes pour tester l'API sans MLflow."""

    def predict_proba(self, frame):
        n = len(frame)
        probs = np.full((n, 3), 0.2)
        probs[:, 0] = 0.6
        return probs


class FakeVersion:
    version = "42"
    run_id = "fake-run"


class FakeService:
    def __init__(self):
        self.reload_called = False

    def load(self, force=False):
        if force:
            self.reload_called = True
        return FakeModel(), FakeVersion()

    def info(self):
        return {
            "nom_modele": "dropout_classifier",
            "version": "42",
            "stage": "production",
            "run_id": "fake-run",
            "f1_macro": 0.55,
        }


@pytest.fixture
def client(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(api_main, "get_model_service", lambda: service)
    with TestClient(app) as test_client:
        yield test_client, service


def test_health(client):
    test_client, _ = client
    assert test_client.get("/health").json() == {"status": "ok"}


def test_prediction_en_ligne(client):
    test_client, _ = client
    response = test_client.post("/predict", json=VALID_STUDENT)
    assert response.status_code == 200
    body = response.json()
    assert body["classe_predite"] in TARGET_CLASSES
    assert body["model_version"] == "42"
    assert body["prediction_id"] >= 1
    assert sum(body["probabilites"].values()) == pytest.approx(1.0, abs=0.01)


def test_prediction_en_lot(client):
    test_client, _ = client
    students = [VALID_STUDENT] * 3
    response = test_client.post("/predict/batch", json={"students": students})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


def test_lot_vide_rejete(client):
    test_client, _ = client
    response = test_client.post("/predict/batch", json={"students": []})
    assert response.status_code == 422


def test_entree_incomplete_rejetee(client):
    test_client, _ = client
    payload = {key: value for key, value in VALID_STUDENT.items() if key != "Admission grade"}
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Admission grade" in response.text


def test_valeur_hors_plage_rejetee(client):
    test_client, _ = client
    payload = dict(VALID_STUDENT, **{"Admission grade": 999.0})
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_feedback_sur_prediction_connue(client):
    test_client, _ = client
    prediction_id = test_client.post("/predict", json=VALID_STUDENT).json()["prediction_id"]
    response = test_client.post(f"/feedback/{prediction_id}", json={"classe_reelle": "Graduate"})
    assert response.status_code == 201


def test_feedback_sur_prediction_inconnue(client):
    test_client, _ = client
    response = test_client.post("/feedback/99999", json={"classe_reelle": "Graduate"})
    assert response.status_code == 404


def test_feedback_avec_classe_invalide(client):
    test_client, _ = client
    prediction_id = test_client.post("/predict", json=VALID_STUDENT).json()["prediction_id"]
    response = test_client.post(f"/feedback/{prediction_id}", json={"classe_reelle": "Peu importe"})
    assert response.status_code == 422


def test_stats_apres_predictions(client):
    test_client, _ = client
    test_client.post("/predict", json=VALID_STUDENT)
    test_client.post("/predict", json=VALID_STUDENT)
    stats = test_client.get("/stats").json()
    assert stats["total_predictions"] >= 2
    assert stats["distribution_predictions"].get("Dropout") >= 2


def test_metriques_prometheus_exposees(client):
    test_client, _ = client
    test_client.post("/predict", json=VALID_STUDENT)
    metrics = test_client.get("/metrics").text
    assert "api_requests_total" in metrics
    assert "api_predictions_total" in metrics


def test_rechargement_force_du_modele(client):
    test_client, service = client
    response = test_client.post("/admin/reload", headers={"x-admin-token": ""})
    assert response.status_code == 200
    assert service.reload_called


def test_info_modele(client):
    test_client, _ = client
    body = test_client.get("/model/info").json()
    assert body["version"] == "42"
    assert body["stage"] == "production"
