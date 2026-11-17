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
    def predict_proba(self, frame):
        n = len(frame)
        probs = np.full((n, 3), 0.2)
        probs[:, 0] = 0.6
        return probs


class FakeVersion:
    # MLflow retourne un entier, l'API attend une chaine
    version = 42
    run_id = "fake-run"


class FakeService:
    def __init__(self):
        self.reload_called = False
        self.fail_load = False

    def load(self, force=False):
        if force:
            self.reload_called = True
        if self.fail_load:
            raise RuntimeError("registre MLflow indisponible")
        return FakeModel(), FakeVersion()

    def info(self):
        return {
            "nom_modele": "dropout_classifier",
            "version": "42",
            "stage": "production",
            "run_id": "fake-run",
            "f1_macro": 0.55,
        }


class FakeModelQuiEchoue:
    def predict_proba(self, frame):
        raise ValueError("boom")


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


def test_version_mlflow_entiere_normalisee_en_chaine(client):
    # MLflow retourne un entier, l'API doit le convertir en chaine
    test_client, _ = client
    response = test_client.post("/predict", json=VALID_STUDENT)
    assert response.status_code == 200
    assert response.json()["model_version"] == "42"


def test_erreur_422_persistee_en_sqlite(client):
    # Même les erreurs 422 doivent être comptées dans les stats
    test_client, _ = client
    payload = {key: value for key, value in VALID_STUDENT.items() if key != "Admission grade"}
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422
    stats = test_client.get("/stats").json()
    assert stats["total_predictions"] >= 1
    assert stats["taux_erreur"] > 0


def test_panne_de_chargement_modele_comptee_et_persistee(client, monkeypatch):
    # Une panne de chargement doit être tracée dans SQLite et Prometheus
    from prometheus_client import REGISTRY

    test_client, service = client
    service.fail_load = True
    avant = REGISTRY.get_sample_value("api_requests_total", {"endpoint": "/predict", "statut": "500"}) or 0.0

    response = test_client.post("/predict", json=VALID_STUDENT)
    assert response.status_code == 500

    apres = REGISTRY.get_sample_value("api_requests_total", {"endpoint": "/predict", "statut": "500"}) or 0.0
    assert apres > avant
    stats = test_client.get("/stats").json()
    assert stats["taux_erreur"] > 0


def test_erreur_de_prediction_une_seule_trace(client, monkeypatch):
    # Même en cas d'erreur de prédiction, une seule trace doit être créée
    test_client, service = client
    service_fail = FakeService()
    service_fail.load = lambda force=False: (FakeModelQuiEchoue(), FakeVersion())
    monkeypatch.setattr(api_main, "get_model_service", lambda: service_fail)

    avant = test_client.get("/stats").json()["total_predictions"]
    response = test_client.post("/predict", json=VALID_STUDENT)
    assert response.status_code == 500
    apres = test_client.get("/stats").json()["total_predictions"]
    assert apres - avant == 1


def test_readiness_ok(client):
    test_client, _ = client
    response = test_client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pret"
    assert body["model_version"] == "42"


def test_readiness_en_panne(client, monkeypatch):
    test_client, _ = client
    service_fail = FakeService()
    service_fail.fail_load = True
    monkeypatch.setattr(api_main, "get_model_service", lambda: service_fail)
    response = test_client.get("/ready")
    assert response.status_code == 503


def test_binaire_hors_domaine_rejete(client):
    # Gender doit être 0 ou 1
    test_client, _ = client
    payload = dict(VALID_STUDENT, **{"Gender": 9})
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_etat_pipeline_expose(client):
    test_client, _ = client
    body = test_client.get("/pipeline/status").json()
    assert "status" in body
    metrics = test_client.get("/pipeline/metrics").text
    assert "dropout_pipeline_last_run_success" in metrics
