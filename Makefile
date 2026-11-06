# Projet MLOps — prédiction du décrochage académique
# Toutes les commandes s'exécutent depuis la racine du dépôt.

PYTHON := .venv/bin/python

.PHONY: install test lint ingest mlflow-server retrain serve-api demo-incident rollback loadtest monitoring clean

## Installation
install:
	uv venv .venv
	uv pip install --python .venv/bin/python -e ".[dev,loadtest]"
	cp -n .env.example .env

## Qualité
test:
	$(PYTHON) -m pytest tests/

lint:
	$(PYTHON) -m ruff check src tests pipelines

## Données
ingest:
	$(PYTHON) -c "from dropout_pred.data.ingest import ingest; print(ingest())"

## Services (0.0.0.0 : accessibles depuis les conteneurs Prometheus/Grafana)
mlflow-server:
	$(PYTHON) -m mlflow server \
		--backend-store-uri sqlite:///artifacts/mlflow/mlflow.db \
		--default-artifact-root ./artifacts/mlflow/artifacts \
		--host 0.0.0.0 --port 5000

## Pipeline d'entraînement (Prefect)
retrain:
	$(PYTHON) pipelines/retrain.py --batches $(N)

## Déclenchement automatique (scheduler Prefect, hebdomadaire)
serve-pipeline:
	$(PYTHON) pipelines/retrain.py --serve

## API de prédiction (0.0.0.0 : accessible depuis les conteneurs Prometheus/Grafana)
serve-api:
	$(PYTHON) -m uvicorn dropout_pred.api.main:app --host 0.0.0.0 --port 8000 --workers $(WORKERS)

## Démo d'incident contrôlé (lot de données corrompu)
demo-incident:
	$(PYTHON) scripts/demo_incident.py

## Retour arrière vers la version précédente
rollback:
	$(PYTHON) -m dropout_pred.rollback

## Test de charge (Locust, interface web sur http://localhost:8089)
loadtest:
	$(PYTHON) -m locust -f load_tests/locustfile.py --host http://127.0.0.1:8000

## Test de charge sans interface (protocole de dimensionnement)
loadtest-headless:
	$(PYTHON) -m locust -f load_tests/locustfile.py --host http://127.0.0.1:8000 \
		--headless -u $(USERS) -r $(SPAWN_RATE) -t $(DURATION) \
		--csv load_tests/results/resultats

## Supervision (Prometheus + Grafana)
monitoring:
	docker compose -f monitoring/docker-compose.yml up -d

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
