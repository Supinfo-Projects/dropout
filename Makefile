PYTHON := .venv/bin/python

.PHONY: install test lint ingest mlflow-server retrain serve-api demo-incident rollback loadtest monitoring clean

install:
	uv venv .venv
	uv pip install --python .venv/bin/python -r requirements.txt
	uv pip install --python .venv/bin/python -e . --no-deps
	cp -n .env.example .env

test:
	$(PYTHON) -m pytest tests/

lint:
	$(PYTHON) -m ruff check src tests pipelines

ingest:
	$(PYTHON) -c "from dropout_pred.data.ingest import ingest; print(ingest())"

mlflow-server:
	$(PYTHON) -m mlflow server \
		--backend-store-uri sqlite:///artifacts/mlflow/mlflow.db \
		--default-artifact-root ./artifacts/mlflow/artifacts \
		--host 0.0.0.0 --port 5000

retrain:
	$(PYTHON) pipelines/retrain.py --batches $(N)

serve-pipeline:
	$(PYTHON) pipelines/retrain.py --serve

serve-api:
	mkdir -p .prometheus_multiproc
	rm -f .prometheus_multiproc/*
	PROMETHEUS_MULTIPROC_DIR=.prometheus_multiproc $(PYTHON) -m uvicorn dropout_pred.api.main:app --host 0.0.0.0 --port 8000 --workers $(WORKERS)

demo-incident:
	$(PYTHON) scripts/demo_incident.py

rollback:
	$(PYTHON) -m dropout_pred.rollback

loadtest:
	$(PYTHON) -m locust -f load_tests/locustfile.py --host http://127.0.0.1:8000

loadtest-headless:
	mkdir -p load_tests/results
	$(PYTHON) -m locust -f load_tests/locustfile.py --host http://127.0.0.1:8000 \
		--headless -u $(USERS) -r $(SPAWN_RATE) -t $(DURATION) \
		--csv load_tests/results/resultats

monitoring:
	docker compose -f monitoring/docker-compose.yml up -d

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
