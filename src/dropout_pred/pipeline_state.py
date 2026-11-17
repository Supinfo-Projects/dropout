"""Supervision des traitements Prefect.

La flow Prefect écrit son état (démarrage, succès ou échec, durée, version produite)
dans artifacts/pipeline/state.json. L'API expose ce fichier de deux facons:
- GET /pipeline/status = le JSON brut
- /pipeline/metrics = métriques Prometheus (horodatage du dernier succès, durée,
  indicateur de succès), montées sur un registre dédié pour rester lisibles même
  quand les compteurs de l'API sont agrégés en mode multiprocess.

Le dashboard Grafana consomme ces métriques. La supervision couvre ainsi les
traitements, et pas seulement le service.
"""

import json
import time
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, make_asgi_app

from .config import get_settings

STATE_FILENAME = "state.json"


def state_path() -> Path:
    return get_settings().artifacts_dir / "pipeline" / STATE_FILENAME


def write_pipeline_state(
    state: str,
    started_at: float | None = None,
    duration_s: float | None = None,
    version: str | None = None,
    n_batches: int | None = None,
    error: str | None = None,
) -> Path:
    """Ecrit l'état du traitement courant."""
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "updated_at": time.time(),
        "started_at": started_at,
        "duration_s": duration_s,
        "version": version,
        "n_batches": n_batches,
        "error": error,
    }
    previous = read_pipeline_state()
    if previous and state == "running":
        # On conserve le dernier résultat connu pendant l'exécution suivante.
        payload["last_success"] = previous.get("last_success")
    elif state == "succes":
        payload["last_success"] = {
            "version": version,
            "duration_s": duration_s,
            "finished_at": time.time(),
        }
    elif previous:
        payload["last_success"] = previous.get("last_success")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_pipeline_state() -> dict | None:
    """Lit l'état du dernier traitement. None si aucun n'a été enregistré."""
    path = state_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _last_success_timestamp() -> float:
    state = read_pipeline_state() or {}
    last = state.get("last_success") or {}
    return float(last.get("finished_at") or 0)


def _last_success_duration() -> float:
    state = read_pipeline_state() or {}
    last = state.get("last_success") or {}
    return float(last.get("duration_s") or 0)


def _last_run_success() -> float:
    state = read_pipeline_state() or {}
    return 1.0 if state.get("state") == "succes" else 0.0


_pipeline_registry = CollectorRegistry()
PIPELINE_LAST_SUCCESS_TIMESTAMP = Gauge(
    "dropout_pipeline_last_success_timestamp_seconds",
    "Horodatage du dernier réentraînement réussi",
    registry=_pipeline_registry,
)
PIPELINE_LAST_SUCCESS_DURATION = Gauge(
    "dropout_pipeline_last_success_duration_seconds",
    "Durée du dernier réentraînement réussi",
    registry=_pipeline_registry,
)
PIPELINE_LAST_RUN_SUCCESS = Gauge(
    "dropout_pipeline_last_run_success",
    "1 si le dernier traitement a réussi, 0 sinon",
    registry=_pipeline_registry,
)
PIPELINE_LAST_SUCCESS_TIMESTAMP.set_function(_last_success_timestamp)
PIPELINE_LAST_SUCCESS_DURATION.set_function(_last_success_duration)
PIPELINE_LAST_RUN_SUCCESS.set_function(_last_run_success)


def pipeline_metrics_app():
    """Application ASGI exposant les métriques des traitements Prefect."""
    return make_asgi_app(registry=_pipeline_registry)
