import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.retrain import retrain_flow

from dropout_pred.config import get_settings
from dropout_pred.data.schema import FEATURE_COLUMNS, TARGET_CLASSES
from dropout_pred.pipeline_state import read_pipeline_state
from dropout_pred.registry import get_production_version, get_versions, load_production_model

ETUDIANT_PATH = Path(__file__).resolve().parents[1] / "load_tests" / "exemple_etudiant.json"


def _predire_production() -> dict:
    model, version = load_production_model()
    entree = json.loads(ETUDIANT_PATH.read_text(encoding="utf-8"))
    frame = pd.DataFrame([entree]).reindex(columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(frame)[0]
    return {
        "version": str(version.version),
        "classe_predite": TARGET_CLASSES[int(probabilities.argmax())],
    }


def main() -> None:
    settings = get_settings()
    lignes: list[str] = []

    def trace(message: str) -> None:
        print(message)
        lignes.append(message)

    production_avant = get_production_version(settings)
    if production_avant is None:
        raise SystemExit(
            "Aucun modèle en production. Lancez d'abord make retrain N=1"
        )
    versions_avant = {str(v.version) for v in get_versions(settings)}
    prediction_avant = _predire_production()

    trace("=== DOMO D'INCIDENT CONTROLE ===")
    trace(f"Horodatage: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    trace(f"Production avant l'incident: version {production_avant.version}")
    trace(f"Versions enregistrées avant: {sorted(versions_avant, key=int)}")
    trace(
        f"Prédiction de production avant: version {prediction_avant['version']}, "
        f"classe {prediction_avant['classe_predite']}"
    )
    trace("")
    trace("Injection d'un lot corrompu et exécution de la flow Prefect...")
    trace("")

    etat = retrain_flow(n_batches=5, corrupt_data=True, return_state=True)
    trace(f"État final de la flow Prefect: {etat.name} (échec: {etat.is_failed()})")

    production_apres = get_production_version(settings)
    versions_apres = {str(v.version) for v in get_versions(settings)}
    prediction_apres = _predire_production()
    etat_pipeline = read_pipeline_state() or {}

    trace("")
    trace("=== VERIFICATIONS ===")
    trace(
        f"1. Aucune nouvelle version créée: "
        f"{'OK' if versions_apres == versions_avant else 'ECHEC'} "
        f"({len(versions_apres)} versions avant et après)"
    )
    trace(
        f"2. Production inchangée: "
        f"{'OK' if str(production_apres.version) == str(production_avant.version) else 'ECHEC'} "
        f"(version {production_apres.version})"
    )
    trace(
        f"3. La production sert toujours: version {prediction_apres['version']}, "
        f"classe {prediction_apres['classe_predite']} "
        f"{'(identique à avant)' if prediction_apres == prediction_avant else '(DIFFERENTE DE AVANT !!!)'}"
    )
    trace(
        f"4. Alerte enregistrée: "
        f"{'OK' if (settings.artifacts_dir / 'alerts' / 'alerts.log').exists() else 'ECHEC'} "
        f"(artifacts/alerts/alerts.log)"
    )
    trace(f"5. État du traitement exposé à la supervision: {etat_pipeline.get('state')}")

    incident_ok = (
        versions_apres == versions_avant
        and str(production_apres.version) == str(production_avant.version)
        and prediction_apres == prediction_avant
        and etat.is_failed()
    )
    trace("")
    trace(f"RÉSULTAT : {'incident maitrisé' if incident_ok else 'ECHEC DE LA DEMO'}")

    preuve_dir = settings.artifacts_dir / "demo_incident"
    preuve_dir.mkdir(parents=True, exist_ok=True)
    preuve_path = preuve_dir / "preuve_incident.txt"
    preuve_path.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    trace(f"Preuve sauvegardée: {preuve_path}")

    if not incident_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
