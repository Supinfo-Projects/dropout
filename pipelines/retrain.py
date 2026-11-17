"""Pipeline de réentrainement orchestré par Prefect."""

import argparse
import time
from datetime import timedelta

from prefect import flow, get_run_logger, task
from prefect.runtime import flow_run as prefect_flow_run
from prefect.tasks import task_input_hash

from dropout_pred.alerting import send_alert
from dropout_pred.config import get_settings
from dropout_pred.data.batches import build_batches, load_cumulative_batches, load_validation_set
from dropout_pred.data.ingest import ingest
from dropout_pred.data.preprocess import (
    encode_target,
    split_features_target,
    train_test_split_stratified,
)
from dropout_pred.data.validate import validate_or_raise
from dropout_pred.evaluation.evaluate import (
    compute_metrics,
    meets_production_requirements,
    plot_confusion_matrix,
)
from dropout_pred.pipeline_state import write_pipeline_state
from dropout_pred.registry import (
    STAGE_RELEASE_CANDIDATE,
    get_production_version,
    load_production_model,
    promote_to_production,
    tag_version,
)
from dropout_pred.training.train import log_training_to_mlflow, train_model


@task(retries=5, retry_delay_seconds=10, description="Télécharge et normalise le dataset UCI")
def t_ingest() -> dict:
    return ingest()


@task(description="Découpe le dataset historique en lots + jeu de validation commun")
def t_build_batches() -> int:
    paths = build_batches()
    return len(paths)


@task(description="Charge les lots cumulés (simulation de nouvelles données)")
def t_load_batches(n_batches: int):
    return load_cumulative_batches(n_batches)


@task(description="Charge le jeu de validation commun (exclu de tout entrainement)")
def t_load_validation():
    return load_validation_set()


@task(description="Controles qualité bloquants avant prétraitement")
def t_validate(frame) -> dict:
    report, report_path = validate_or_raise(frame)
    return {
        "passed": report.passed,
        "n_rows": report.n_rows,
        "n_failed_rules": report.n_failed_rules,
        "summary": report.summary(),
        "report_path": str(report_path),
    }


@task(
    description="Prétraitement: split stratifié (mis en cache tant que les entrées sont identiques)",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def t_preprocess(frame):
    features, target = split_features_target(frame)
    target_encoded = encode_target(target)
    return train_test_split_stratified(features, target_encoded)


@task(description="Préparation du jeu de validation commun (features + cible encodée)")
def t_preprocess_validation(validation_frame):
    features, target = split_features_target(validation_frame)
    return features, encode_target(target)


@task(description="Entraînement du modèle")
def t_train(x_train, y_train):
    return train_model(x_train, y_train)


@task(description="Évaluation sur le jeu de test")
def t_evaluate(model, x_test, y_test) -> dict:
    return compute_metrics(model, x_test, y_test)


@task(description="Traçabilité MLflow: paramètres, métriques, artefacts, registre")
def t_log_to_mlflow(model, params: dict, metrics: dict, artifacts: dict) -> tuple[str, str]:
    _, run_id, version_number = log_training_to_mlflow(
        model, params=params, metrics=metrics, artifacts=artifacts
    )
    return run_id, version_number


def _corrupt_frame(frame):
    """Simule un bug de collecte en amont: texte dans une colonne numérique,
    valeurs hors domaine binaire et classes cibles écrasées."""
    frame = frame.copy()
    frame["Admission grade"] = "invalide"
    frame["Gender"] = 9
    frame["Target"] = "Graduate"
    return frame


@task(description="Décision de mise en service (promotion ou maintien en release_candidate)")
def t_promote(metrics: dict, validation_metrics: dict, version_number: str, run_id: str, x_val, y_val) -> dict:
    """Compare le candidat à la production sur le jeu de validation commun.

    Les deux modèles sont évalués sur le même jeu de validation, exclu de tous
    les entrainements. La décision, ses seuils, la version comparée et l'exécution
    Prefect sont journalisés dans le run MLflow du candidat.
    """
    settings = get_settings()
    logger = get_run_logger()

    tag_version(version_number, STAGE_RELEASE_CANDIDATE, settings)

    eligible, reason = meets_production_requirements(metrics, settings)
    decision = {
        "decision": "maintenu_release_candidate",
        "raison": reason,
        "seuils": {"min_f1_macro": settings.min_f1_macro, "min_recall_dropout": settings.min_recall_dropout},
        "f1_macro_validation": validation_metrics["f1_macro"],
        "run_prefect": prefect_flow_run.id,
    }
    if not eligible:
        logger.warning("Modèle non éligible à la production: %s", reason)
        _log_decision_to_mlflow(run_id, decision, settings)
        return decision

    production = get_production_version(settings)
    if production is not None:
        production_model, _ = load_production_model(settings)
        production_validation_metrics = compute_metrics(production_model, x_val, y_val)
        decision["version_comparée"] = str(production.version)
        decision["f1_macro_validation_production"] = production_validation_metrics["f1_macro"]
        if validation_metrics["f1_macro"] < production_validation_metrics["f1_macro"]:
            reason = (
                f"f1_macro validation={validation_metrics['f1_macro']} < "
                f"f1_macro validation production={production_validation_metrics['f1_macro']} "
                f"(version {production.version}, même jeu de validation)"
            )
            decision["raison"] = reason
            logger.warning("Modèle moins bon que la production sur le jeu de validation: %s", reason)
            _log_decision_to_mlflow(run_id, decision, settings)
            return decision

    promote_to_production(version_number, settings)
    decision["decision"] = "promu_production"
    decision["version"] = version_number
    logger.info("Version %s promue en production", version_number)
    _log_decision_to_mlflow(run_id, decision, settings)
    return decision


def _log_decision_to_mlflow(run_id: str, decision: dict, settings) -> None:
    """Journalise la décision de mise en service dans le run MLflow du candidat."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    for key, value in decision.items():
        if value is None:
            continue
        client.log_param(run_id, f"decision_{key}", str(value))
    client.set_tag(run_id, "decision", decision["decision"])
    client.set_tag(run_id, "run_prefect", decision.get("run_prefect", ""))


@flow(name="reentrainement_decrochage", log_prints=True)
def retrain_flow(n_batches: int = 1, corrupt_data: bool = False) -> dict:
    """Pipeline complète de réentrainement sur les n premiers lots de données.

    corrupt_data=True simule un lot corrompu après la construction des lots (point
    d'injection utilisé par la démo d'incident): les controles qualité doivent alors
    interrompre la flow avant l'entrainement.
    """
    settings = get_settings()
    logger = get_run_logger()
    started_at = time.time()
    write_pipeline_state("running", started_at=started_at, n_batches=n_batches)
    try:
        ingestion = t_ingest()
        t_build_batches()
        frame = t_load_batches(n_batches)
        if corrupt_data:
            logger.warning("Injection d'un lot corrompu (point d'entrée de la démo d'incident)")
            frame = _corrupt_frame(frame)
        validation = t_validate(frame)
        logger.info("Controles qualité:\n%s", validation["summary"])

        x_train, x_test, y_train, y_test = t_preprocess(frame)
        validation_frame = t_load_validation()
        x_val, y_val = t_preprocess_validation(validation_frame)

        params = {
            "n_batches": n_batches,
            "n_rows": validation["n_rows"],
            "data_checksum": ingestion["checksum"],
            "modele": "logistic_regression",
            "test_size": 0.2,
        }
        model = t_train(x_train, y_train)
        metrics = t_evaluate(model, x_test, y_test)
        validation_metrics = t_evaluate(model, x_val, y_val)
        metrics["f1_macro_validation"] = validation_metrics["f1_macro"]

        figure_path = plot_confusion_matrix(metrics["confusion_matrix"], settings)
        run_id, version_number = t_log_to_mlflow(
            model,
            params,
            metrics,
            artifacts={
                "rapport_controles": validation["report_path"],
                "figures": str(figure_path),
            },
        )

        decision = t_promote(metrics, validation_metrics, version_number, run_id, x_val, y_val)
        logger.info("Décision de mise en service: %s", decision)
        write_pipeline_state(
            "succes",
            started_at=started_at,
            duration_s=round(time.time() - started_at, 1),
            version=version_number,
            n_batches=n_batches,
        )
        return {"metrics": metrics, "decision": decision}
    except Exception as exc:  # noqa: BLE001
        send_alert(
            "Echec du pipeline de réentrainement",
            f"Erreur: {type(exc).__name__}: {exc}",
            settings,
        )
        write_pipeline_state(
            "echec",
            started_at=started_at,
            duration_s=round(time.time() - started_at, 1),
            n_batches=n_batches,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de réentrainement (Prefect)")
    parser.add_argument("--batches", type=int, default=1, help="Nombre de lots cumulés à consommer")
    parser.add_argument(
        "--corrupt-data",
        action="store_true",
        help="Injecte un lot corrompu après la construction des lots (démo d'incident)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Démarre le scheduler Prefect (déclenchement automatique hebdomadaire)",
    )
    args = parser.parse_args()

    if args.serve:
        retrain_flow.serve(
            name="reentrainement_hebdomadaire",
            cron="0 6 * * 1",
            parameters={"n_batches": 5},
        )
    else:
        result = retrain_flow(n_batches=args.batches, corrupt_data=args.corrupt_data)
        print(f"\nRésultat: {result['decision']}")
        print(f"Métriques: {result['metrics']}")


if __name__ == "__main__":
    main()
