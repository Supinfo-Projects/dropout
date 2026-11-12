"""Pipeline de réentraînement orchestré par Prefect.

Étapes : ingestion -> découpage en lots -> chargement cumulé -> contrôles qualité ->
prétraitement -> entraînement -> évaluation -> décision de mise en service.

Mécanismes opérationnels (cf. cours, chapitre 5) :
- retries sur l'ingestion (la source UCI peut être momentanément indisponible) ;
- contrôles qualité bloquants : une DataValidationError interrompt la pipeline ;
- alerte systématique en cas d'échec (fichier de log + mail si SMTP configuré) ;
- cache du prétraitement tant que les entrées n'ont pas changé ;
- traçabilité complète dans MLflow (paramètres, données, métriques, artefacts).
"""

import argparse
from datetime import timedelta

from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash

from dropout_pred.alerting import send_alert
from dropout_pred.config import get_settings
from dropout_pred.data.batches import build_batches, load_cumulative_batches
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
from dropout_pred.registry import (
    STAGE_RELEASE_CANDIDATE,
    get_production_version,
    promote_to_production,
    tag_version,
)
from dropout_pred.training.train import log_training_to_mlflow, train_model


@task(retries=5, retry_delay_seconds=10, description="Télécharge et normalise le dataset UCI")
def t_ingest() -> dict:
    return ingest()


@task(description="Découpe le dataset historique en lots simulés")
def t_build_batches() -> int:
    paths = build_batches()
    return len(paths)


@task(description="Charge les lots cumulés (simulation de nouvelles données)")
def t_load_batches(n_batches: int):
    return load_cumulative_batches(n_batches)


@task(description="Contrôles qualité bloquants avant prétraitement")
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
    description="Prétraitement : split stratifié (mis en cache tant que les entrées sont identiques)",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def t_preprocess(frame):
    features, target = split_features_target(frame)
    target_encoded = encode_target(target)
    return train_test_split_stratified(features, target_encoded)


@task(description="Entraînement du modèle")
def t_train(x_train, y_train):
    return train_model(x_train, y_train)


@task(description="Évaluation sur le jeu de test")
def t_evaluate(model, x_test, y_test) -> dict:
    return compute_metrics(model, x_test, y_test)


@task(description="Traçabilité MLflow : paramètres, métriques, artefacts, registre")
def t_log_to_mlflow(model, params: dict, metrics: dict, artifacts: dict) -> str:
    _, _, version_number = log_training_to_mlflow(
        model, params=params, metrics=metrics, artifacts=artifacts
    )
    return version_number


@task(description="Décision de mise en service (promotion ou maintien en release_candidate)")
def t_promote(metrics: dict, version_number: str) -> dict:
    settings = get_settings()
    logger = get_run_logger()

    tag_version(version_number, STAGE_RELEASE_CANDIDATE, settings)

    eligible, reason = meets_production_requirements(metrics, settings)
    if not eligible:
        logger.warning("Modèle non éligible à la production : %s", reason)
        return {"decision": "maintenu_release_candidate", "raison": reason}

    production = get_production_version(settings)
    if production is not None:
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        production_metrics = client.get_run(production.run_id).data.metrics
        current_f1 = production_metrics.get("f1_macro", 0.0)
        if metrics["f1_macro"] < current_f1:
            reason = (
                f"f1_macro={metrics['f1_macro']} < f1_macro production={current_f1:.4f} "
                f"(version {production.version})"
            )
            logger.warning("Modèle moins bon que la production : %s", reason)
            return {"decision": "maintenu_release_candidate", "raison": reason}

    promote_to_production(version_number, settings)
    logger.info("Version %s promue en production", version_number)
    return {"decision": "promu_production", "version": version_number, "raison": reason}


@flow(name="reentrainement_decrochage", log_prints=True)
def retrain_flow(n_batches: int = 1) -> dict:
    """Pipeline complète de réentraînement sur les n premiers lots de données."""
    settings = get_settings()
    logger = get_run_logger()
    try:
        ingestion = t_ingest()
        t_build_batches()
        frame = t_load_batches(n_batches)
        validation = t_validate(frame)
        logger.info("Contrôles qualité :\n%s", validation["summary"])

        x_train, x_test, y_train, y_test = t_preprocess(frame)

        params = {
            "n_batches": n_batches,
            "n_rows": validation["n_rows"],
            "data_checksum": ingestion["checksum"],
            "modele": "logistic_regression",
            "test_size": 0.2,
        }
        model = t_train(x_train, y_train)
        metrics = t_evaluate(model, x_test, y_test)

        figure_path = plot_confusion_matrix(metrics["confusion_matrix"], settings)
        version_number = t_log_to_mlflow(
            model,
            params,
            metrics,
            artifacts={
                "rapport_controles": validation["report_path"],
                "figures": str(figure_path),
            },
        )

        decision = t_promote(metrics, version_number)
        logger.info("Décision de mise en service : %s", decision)
        return {"metrics": metrics, "decision": decision}
    except Exception as exc:  # noqa: BLE001
        send_alert(
            "Échec du pipeline de réentraînement",
            f"Erreur : {type(exc).__name__}: {exc}",
            settings,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de réentraînement (Prefect)")
    parser.add_argument("--batches", type=int, default=1, help="Nombre de lots cumulés à consommer")
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
        result = retrain_flow(n_batches=args.batches)
        print(f"\nRésultat : {result['decision']}")
        print(f"Métriques : {result['metrics']}")


if __name__ == "__main__":
    main()
