"""Entrainement du modèle et tracabilité MLflow."""


import mlflow
from mlflow import MlflowClient
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Settings, get_settings
from ..data.preprocess import RANDOM_STATE


def build_model() -> Pipeline:
    """Construit le pipeline de prétraitement + modèle."""
    return Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="median")),
            ("standardisation", StandardScaler()),
            (
                "modele",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_model(features, target) -> Pipeline:
    """Entraine le pipeline sur les données fournies."""
    model = build_model()
    model.fit(features, target)
    return model


def log_training_to_mlflow(
    model: Pipeline,
    params: dict,
    metrics: dict,
    artifacts: dict | None = None,
    settings: Settings | None = None,
) -> tuple[str, str, str]:
    """Trace l'entrainement dans MLflow et enregistre le modèle dans le registre."""
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        # Seules les métriques scalaires sont loggées. la matrice de confusion
        # est tracée comme artefact (image) par le pipeline.
        scalar_metrics = {
            name: value for name, value in metrics.items() if isinstance(value, (int, float))
        }
        mlflow.log_metrics(scalar_metrics)
        for name, path in (artifacts or {}).items():
            mlflow.log_artifact(str(path), artifact_path=name)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name=settings.mlflow_model_name,
            registered_model_name=settings.mlflow_model_name,
            # Le pipeline référence des types numpy standards
            skops_trusted_types=["numpy.dtype"],
        )

        # L'URI contient un identifiant interne. On récupère le numéro de version
        # créé dans le registre pour ce run.
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        versions = client.search_model_versions(
            f"name='{settings.mlflow_model_name}' and run_id='{run.info.run_id}'"
        )
        version_number = str(max(int(version.version) for version in versions))
        return model_info.model_uri, run.info.run_id, version_number
