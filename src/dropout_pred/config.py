"""Configuration centralisée du projet, chargée depuis l'environnement / le fichier .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres applicatifs. Chaque valeur peut être surchargée par une variable d'environnement."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # MLflow
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment: str = "entrainement_decrochage"
    mlflow_model_name: str = "dropout_classifier"

    # Règles de mise en service
    min_f1_macro: float = 0.45
    min_recall_dropout: float = 0.60

    # Cache du modèle servi
    model_cache_ttl: int = 300

    # Chemins
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    # Source de données
    data_url: str = (
        "https://archive.ics.uci.edu/static/public/697/"
        "predict+students+dropout+and+academic+success.zip"
    )

    # Alertes
    alert_email: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""


def get_settings() -> Settings:
    """Retourne la configuration."""
    return Settings()
