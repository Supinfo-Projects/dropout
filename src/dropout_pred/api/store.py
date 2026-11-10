"""Persistance des traces d'utilisation"""

import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings, get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage TEXT NOT NULL,
    model_version TEXT NOT NULL,
    entree TEXT NOT NULL,
    classe_predite TEXT,
    probabilites TEXT,
    latence_ms REAL,
    statut TEXT NOT NULL,
    erreur TEXT
);
CREATE TABLE IF NOT EXISTS feedbacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horodatage TEXT NOT NULL,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id),
    classe_reelle TEXT NOT NULL,
    commentaire TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_statut ON predictions(statut);
"""


class PredictionStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    def save_prediction(
        self,
        model_version: str,
        entree: dict,
        classe_predite: str | None,
        probabilites: dict | None,
        latence_ms: float,
        statut: str,
        erreur: str | None = None,
    ) -> int:
        with self._lock, closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions
                    (horodatage, model_version, entree, classe_predite, probabilites,
                     latence_ms, statut, erreur)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    model_version,
                    json.dumps(entree, ensure_ascii=False),
                    classe_predite,
                    json.dumps(probabilites, ensure_ascii=False) if probabilites else None,
                    latence_ms,
                    statut,
                    erreur,
                ),
            )
            connection.commit()
            return cursor.lastrowid

    def get_prediction(self, prediction_id: int) -> dict | None:
        with self._lock, closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
            return dict(row) if row else None

    def save_feedback(self, prediction_id: int, classe_reelle: str, commentaire: str | None = None) -> int:
        with self._lock, closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedbacks (horodatage, prediction_id, classe_reelle, commentaire)
                VALUES (?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    prediction_id,
                    classe_reelle,
                    commentaire,
                ),
            )
            connection.commit()
            return cursor.lastrowid

    def stats(self) -> dict:
        with self._lock, closing(self._connect()) as connection:
            total = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            errors = connection.execute(
                "SELECT COUNT(*) FROM predictions WHERE statut = 'erreur'"
            ).fetchone()[0]
            feedbacks = connection.execute("SELECT COUNT(*) FROM feedbacks").fetchone()[0]
            distribution = dict(
                connection.execute(
                    "SELECT classe_predite, COUNT(*) FROM predictions "
                    "WHERE statut = 'succes' GROUP BY classe_predite"
                ).fetchall()
            )
            classes_reelles = dict(
                connection.execute(
                    "SELECT classe_reelle, COUNT(*) FROM feedbacks GROUP BY classe_reelle"
                ).fetchall()
            )
        return {
            "total_predictions": total,
            "total_feedbacks": feedbacks,
            "distribution_predictions": distribution,
            "distribution_classes_reelles": classes_reelles,
            "taux_erreur": round(errors / total, 4) if total else 0.0,
        }


def get_store(settings: Settings | None = None) -> PredictionStore:
    settings = settings or get_settings()
    return PredictionStore(settings.artifacts_dir / "api" / "predictions.db")
