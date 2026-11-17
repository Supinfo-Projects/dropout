"""Découpage du dataset historique en lots, pour simuler l'arrivée de nouvelles données."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import Settings, get_settings
from .ingest import load_raw_frame, normalize_columns
from .schema import TARGET_COLUMN

N_BATCHES = 5
VALIDATION_SHARE = 0.1  # part du dataset réservée au jeu de validation commun
RANDOM_STATE = 42


def build_batches(settings: Settings | None = None) -> list[Path]:
    settings = settings or get_settings()
    batches_dir = settings.data_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = settings.data_dir / "raw" / "data.csv"
    frame = normalize_columns(load_raw_frame(raw_csv))

    train_frame, validation_frame = train_test_split(
        frame,
        test_size=VALIDATION_SHARE,
        stratify=frame[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    validation_path = batches_dir / "validation.csv"
    validation_frame.to_csv(validation_path, index=False, encoding="utf-8")

    paths = []
    batch_size = len(train_frame) // N_BATCHES
    for index in range(N_BATCHES):
        start = index * batch_size
        end = start + batch_size if index < N_BATCHES - 1 else len(train_frame)
        batch_path = batches_dir / f"lot_{index + 1:03d}.csv"
        train_frame.iloc[start:end].to_csv(batch_path, index=False, encoding="utf-8")
        paths.append(batch_path)
    return paths


def load_validation_set(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.data_dir / "batches" / "validation.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Jeu de validation introuvable: {path}. Lancez d'abord le pipeline (make retrain N=1)."
        )
    return normalize_columns(pd.read_csv(path, sep=",", encoding="utf-8-sig"))


def load_cumulative_batches(n_batches: int, settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    batches_dir = settings.data_dir / "batches"
    frames = []
    for index in range(1, n_batches + 1):
        path = batches_dir / f"lot_{index:03d}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Lot introuvable : {path}")
        frames.append(normalize_columns(pd.read_csv(path, sep=",", encoding="utf-8-sig")))
    cumulative = pd.concat(frames, ignore_index=True)
    if TARGET_COLUMN not in cumulative.columns:
        raise ValueError(f"Colonne cible absente des lots chargés: {TARGET_COLUMN}")
    return cumulative
