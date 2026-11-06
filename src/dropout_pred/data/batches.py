"""Découpage du dataset historique en lots, pour simuler l'arrivée de nouvelles données.

Le dataset UCI est un instantané historique : pour démontrer une chaîne opérationnelle, il est
rejoué par lots (cf. sujet du projet). Chaque exécution du pipeline consomme un lot
supplémentaire, cumulé avec les précédents, comme si de nouveaux étudiants étaient
enregistrés chaque semaine.
"""

from pathlib import Path

import pandas as pd

from ..config import Settings, get_settings
from .ingest import load_raw_frame, normalize_columns
from .schema import TARGET_COLUMN

N_BATCHES = 5


def build_batches(settings: Settings | None = None) -> list[Path]:
    """Découpe le CSV brut en N lots de taille égale dans data/batches/."""
    settings = settings or get_settings()
    batches_dir = settings.data_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = settings.data_dir / "raw" / "data.csv"
    frame = load_raw_frame(raw_csv)

    paths = []
    batch_size = len(frame) // N_BATCHES
    for index in range(N_BATCHES):
        start = index * batch_size
        end = start + batch_size if index < N_BATCHES - 1 else len(frame)
        batch_path = batches_dir / f"lot_{index + 1:03d}.csv"
        frame.iloc[start:end].to_csv(batch_path, index=False, encoding="utf-8")
        paths.append(batch_path)
    return paths


def load_cumulative_batches(n_batches: int, settings: Settings | None = None) -> pd.DataFrame:
    """Charge les n premiers lots, comme si les données arrivaient progressivement."""
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
        raise ValueError(f"Colonne cible absente des lots chargés : {TARGET_COLUMN}")
    return cumulative
