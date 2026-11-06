"""Ingestion des données"""

import hashlib
import zipfile
from pathlib import Path

import pandas as pd

from ..config import Settings, get_settings
from .schema import FEATURE_COLUMNS, TARGET_COLUMN


def download_dataset(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "dataset.zip"

    import urllib.request

    urllib.request.urlretrieve(settings.data_url, zip_path)
    return zip_path


def extract_csv(zip_path: Path, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "data.csv"

    with zipfile.ZipFile(zip_path) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(member) as source, open(csv_path, "wb") as target:
            target.write(source.read())

    frame = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    frame = normalize_columns(frame)
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [str(col).strip().lstrip("\ufeff").rstrip("\t").strip() for col in renamed.columns]
    return renamed


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_frame(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    frame = normalize_columns(frame)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in frame.columns]
    if missing:
        raise ValueError(f"Colonnes absentes du dataset: {missing}")
    return frame


def ingest(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    zip_path = download_dataset(settings)
    csv_path = extract_csv(zip_path, settings)
    frame = load_raw_frame(csv_path)
    return {
        "zip_path": str(zip_path),
        "csv_path": str(csv_path),
        "checksum": file_checksum(csv_path),
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
    }
