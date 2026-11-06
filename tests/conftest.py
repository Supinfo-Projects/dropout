import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dropout_pred.data.schema import CONTINUOUS_RANGES, DISCRETE_NON_NEGATIVE, TARGET_CLASSES  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MODEL_CACHE_TTL", "300")
    yield


def make_frame(n_rows: int = 500, target_cycle: bool = True) -> pd.DataFrame:
    data = {}
    for column in DISCRETE_NON_NEGATIVE:
        data[column] = [1] * n_rows
    # On varier Application order pour avoir des lignes uniques
    data["Application order"] = [i % 6 for i in range(n_rows)]
    for column, (low, high) in CONTINUOUS_RANGES.items():
        middle = (low + high) / 2
        data[column] = [middle] * n_rows
    data["Admission grade"] = [100.0 + i * 0.1 for i in range(n_rows)]
    if target_cycle:
        data["Target"] = [TARGET_CLASSES[i % 3] for i in range(n_rows)]
    return pd.DataFrame(data)


@pytest.fixture
def frame_factory():
    return make_frame


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return make_frame(600)
