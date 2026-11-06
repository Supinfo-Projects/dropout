import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import FEATURE_COLUMNS, TARGET_CLASSES, TARGET_COLUMN

TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = frame.reindex(columns=FEATURE_COLUMNS)
    target = frame[TARGET_COLUMN].astype(str)
    return features, target


def encode_target(target: pd.Series) -> pd.Series:
    mapping = {label: index for index, label in enumerate(TARGET_CLASSES)}
    unknown = set(target.unique()) - set(mapping)
    if unknown:
        raise ValueError(f"Classes cibles inconnues: {sorted(unknown)}")
    return target.map(mapping)


def train_test_split_stratified(features: pd.DataFrame, target_encoded: pd.Series) -> tuple:
    return train_test_split(
        features,
        target_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target_encoded,
    )
