import pytest

from dropout_pred.data.preprocess import (
    encode_target,
    split_features_target,
    train_test_split_stratified,
)
from dropout_pred.data.schema import FEATURE_COLUMNS, TARGET_COLUMN


def test_separation_features_cible(sample_frame):
    features, target = split_features_target(sample_frame)
    assert list(features.columns) == FEATURE_COLUMNS
    assert target.name == TARGET_COLUMN


def test_encodage_de_la_cible(sample_frame):
    _, target = split_features_target(sample_frame)
    encoded = encode_target(target)
    assert set(encoded.unique()) == {0, 1, 2}


def test_encodage_refuse_les_classes_inconnues(sample_frame):
    frame = sample_frame.copy()
    frame.loc[0, TARGET_COLUMN] = "Absent"
    _, target = split_features_target(frame)
    with pytest.raises(ValueError, match="Classes cibles inconnues"):
        encode_target(target)


def test_decoupage_stratifie(sample_frame):
    features, target = split_features_target(sample_frame)
    encoded = encode_target(target)
    x_train, x_test, y_train, y_test = train_test_split_stratified(features, encoded)
    assert len(x_train) + len(x_test) == len(sample_frame)
    assert len(x_test) == int(len(sample_frame) * 0.2)
    # Les proportions de classes sont conservées à +/- 2 points
    for index, share in enumerate(y_train.value_counts(normalize=True).sort_index()):
        expected = (encoded == index).mean()
        assert abs(share - expected) < 0.02


def test_colonne_inconnue_ignoree(sample_frame):
    frame = sample_frame.copy()
    frame["Colonne inconnue"] = 1
    features, _ = split_features_target(frame)
    assert "Colonne inconnue" not in features.columns
