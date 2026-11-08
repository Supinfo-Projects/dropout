from dropout_pred.data.preprocess import (
    encode_target,
    split_features_target,
    train_test_split_stratified,
)
from dropout_pred.evaluation.evaluate import (
    compute_metrics,
    meets_production_requirements,
    plot_confusion_matrix,
)
from dropout_pred.training.train import build_model, train_model


def _fit_model(frame):
    features, target = split_features_target(frame)
    encoded = encode_target(target)
    x_train, _, y_train, _ = train_test_split_stratified(features, encoded)
    return train_model(x_train, y_train)


def test_le_pipeline_contient_pretraitement_et_modele():
    model = build_model()
    steps = dict(model.named_steps)
    assert set(steps) == {"imputation", "standardisation", "modele"}


def test_entrainement_et_metrics(sample_frame):
    model = _fit_model(sample_frame)
    features, target = split_features_target(sample_frame)
    encoded = encode_target(target)
    _, x_test, _, y_test = train_test_split_stratified(features, encoded)
    metrics = compute_metrics(model, x_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 3


def test_matrice_de_confusion_genere_un_fichier(sample_frame):
    model = _fit_model(sample_frame)
    features, target = split_features_target(sample_frame)
    encoded = encode_target(target)
    _, x_test, _, y_test = train_test_split_stratified(features, encoded)
    metrics = compute_metrics(model, x_test, y_test)
    path = plot_confusion_matrix(metrics["confusion_matrix"])
    assert path.exists()
    assert path.suffix == ".png"


def test_regle_de_mise_en_service():
    bonnes_metrics = {"f1_macro": 0.60, "recall_dropout": 0.70}
    eligible, _ = meets_production_requirements(bonnes_metrics)
    assert eligible

    f1_faible = {"f1_macro": 0.30, "recall_dropout": 0.70}
    eligible, raison = meets_production_requirements(f1_faible)
    assert not eligible
    assert "f1_macro" in raison

    recall_faible = {"f1_macro": 0.60, "recall_dropout": 0.40}
    eligible, raison = meets_production_requirements(recall_faible)
    assert not eligible
    assert "recall_dropout" in raison
