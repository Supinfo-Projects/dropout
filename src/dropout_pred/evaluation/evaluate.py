"""Évaluation du modèle."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from ..config import Settings, get_settings  # noqa: E402
from ..data.schema import TARGET_CLASSES  # noqa: E402


def compute_metrics(model, x_test, y_test) -> dict:
    """Calcule les métriques clés sur le jeu de test."""
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    f1_macro = float(f1_score(y_test, predictions, average="macro"))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average=None, labels=range(len(TARGET_CLASSES)), zero_division=0
    )
    return {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "recall_dropout": round(float(recall[0]), 4),
        "precision_dropout": round(float(precision[0]), 4),
        "recall_enrolled": round(float(recall[1]), 4),
        "recall_graduate": round(float(recall[2]), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=range(len(TARGET_CLASSES))).tolist(),
    }


def plot_confusion_matrix(matrix: list, settings: Settings | None = None) -> object:
    """Génère la matrice de confusion en PNG."""

    settings = settings or get_settings()
    figures_dir = settings.artifacts_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "confusion_matrix.png"

    array = np.array(matrix)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(array, cmap="Blues")
    figure.colorbar(image)
    axis.set_xticks(range(len(TARGET_CLASSES)), TARGET_CLASSES, rotation=30)
    axis.set_yticks(range(len(TARGET_CLASSES)), TARGET_CLASSES)
    axis.set_xlabel("Prédiction")
    axis.set_ylabel("Classe réelle")
    axis.set_title("Matrice de confusion (jeu de test)")
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            axis.text(col, row, str(array[row, col]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def meets_production_requirements(metrics: dict, settings: Settings | None = None) -> tuple[bool, str]:
    """Applique la règle de mise en service. Le modèle doit dépasser les seuils configurés.

    Retourne: la justification est tracée dans MLflow.
    """
    settings = settings or get_settings()
    reasons = []
    if metrics["f1_macro"] < settings.min_f1_macro:
        reasons.append(
            f"f1_macro={metrics['f1_macro']} < seuil={settings.min_f1_macro}"
        )
    if metrics["recall_dropout"] < settings.min_recall_dropout:
        reasons.append(
            f"recall_dropout={metrics['recall_dropout']} < seuil={settings.min_recall_dropout}"
        )
    if reasons:
        return False, " ; ".join(reasons)
    return True, (
        f"f1_macro={metrics['f1_macro']} >= {settings.min_f1_macro} et "
        f"recall_dropout={metrics['recall_dropout']} >= {settings.min_recall_dropout}"
    )
