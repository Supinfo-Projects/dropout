"""Tests unitaires des contrôles qualité : chaque règle doit bloquer quand elle doit."""

import json

import pytest

from dropout_pred.data.schema import TARGET_COLUMN
from dropout_pred.data.validate import DataValidationError, validate_frame, validate_or_raise


def test_donnees_conformes_passent(sample_frame):
    report = validate_frame(sample_frame)
    assert report.passed
    assert report.n_failed_rules == 0
    assert report.n_rows == len(sample_frame)


def test_volume_insuffisant_bloque(frame_factory):
    report = validate_frame(frame_factory(100))
    assert not report.passed
    assert any(result.name == "volume" for result in report.failures())


def test_valeurs_manquantes_bloquent(sample_frame):
    frame = sample_frame.copy()
    frame.loc[0:10, "Admission grade"] = None
    report = validate_frame(frame)
    assert not report.passed
    assert any(result.name == "valeurs_manquantes" for result in report.failures())


def test_valeurs_hors_plage_bloquent(sample_frame):
    frame = sample_frame.copy()
    frame["Admission grade"] = 999.0
    report = validate_frame(frame)
    assert not report.passed
    failures = {result.name for result in report.failures()}
    assert "plages_de_valeurs" in failures


def test_cible_inconnue_bloque(sample_frame):
    frame = sample_frame.copy()
    frame.loc[0, TARGET_COLUMN] = "Inconnu"
    report = validate_frame(frame)
    assert not report.passed
    assert any(result.name == "cible" for result in report.failures())


def test_classe_sous_representee_bloque(sample_frame):
    frame = sample_frame.copy()
    frame.loc[frame[TARGET_COLUMN] == "Enrolled", TARGET_COLUMN] = "Graduate"
    report = validate_frame(frame)
    assert not report.passed
    assert any(result.name == "representativite_classes" for result in report.failures())


def test_colonne_absente_bloque(sample_frame):
    frame = sample_frame.drop(columns=["Admission grade"])
    report = validate_frame(frame)
    assert not report.passed
    assert any(result.name == "schema" for result in report.failures())


def test_validate_or_raise_interrompt_et_sauvegarde_le_rapport(sample_frame):
    frame = sample_frame.copy()
    frame["Admission grade"] = 999.0
    with pytest.raises(DataValidationError) as excinfo:
        validate_or_raise(frame)
    assert "Contrôles qualité en échec" in str(excinfo.value)

    from dropout_pred.config import get_settings

    reports_dir = get_settings().artifacts_dir / "validation"
    reports = sorted(reports_dir.glob("validation_*.json"))
    assert reports, "Le rapport de validation devait être sauvegardé"
    content = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert content["passed"] is False
    assert content["n_failed_rules"] >= 1


def test_validate_or_raise_passe_sans_exception(sample_frame):
    report, path = validate_or_raise(sample_frame)
    assert report.passed
    assert path.exists()
