import pandas as pd

from dropout_pred.data.ingest import file_checksum, load_raw_frame, normalize_columns


def test_normalisation_des_noms_de_colonnes():
    frame = pd.DataFrame({"\ufeffAge\t": [1], " Target ": [2]})
    cleaned = normalize_columns(frame)
    assert list(cleaned.columns) == ["Age", "Target"]


def test_checksum_est_stable(tmp_path):
    path = tmp_path / "fichier.csv"
    path.write_text("contenu identique", encoding="utf-8")
    first = file_checksum(path)
    assert first == file_checksum(path)
    path.write_text("contenu modifié", encoding="utf-8")
    assert file_checksum(path) != first


def test_chargement_refuse_les_colonnes_manquantes(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Marital status;Target\n1;Dropout\n", encoding="utf-8")
    try:
        load_raw_frame(csv_path)
        raise AssertionError("Une ValueError était attendue")
    except ValueError as exc:
        assert "Colonnes absentes" in str(exc)
