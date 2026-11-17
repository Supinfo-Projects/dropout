from dropout_pred.pipeline_state import (
    read_pipeline_state,
    write_pipeline_state,
)


def test_ecriture_lecture_etat_succes():
    write_pipeline_state("succes", started_at=100.0, duration_s=42.5, version="3", n_batches=5)
    state = read_pipeline_state()
    assert state["state"] == "succes"
    assert state["duration_s"] == 42.5
    assert state["version"] == "3"
    assert state["last_success"]["version"] == "3"


def test_etat_echec_conserve_le_dernier_succes():
    write_pipeline_state("succes", started_at=100.0, duration_s=42.5, version="3")
    write_pipeline_state("echec", started_at=200.0, duration_s=5.0, error="DataValidationError")
    state = read_pipeline_state()
    assert state["state"] == "echec"
    assert state["error"]
    # On garde le dernier succès même en cas d'échec
    assert state["last_success"]["version"] == "3"


def test_etat_running_conserve_le_dernier_succes():
    write_pipeline_state("succes", started_at=100.0, duration_s=42.5, version="3")
    write_pipeline_state("running", started_at=300.0, n_batches=5)
    state = read_pipeline_state()
    assert state["state"] == "running"
    assert state["last_success"]["version"] == "3"


def test_aucun_etat():
    import shutil

    from dropout_pred.config import get_settings

    shutil.rmtree(get_settings().artifacts_dir, ignore_errors=True)
    assert read_pipeline_state() is None
