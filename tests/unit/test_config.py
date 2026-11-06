from dropout_pred.config import get_settings


def test_configuration_par_defaut():
    settings = get_settings()
    assert settings.api_port == 8000
    assert settings.min_f1_macro > 0
    assert settings.mlflow_model_name


def test_configuration_surchargeable_par_environnement(monkeypatch):
    monkeypatch.setenv("API_PORT", "9999")
    monkeypatch.setenv("MIN_F1_MACRO", "0.7")
    settings = get_settings()
    assert settings.api_port == 9999
    assert settings.min_f1_macro == 0.7
