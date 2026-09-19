import pytest

from app.core.config import Settings


def _settings(**overrides):
    # Explicit kwargs win over any local .env, so these tests are environment-independent.
    return Settings(_env_file=None, **overrides)


def test_production_without_an_api_key_refuses_to_start():
    with pytest.raises(RuntimeError, match="API_KEY must be set"):
        _settings(environment="production", api_key="").ensure_production_ready()


def test_production_with_an_api_key_starts():
    _settings(environment="production", api_key="secret").ensure_production_ready()


def test_development_without_an_api_key_is_allowed():
    _settings(environment="development", api_key="").ensure_production_ready()


def test_environment_defaults_to_development():
    assert _settings().environment == "development"


def test_unknown_environment_values_are_rejected():
    with pytest.raises(ValueError):
        _settings(environment="staging")
