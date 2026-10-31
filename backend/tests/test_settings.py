import pytest
from pydantic import ValidationError

from app.settings import EXAMPLE_CONFIG, Settings, load_settings


def test_example_config_loads():
    settings = load_settings(EXAMPLE_CONFIG)
    assert settings.app.host == "127.0.0.1"
    assert settings.app.mode == "manual"
    assert settings.model.backend == "none"
    assert settings.simulation.url == "http://127.0.0.1:8001"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_allowed(host):
    assert Settings(app={"host": host}).app.host == host


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com"])
def test_non_loopback_hosts_rejected(host):
    with pytest.raises(ValidationError):
        Settings(app={"host": host})


def test_simulator_url_must_be_loopback():
    with pytest.raises(ValidationError):
        Settings(simulation={"url": "http://10.0.0.5:8001"})


def test_unknown_config_keys_rejected():
    with pytest.raises(ValidationError):
        Settings(app={"hostt": "127.0.0.1"})
