"""Configuration must be overridable by environment, with safe local defaults."""

import importlib
import os

import src.config


def _reload(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(src.config)


def test_defaults_are_local_and_leak_no_infrastructure():
    config = importlib.reload(src.config)
    assert config.MLFLOW_TRACKING_URI.startswith('file://')
    assert 'amazonaws.com' not in config.MLFLOW_TRACKING_URI


def test_tracking_uri_is_overridable(monkeypatch):
    config = _reload(monkeypatch, MLFLOW_TRACKING_URI='http://tracking.example:5000')
    assert config.MLFLOW_TRACKING_URI == 'http://tracking.example:5000'


def test_raw_data_path_is_overridable(monkeypatch):
    config = _reload(monkeypatch, RAW_DATA_PATH='/tmp/other.csv')
    assert config.RAW_DATA_PATH == '/tmp/other.csv'


def test_path_resolves_relative_to_repo_root():
    config = importlib.reload(src.config)
    assert config.path('params.yaml') == os.path.join(config.ROOT_DIR, 'params.yaml')
    assert os.path.isfile(config.path('params.yaml'))


def teardown_module():
    """Leave the module in its unpatched state for other tests."""
    importlib.reload(src.config)
