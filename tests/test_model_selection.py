"""Tests for metric-gated model registration and the imbalance sweep."""

import types

import numpy as np
import pytest

from src.model import register_model as rm


def _run(run_id, metric_value):
    """Minimal stand-in for an MLflow Run object."""
    return types.SimpleNamespace(
        info=types.SimpleNamespace(run_id=run_id),
        data=types.SimpleNamespace(metrics={rm.SELECTION_METRIC: metric_value}),
    )


def test_promotes_the_best_run_not_the_latest(monkeypatch):
    """A worse candidate must not displace a better historical run."""
    monkeypatch.setattr(rm, 'find_best_run',
                        lambda *_: _run('older_but_better', 0.81))

    chosen = rm.select_run({'run_id': 'just_evaluated'}, 'exp', rm.SELECTION_METRIC)

    assert chosen == 'older_but_better'


def test_promotes_the_candidate_when_it_wins(monkeypatch):
    monkeypatch.setattr(rm, 'find_best_run',
                        lambda *_: _run('just_evaluated', 0.83))

    chosen = rm.select_run({'run_id': 'just_evaluated'}, 'exp', rm.SELECTION_METRIC)

    assert chosen == 'just_evaluated'


def test_falls_back_to_candidate_when_no_metric_available(monkeypatch):
    """A tracking server without the metric must not fail the pipeline."""
    monkeypatch.setattr(rm, 'find_best_run', lambda *_: None)

    chosen = rm.select_run({'run_id': 'just_evaluated'}, 'exp', rm.SELECTION_METRIC)

    assert chosen == 'just_evaluated'


def test_selection_metric_is_macro_not_accuracy():
    """Accuracy would let a model that ignores the minority class win."""
    assert 'macro' in rm.SELECTION_METRIC


class TestSweep:
    """The sweep imports LightGBM, which is optional for the light test run."""

    @pytest.fixture(autouse=True)
    def _skip_without_lightgbm(self):
        pytest.importorskip('lightgbm')

    def test_oversampling_equalises_classes(self):
        from src.model.run_experiments import resample_indices

        y = np.array([-1] * 10 + [0] * 30 + [1] * 60)
        idx = resample_indices(y, 'over', np.random.default_rng(0))

        _, counts = np.unique(y[idx], return_counts=True)
        assert set(counts) == {60}

    def test_undersampling_equalises_classes(self):
        from src.model.run_experiments import resample_indices

        y = np.array([-1] * 10 + [0] * 30 + [1] * 60)
        idx = resample_indices(y, 'under', np.random.default_rng(0))

        _, counts = np.unique(y[idx], return_counts=True)
        assert set(counts) == {10}

    def test_strategies_map_to_distinct_model_configs(self):
        from src.model.run_experiments import STRATEGIES, build_model

        balanced = build_model('class_weight_balanced', 42)
        unbalance = build_model('is_unbalance', 42)
        plain = build_model('none', 42)

        assert balanced.class_weight == 'balanced'
        assert unbalance.get_params().get('is_unbalance') is True
        assert plain.class_weight is None
        assert len(STRATEGIES) == 5

    def test_sweep_grid_is_the_documented_size(self):
        """The CV and README both cite 5 strategies x 10 feature sizes."""
        from src.model.run_experiments import FEATURE_SIZES, STRATEGIES

        assert len(STRATEGIES) == 5
        assert len(FEATURE_SIZES) == 10
        assert len(STRATEGIES) * len(FEATURE_SIZES) == 50
