"""Focused checks for the real-series EWMA experiment.

    PYTHONPATH=/Users/davidwardan/canari_DW/src python -m pytest \
        experiments/ewma_real_series_rsr/scripts/test_experiment.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment as rx  # noqa: E402
from metrics import score  # noqa: E402


def toy_setup():
    detrended = rx.detrended_series("toy")
    data_processor = rx.make_processor(rx.with_trend(detrended))
    level, baseline = rx.true_baseline(detrended)
    return detrended, data_processor, level, baseline


def test_initial_state_predicts_the_true_baseline_at_week_zero():
    _, data_processor, level, baseline = toy_setup()
    model = rx.build_model(0.1, seed=1)
    rx.set_baseline(model, rx.initial_baseline(level, data_processor))
    y_mean, y_std = rx.scale(data_processor)
    prior = model.transition_matrix @ model.mu_states
    assert prior[model.get_states_index("level"), 0] * y_std + y_mean == pytest.approx(baseline[0])
    assert prior[model.get_states_index("trend"), 0] * y_std == pytest.approx(rx.SLOPE)
    level_index = model.get_states_index("level")
    assert model.var_states[level_index, level_index] == rx.LEVEL_VAR
    assert np.count_nonzero(model.var_states[level_index]) == 1


def test_every_epoch_starts_from_the_true_baseline(monkeypatch):
    _, data_processor, level, _ = toy_setup()
    baseline = rx.initial_baseline(level, data_processor)
    starts = []
    original = rx.Model.lstm_train

    def record(self, *args, **kwargs):
        indices = [self.get_states_index(name) for name in ("level", "trend")]
        starts.append((self.mu_states[indices, 0].copy(),
                       np.diag(self.var_states)[indices].copy()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(rx.Model, "lstm_train", record)
    # canari's early stopping only keeps epochs from the 6th on
    snapshot = rx.train_offline(data_processor, 0.1, seed=1, taus=(0.06, 6e-4),
                                baseline=baseline, max_epochs=7)[2]
    assert len(starts) == 7
    indices = [snapshot["states_name"].index(name) for name in ("level", "trend")]
    np.testing.assert_allclose(snapshot["mu_states"][indices, 0], baseline)
    for means, variances in starts:
        np.testing.assert_allclose(means, baseline)
        np.testing.assert_allclose(variances, [rx.LEVEL_VAR, rx.TREND_VAR])


def test_true_level_and_leak_on_the_toy():
    detrended, data_processor, level, baseline = toy_setup()
    assert abs(level) < 0.02
    np.testing.assert_allclose(np.diff(baseline), rx.SLOPE)
    leak = rx.leak_of(detrended, level, data_processor)
    # The toy experiment's exact leakage of the true pattern
    assert leak[0] == pytest.approx(0.061, rel=0.05)
    assert leak[1] == pytest.approx(5.9e-4, rel=0.05)


def test_scores_skip_missing_weeks():
    obs = np.array([1.0, np.nan, 3.0])
    result = score(obs, np.array([1.0, 0.0, 3.0]), np.ones(3))
    assert result["n_obs"] == 2 and result["rmse"] == 0.0


def test_every_arm_has_a_snapshot_and_tolerance():
    leak = (0.06, 6e-4)
    for arm, spec in rx.ARMS.items():
        assert spec["snapshot"] in rx.SNAPSHOTS
        taus = rx.taus_of(spec["snapshot"], leak)
        assert (taus is None) == (spec["snapshot"] == "none")
    assert rx.taus_of("ewma_k0.25", leak) == pytest.approx((0.015, 1.5e-4))
