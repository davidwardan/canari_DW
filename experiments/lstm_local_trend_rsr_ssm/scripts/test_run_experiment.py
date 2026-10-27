import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

import run_experiment as experiment
from canari import Model
from canari.component import EwmaConstraint, LocalTrend, LstmNetwork, WhiteNoise
from run_experiment import score
from ssm_constraint import LocalTrendConstraint

CONSTRAINT_TYPES = {"ewma": EwmaConstraint, "ssm": LocalTrendConstraint}


def small_model(arm):
    components = [
        LocalTrend(),
        LstmNetwork(look_back_len=1, num_features=1, num_hidden_unit=4, manual_seed=1),
        WhiteNoise(std_error=0.1),
    ]
    if arm == "ewma":
        components.append(EwmaConstraint(source="lstm", fast_span=4, slow_span=8))
    elif arm == "ssm":
        components.append(LocalTrendConstraint(source="lstm", std_error=1e-2, std_residual=0.7))
    return Model(*components)


def test_crps_standard_normal_at_mean():
    # CRPS(N(0, 1), 0) = (sqrt(2) - 1) / sqrt(pi)
    result = score(np.array([0.0]), np.array([0.0]), np.array([1.0]))
    assert np.isclose(result["crps"], (np.sqrt(2) - 1) / np.sqrt(np.pi))


def test_crps_matches_numerical_integral():
    obs, mean, std = 1.3, 0.4, 0.7
    below, _ = integrate.quad(lambda x: norm.cdf(x, mean, std) ** 2, -np.inf, obs)
    above, _ = integrate.quad(lambda x: norm.sf(x, mean, std) ** 2, obs, np.inf)
    result = score(np.array([obs]), np.array([mean]), np.array([std]))
    assert np.isclose(result["crps"], below + above)


def test_scores_ignore_missing_observations():
    obs = np.array([1.0, np.nan, -0.5])
    mean = np.array([0.5, 9.0, 0.0])
    std = np.array([0.5, 1.0, 2.0])
    result = score(obs, mean, std)
    log_lik = norm.logpdf(obs[[0, 2]], mean[[0, 2]], std[[0, 2]])
    assert result["n_obs"] == 2
    assert np.isclose(result["rmse"], np.sqrt(0.25))
    assert np.isclose(result["mae"], 0.5)
    assert np.isclose(result["ll_mean"], log_lik.mean())
    assert np.isclose(result["ll_total"], log_lik.sum())


@pytest.mark.parametrize("method", experiment.METHODS)
@pytest.mark.parametrize("arm", experiment.ARMS)
def test_full_history_preserves_test_predictions_and_states(monkeypatch, method, arm):
    monkeypatch.setattr(experiment, "WINDOW_LEN", 3)
    data = {
        "x": np.empty((20, 0), dtype=np.float32),
        "y": np.sin(np.arange(20, dtype=np.float32)).reshape(-1, 1),
    }
    data["y"][9] = np.nan
    snapshot = small_model(arm).get_dict()
    test_mean, test_std, test_states = experiment.predict_test(snapshot, data, 10, method)
    mean, std, states = experiment.predict_test(
        snapshot, data, 10, method, include_history=True
    )
    assert len(mean) == len(std) == len(data["y"])
    assert np.all(np.isfinite(mean))
    assert np.all(std > 0)
    np.testing.assert_array_equal(mean[10:], test_mean)
    np.testing.assert_array_equal(std[10:], test_std)
    for key, values in states.items():
        assert len(values) == len(data["y"])
        np.testing.assert_array_equal(values[10:], test_states[key])

    # Online histories must retain the shared frozen prefix before their first window.
    offline_mean, offline_std, offline_states = experiment.predict_test(
        snapshot, data, 10, "offline", include_history=True
    )
    np.testing.assert_array_equal(mean[:7], offline_mean[:7])
    np.testing.assert_array_equal(std[:7], offline_std[:7])
    for key, values in states.items():
        np.testing.assert_array_equal(values[:7], offline_states[key][:7])


@pytest.mark.parametrize("error", [
    np.linalg.LinAlgError("SVD did not converge"),
    ValueError("Prediction moments must be finite with variance >= 0"),
])
@pytest.mark.parametrize("failure_epoch", [0, 6])
def test_training_divergence_retains_only_valid_best_snapshot(monkeypatch, failure_epoch, error):
    class Processor:
        def get_splits(self):
            data = {"y": np.zeros((2, 1))}
            return data, data, data, data

    class DivergingModel:
        early_stop_metric = None
        optimal_epoch = 0
        stop_training = False
        epoch = 0

        def auto_initialize_baseline_states(self, data):
            pass

        def lstm_train(self, **kwargs):
            if self.epoch == failure_epoch:
                raise error
            return np.zeros(2), np.ones(2), None

        def early_stopping(self, evaluate_metric, current_epoch, **kwargs):
            self.early_stop_metric = evaluate_metric if current_epoch >= 5 else -np.inf
            self.optimal_epoch = current_epoch if current_epoch >= 5 else 0

        def set_memory(self, **kwargs):
            self.epoch += 1

        def get_dict(self):
            return {"epoch": self.epoch - 1}

    monkeypatch.setattr(experiment, "build_model", lambda *args: DivergingModel())
    monkeypatch.setattr(experiment.metric, "log_likelihood", lambda *args: 1.0)
    best_ll, best_epoch, snapshot, history, status = experiment.train_offline(Processor(), 0.01, 1)
    assert status["status"] == "numerical_failure"
    assert status["failure_epoch"] == failure_epoch
    assert len(history) == failure_epoch
    if failure_epoch == 0:
        assert snapshot is None
        assert best_ll == -np.inf
    else:
        assert snapshot == {"epoch": 5}
        assert best_epoch == 5
        assert best_ll == 1.0


def test_unrelated_value_error_is_not_treated_as_divergence(monkeypatch):
    class Processor:
        def get_splits(self):
            data = {"y": np.zeros((2, 1))}
            return data, data, data, data

    class BrokenModel:
        def auto_initialize_baseline_states(self, data):
            pass

        def lstm_train(self, **kwargs):
            raise ValueError("Unknown EWMA source state")

    monkeypatch.setattr(experiment, "build_model", lambda *args: BrokenModel())
    with pytest.raises(ValueError, match="Unknown EWMA source"):
        experiment.train_offline(Processor(), 0.01, 1)


@pytest.mark.parametrize("arm", ["ewma", "ssm"])
def test_constrained_snapshot_preserves_joint_state_and_constraint(arm):
    model = experiment.build_model(0.1, 1, arm, std_residual=0.8)
    rng = np.random.default_rng(20)
    transform = rng.normal(size=(model.num_states, model.num_states))
    model.set_states(rng.normal(size=(model.num_states, 1)),
                     transform @ transform.T + np.eye(model.num_states))
    loaded = experiment.load_model(model.get_dict())
    assert [type(c) for c in loaded.ewma_constraints] == [CONSTRAINT_TYPES[arm]]
    np.testing.assert_array_equal(loaded._constraint_matrix, model._constraint_matrix)
    np.testing.assert_array_equal(loaded._constraint_noise, model._constraint_noise)
    np.testing.assert_array_equal(loaded.transition_matrix, model.transition_matrix)
    np.testing.assert_array_equal(loaded.mu_states, model.mu_states)
    np.testing.assert_array_equal(loaded.var_states, model.var_states)
    for expected, actual in zip(
        model.forward(mu_lstm_pred=[0.3], var_lstm_pred=[0.2]),
        loaded.forward(mu_lstm_pred=[0.3], var_lstm_pred=[0.2]),
    ):
        np.testing.assert_array_equal(actual, expected)


def test_ssm_arm_uses_the_declared_noise_ratio():
    model = experiment.build_model(0.1, 1, "ssm", std_residual=0.8)
    (component,) = model.ewma_constraints
    assert component.std_residual == 0.8
    assert component.std_error == pytest.approx(0.8 * experiment.SSM_STD_RATIO)
    assert experiment.SSM_STD_RATIO == pytest.approx(7.07e-6, rel=1e-3)


@pytest.mark.parametrize("arm", ["ewma", "ssm"])
def test_epoch_restart_resets_auxiliary_prior(arm):
    model = small_model(arm)
    data = {"x": np.empty((15, 0), dtype=np.float32),
            "y": (1 + 0.1 * np.arange(15, dtype=np.float32)).reshape(-1, 1)}
    model.lstm_net.num_samples = 15
    model.filter(data)
    states = model.smoother()
    model.set_memory(states, 0)
    indices = model._ewma_indices
    assert np.abs(np.asarray(states.mu_smooth[0])[indices]).max() > 0
    np.testing.assert_array_equal(model.mu_states[indices], 0)
    np.testing.assert_array_equal(model.var_states[indices, :], 0)
    np.testing.assert_array_equal(model.var_states[:, indices], 0)


def test_lstm_drift_recovers_added_linear_component():
    t = np.arange(5 * experiment.PERIOD + 7)
    values = 0.003 * t + np.sin(2 * np.pi * t / experiment.PERIOD)
    assert experiment.lstm_drift(values) == pytest.approx(0.003)


def test_build_model_preserves_actual_source_baseline():
    for arm in experiment.ARMS:
        model = experiment.build_model(0.1, 1, arm)
        np.testing.assert_array_equal(model.components["local trend"].process_noise_matrix, 0)
