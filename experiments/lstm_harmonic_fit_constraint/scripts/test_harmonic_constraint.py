import numpy as np
import pytest

from canari import Model
from canari.component import LocalLevel, LocalTrend, LstmNetwork, WhiteNoise
from harmonic_constraint import (
    HarmonicTrendConstraint,
    basis,
    discounted_fit,
    shift_matrix,
    steady_state,
)

PERIOD, RHO = 52.0, np.exp(-1 / 208)


def seasonal_line(time, a=0.3, b=0.004):
    return (a + b * time + 0.8 * np.cos(2 * np.pi * time / PERIOD + 0.4)
            + 0.5 * np.sin(4 * np.pi * time / PERIOD + np.pi / 4))


def run_memories(model, sequence):
    names = model.components["level harmonic"].states_name
    indices = [model.get_states_index(name) for name in names]
    memories = []
    for value in sequence:
        model.forward(state_predictions={"level": (value, 0.0)})
        model.set_states(model.mu_states_prior, model.var_states_prior)
        memories.append(model.mu_states[indices].ravel().copy())
    return np.array(memories)


def test_shift_matrix_advances_basis():
    lags = np.arange(-5, 5)
    np.testing.assert_allclose(shift_matrix(PERIOD, 3) @ basis(lags, PERIOD, 3),
                               basis(lags + 1, PERIOD, 3), atol=1e-12)


def test_steady_state_recursion_equals_batch_fit():
    gain, transition = steady_state(PERIOD, 2, RHO)
    assert np.allclose(np.abs(np.linalg.eigvals(transition)), RHO)
    rng = np.random.default_rng(1)
    values = rng.normal(size=6000)
    theta = np.zeros(6)
    for value in values:
        theta = transition @ theta + gain.ravel() * value
    np.testing.assert_allclose(theta, discounted_fit(values, PERIOD, 2, RHO), atol=1e-9)


@pytest.mark.parametrize("length", [13, 30, 300])
def test_finite_fit_rejects_modelled_wave_exactly(length):
    # Exact from a quarter cycle; shorter windows are exact in algebra but ill-conditioned
    time = np.arange(length)
    theta = discounted_fit(seasonal_line(time), PERIOD, 2, RHO)
    np.testing.assert_allclose(theta[:2], [0.3 + 0.004 * time[-1], 0.004], atol=1e-8)


def test_warm_started_memories_stay_exact_through_model():
    component = HarmonicTrendConstraint(source="level", period=PERIOD, num_harmonics=2)
    model = Model(LocalLevel(), component)
    history = seasonal_line(np.arange(260))
    mu = model.mu_states.copy()
    mu[1:] = discounted_fit(history, PERIOD, 2, component.rho).reshape(-1, 1)
    model.set_states(mu, np.zeros_like(model.var_states))
    time = np.arange(260, 900)
    memories = run_memories(model, seasonal_line(time))
    np.testing.assert_allclose(memories[:, 0], 0.3 + 0.004 * time, atol=1e-8)
    np.testing.assert_allclose(memories[:, 1], 0.004, atol=1e-10)


def test_unmodelled_harmonic_leaks():
    time = np.arange(3000)
    theta = discounted_fit(np.sin(4 * np.pi * time / PERIOD), PERIOD, 1, RHO)
    assert abs(theta[0]) > 1e-3


def test_rows_noise_and_names():
    component = HarmonicTrendConstraint(source="level", num_harmonics=2,
                                        tau_level=0.1, tau_slope=0.001)
    model = Model(LocalLevel(), component)
    rows, noise = component.constraint_observation(model.states_name)
    expected = np.zeros((2, 7))
    expected[0, 1] = expected[1, 2] = 1
    np.testing.assert_array_equal(rows, expected)
    np.testing.assert_allclose(noise, np.diag([0.01, 1e-6]))
    assert model.states_name[1:4] == ["level harmonic level", "level harmonic slope",
                                      "level harmonic cos1"]
    np.testing.assert_array_equal(model.observation_matrix, [[1, 0, 0, 0, 0, 0, 0]])


@pytest.mark.parametrize("strategy", ["replay_filtered", "replay_smoothed"])
def test_rsr_runs_and_snapshot_round_trips(strategy):
    model = Model(
        LocalTrend(),
        LstmNetwork(look_back_len=4, num_features=1, num_hidden_unit=4, manual_seed=1),
        WhiteNoise(std_error=0.1),
        HarmonicTrendConstraint(source="lstm", period=8, num_harmonics=1, span=16),
    )
    loaded = Model.load_dict(model.get_dict())
    np.testing.assert_array_equal(loaded.transition_matrix, model.transition_matrix)
    np.testing.assert_array_equal(loaded._constraint_noise, model._constraint_noise)
    time = np.arange(40, dtype=np.float32)
    data = {"x": np.empty((40, 0), dtype=np.float32),
            "y": (0.02 * time + np.sin(2 * np.pi * time / 8)).reshape(-1, 1)}
    mean, std, states = loaded.rsr_filter(data, start=8, window_len=8, strategy=strategy)
    assert len(mean) == 32 and np.all(np.isfinite(mean)) and np.all(std > 0)


@pytest.mark.parametrize("kwargs", [{"span": 0.0}, {"tau_level": -1.0},
                                    {"num_harmonics": 30, "period": 52.0}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        HarmonicTrendConstraint(**kwargs)
