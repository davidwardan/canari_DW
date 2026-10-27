import numpy as np
import pytest

from canari import Model
from canari.component import LocalLevel
from ssm_constraint import (
    NOISE_SHAPE,
    OBSERVATION,
    TRANSITION,
    LocalTrendConstraint,
    detrended_std,
    std_error_for_response_time,
    step_response_time,
    steady_state,
)

# A fast filter so that transients vanish within a few thousand steps
FAST = {"std_error": 1e-2, "std_residual": 1.0}


def memory_model(**kwargs):
    """Deterministic source 'level' feeding the memories, so the filter is not altered."""
    return Model(LocalLevel(), LocalTrendConstraint(source="level", **kwargs))


def run_memories(model, sequence):
    memories = []
    indices = [model.get_states_index(name) for name in model.components["level aux"].states_name]
    for value in sequence:
        model.forward(state_predictions={"level": (value, 0.0)})
        model.set_states(model.mu_states_prior, model.var_states_prior)
        memories.append(model.mu_states[indices].ravel().copy())
    return np.array(memories)


def test_steady_state_matches_iterated_riccati():
    std_error, std_residual = 3e-3, 0.8
    gain, posterior_var, memory_transition = steady_state(std_error, std_residual)
    process_noise = std_error**2 * NOISE_SHAPE
    posterior = np.eye(2)
    for _ in range(200_000):
        prior = TRANSITION @ posterior @ TRANSITION.T + process_noise
        k = prior @ OBSERVATION.T / (OBSERVATION @ prior @ OBSERVATION.T + std_residual**2)
        posterior = (np.eye(2) - k @ OBSERVATION) @ prior
    np.testing.assert_allclose(gain, k, rtol=1e-8)
    np.testing.assert_allclose(posterior_var, posterior, rtol=1e-8)
    np.testing.assert_allclose(memory_transition, (np.eye(2) - k @ OBSERVATION) @ TRANSITION)


def test_memories_equal_standalone_kalman_filter():
    rng = np.random.default_rng(0)
    sequence = np.sin(np.arange(300) / 5) + 0.01 * np.arange(300) + rng.normal(0, 0.3, 300)
    component = LocalTrendConstraint(source="level", **FAST)
    # Time-varying filter started at the steady-state posterior with a zero mean
    mean, var = np.zeros((2, 1)), component.posterior_var.copy()
    process_noise = FAST["std_error"] ** 2 * NOISE_SHAPE
    expected = []
    for value in sequence:
        mean, var = TRANSITION @ mean, TRANSITION @ var @ TRANSITION.T + process_noise
        gain = var @ OBSERVATION.T / (OBSERVATION @ var @ OBSERVATION.T + FAST["std_residual"] ** 2)
        mean = mean + gain * (value - (OBSERVATION @ mean).item())
        var = (np.eye(2) - gain @ OBSERVATION) @ var
        expected.append(mean.ravel().copy())
    np.testing.assert_allclose(run_memories(memory_model(**FAST), sequence), expected, atol=1e-12)


def test_ramp_is_tracked_exactly():
    time = np.arange(6000)
    memories = run_memories(memory_model(**FAST), 0.3 + 0.01 * time)
    np.testing.assert_allclose(memories[-1], [0.3 + 0.01 * time[-1], 0.01], rtol=1e-6)


def test_seasonal_leakage_matches_frequency_response():
    period = 52
    component = LocalTrendConstraint(source="level", **FAST)
    response = np.linalg.solve(
        np.eye(2) - component.memory_transition * np.exp(-2j * np.pi / period),
        component.gain,
    ).ravel()
    time = np.arange(6000)
    memories = run_memories(memory_model(**FAST), np.cos(2 * np.pi * time / period))
    last = memories[-10 * period:]
    np.testing.assert_allclose(np.abs(last).max(axis=0), np.abs(response), rtol=1e-3)


def test_single_step_constraint_matches_hand_computation():
    component = LocalTrendConstraint(source="level", std_error=1e-3, std_residual=0.5)
    model = Model(LocalLevel(), component)
    previous = np.array([[0.0], [0.2], [0.001]])  # level, aux level, aux slope
    model.set_states(previous, np.zeros((3, 3)))
    mu, var = 0.6, 0.04
    model.forward(state_predictions={"level": (mu, var)})
    k, posterior_var, memory_transition = (
        component.gain, component.posterior_var, component.memory_transition
    )
    memory_mean = memory_transition @ previous[1:] + k * mu
    innovation_var = var * k @ k.T + posterior_var
    expected_mean = mu - var * (k.T @ np.linalg.solve(innovation_var, memory_mean)).item()
    expected_var = var - var**2 * (k.T @ np.linalg.solve(innovation_var, k)).item()
    level = model.get_states_index("level")
    assert model.mu_states_prior[level].item() == pytest.approx(expected_mean, rel=1e-10)
    assert model.var_states_prior[level, level] == pytest.approx(expected_var, rel=1e-10)
    # A positive filtered level is pulled down
    assert model.mu_states_prior[level].item() < mu


def test_response_time_rule():
    std_error = std_error_for_response_time(0.7, 208)
    assert step_response_time(std_error / 0.7) == 208
    assert step_response_time(1.01 * std_error / 0.7) < 208
    np.testing.assert_allclose(
        steady_state(std_error, 0.7)[0], steady_state(std_error / 0.7, 1.0)[0]
    )


def test_detrended_std_of_line_plus_sinusoid():
    time = np.arange(52 * 40, dtype=float)
    values = 2.0 + 0.01 * time + 1.5 * np.sin(2 * np.pi * time / 52)
    values[3] = np.nan
    assert detrended_std(values) == pytest.approx(1.5 / np.sqrt(2), rel=1e-2)


def test_noise_and_rows_select_memories():
    component = LocalTrendConstraint(source="level", **FAST)
    model = Model(LocalLevel(), component)
    rows, noise = component.constraint_observation(model.states_name)
    np.testing.assert_array_equal(rows, [[0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(noise, component.posterior_var)
    np.testing.assert_array_equal(model.observation_matrix, [[1, 0, 0]])


@pytest.mark.parametrize("kwargs", [{"std_error": 0.0}, {"std_residual": -1.0},
                                    {"std_error": np.nan}])
def test_invalid_noise_rejected(kwargs):
    with pytest.raises(ValueError):
        LocalTrendConstraint(**kwargs)
