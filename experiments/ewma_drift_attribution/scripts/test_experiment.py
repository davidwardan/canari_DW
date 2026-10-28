"""Focused checks for the EWMA drift-attribution toy.

Run from the repository root with the source tree that has `EwmaConstraint`:

    PYTHONPATH=/Users/davidwardan/canari_DW/src python -m pytest \
        experiments/ewma_drift_attribution/scripts/test_experiment.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment as rx  # noqa: E402
from canari import Model  # noqa: E402
from canari.component import EwmaConstraint, LocalTrend, LstmNetwork, WhiteNoise  # noqa: E402
from metrics import PERIOD, block_slope, ewma_readings, leakage, slope_reading  # noqa: E402


def test_block_slope_is_exact_for_ramp_plus_periodic_pattern():
    time = np.arange(300, 300 + 11 * PERIOD + 7)
    values = 3.0 + 0.004 * time + rx.pattern(time)
    assert block_slope(values, 300) == pytest.approx(0.004, abs=1e-10)


def test_shares_add_up_to_the_observation_slope():
    rng = np.random.default_rng(0)
    time = np.arange(11 * PERIOD)
    level = 0.003 * time + rng.normal(size=time.size)
    lstm = 0.002 * time + rng.normal(size=time.size)
    noise = rng.normal(size=time.size)
    parts = sum(block_slope(values, 0) for values in (level, lstm, noise))
    assert parts == pytest.approx(block_slope(level + lstm + noise, 0), rel=1e-10)


def test_leakage_of_one_harmonic_matches_transfer_function():
    omega = 2 * np.pi / PERIOD
    rhos = np.exp(-1 / np.array([104.0, 208.0]))
    lags = rhos / (1 - rhos)
    gains = (1 - rhos) / (1 - rhos * np.exp(-1j * omega))
    mean_leak, slope_leak = leakage(1.3 * np.sin(omega * np.arange(PERIOD)), 104, 208)
    # Sampling the peak at integer steps loses at most 1 - cos(pi / 52)
    assert mean_leak == pytest.approx(1.3 * abs(gains[1]), rel=2e-3)
    assert slope_leak == pytest.approx(
        1.3 * abs(gains[0] - gains[1]) / (lags[1] - lags[0]), rel=2e-3)


def test_slope_reading_recovers_a_steady_ramp_and_matches_component():
    _, slope = ewma_readings(0.01 * np.arange(5000), 104, 208)
    assert slope[-1] == pytest.approx(0.01, rel=1e-6)
    component = EwmaConstraint(source="lstm", fast_span=104, slow_span=208,
                               tau_mean=0.1, tau_slope=0.01)
    matrix, _ = component.constraint_observation(["lstm", *component.states_name])
    state = np.array([0.0, 0.3, -0.2])
    assert matrix[0] @ state == pytest.approx(-0.2)
    assert matrix[1] @ state == pytest.approx(slope_reading(0.3, -0.2, 104, 208))


def test_scenarios_are_paired():
    a, b, c, d = (rx.toy_series(name, seed=3)["values"].to_numpy() for name in "ABCD")
    np.testing.assert_allclose(b - a, rx.SLOPE * np.arange(rx.NUM_STEPS), atol=1e-12)
    np.testing.assert_array_equal(b, c)
    np.testing.assert_array_equal(b[: rx.CHANGE_START], d[: rx.CHANGE_START])
    assert np.abs(b[rx.CHANGE_START:] - d[rx.CHANGE_START:]).max() > 0.5
    assert (rx.CHANGE_START - rx.TEST_START) % PERIOD == 0
    shifted = rx.true_pattern("D")[rx.TEST_START:][: 11 * PERIOD]
    np.testing.assert_allclose(shifted.reshape(11, PERIOD).mean(axis=1), 0, atol=1e-12)


def test_leak_matches_planning_values():
    leak = rx.leak_of(rx.make_processor(rx.toy_series("B", seed=1)))
    assert leak[0] == pytest.approx(0.061, rel=0.02)
    assert leak[1] == pytest.approx(5.9e-4, rel=0.02)


@pytest.mark.parametrize("arm, variances", [
    ("rsr_kinf", []),
    ("rsr_k2", [0.12**2, 1.2e-3**2]),
    ("frozen", [0.06**2, 6e-4**2]),
    ("rsr_mean_k1", [0.06**2]),
    ("rsr_slope_k1", [6e-4**2]),
])
def test_arm_tolerances_are_swapped_into_the_snapshot(arm, variances):
    snapshot = rx.build_model(0.1, seed=1, taus=(0.5, 0.5)).get_dict()
    model = rx.arm_model(snapshot, arm, leak=(0.06, 6e-4))
    assert model._constraint_matrix.shape[0] == len(variances)
    np.testing.assert_allclose(np.diag(model._constraint_noise), variances)
    np.testing.assert_array_equal(model.mu_states, snapshot["mu_states"])
    assert model.states_name == snapshot["states_name"]


def test_none_arm_has_no_constraint():
    snapshot = rx.build_model(0.1, seed=1).get_dict()
    model = rx.arm_model(snapshot, "none", leak=(0.06, 6e-4))
    assert not model.ewma_constraints


def test_rsr_posterior_components_add_up_to_the_observation():
    data = {"x": np.empty((16, 0), dtype=np.float32),
            "y": (0.2 + np.sin(np.arange(16) * 0.5)).astype(np.float32)[:, None]}
    model = Model(
        LocalTrend(std_error=0.0, mu_states=[0.1, 0.005], var_states=[0.2, 0.01]),
        LstmNetwork(num_hidden_unit=4, manual_seed=11),
        WhiteNoise(std_error=0.1),
        EwmaConstraint(fast_span=4, slow_span=16, tau_mean=0.05, tau_slope=0.01),
    )
    _, _, states = model.rsr_filter(data, start=6, window_len=3, strategy=rx.STRATEGY)
    total = sum(states.get_mean(name, "posterior") for name in ("level", "lstm", "white noise"))
    np.testing.assert_allclose(total, data["y"][6:, 0], atol=1e-6)
