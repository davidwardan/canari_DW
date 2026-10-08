"""Controlled numerical particle-update tests independent of Chronos."""

import numpy as np
import pytest

from particle import adapted_update


def test_fully_adapted_update_against_exact_gaussian():
    rng = np.random.default_rng(44)
    histories = rng.normal(0, .5, (100000, 2))
    histories[:, 0] = histories[:, 1]  # Perfect ancestry correlation to preserve.
    predicted = .8*histories[:, -1]
    q, r, y = .04, .09, .3
    updated, indices, diagnostics = adapted_update(histories, predicted, y, q, r, rng)
    prior = .8**2 * .25 + q
    expected_mean = prior/(prior+r)*y
    expected_var = prior*r/(prior+r)
    assert abs(updated[:, -1].mean()-expected_mean) < .003
    assert abs(updated[:, -1].var()-expected_var) < .002
    np.testing.assert_array_equal(updated[:, 0], histories[indices, 1])
    assert 0 < diagnostics["ess"] <= len(histories)


def test_zero_measurement_noise_and_zero_process_noise():
    history = np.arange(10, dtype=float).reshape(5, 2)
    predicted = history[:, -1]*.1
    exact, _, _ = adapted_update(history, predicted, .7, .1, 0., np.random.default_rng(0))
    np.testing.assert_allclose(exact[:, -1], .7)
    fixed, indices, _ = adapted_update(history, predicted, .7, 0., .1, np.random.default_rng(0))
    np.testing.assert_array_equal(fixed[:, -1], predicted[indices])
    with pytest.raises(ValueError):
        adapted_update(history, predicted, .7, 0., 0., np.random.default_rng(0))


@pytest.mark.parametrize("observation,mean", [(1e16, 0.), (0., 1e16)])
def test_extreme_equal_likelihoods_keep_uniform_weights_and_ancestry(observation, mean):
    n = 64
    history = np.arange(n * 2, dtype=float).reshape(n, 2)
    predicted = np.full(n, mean)
    _, indices, diagnostics = adapted_update(history, predicted, observation, .01, .04,
                                             np.random.default_rng(123))
    # Equal likelihoods have normalized mass 1/N even at log likelihood -1e33.
    assert diagnostics["ess"] == pytest.approx(n, rel=1e-14)
    assert diagnostics["unique_ancestors"] == n
    np.testing.assert_array_equal(indices, np.arange(n))
