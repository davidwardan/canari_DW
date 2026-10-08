"""Checks for a training-only finite-grid parameter/state/noise mixture."""

import numpy as np
import pytest
from scipy.special import logsumexp

from parameter import noise_grid, parameter_forecast, weighted_quantiles
from posterior import filter_history, forecast_components


def test_noise_grid_normalized_exact_likelihood_weights():
    y = np.array([.2, -.3, .1, .4, -.1, .2, .3, -.2])
    seasonal = np.zeros(len(y))
    grid = noise_grid(y, seasonal, .8)
    assert len(grid["nodes"]) == 81
    weights = np.array([node["weight"] for node in grid["nodes"]])
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1., abs=1e-14)
    logliks = []
    for node in grid["nodes"]:
        filtered = filter_history(y, seasonal, .8, node["q"], node["r"])
        assert node["filtered_residual_mean"] == pytest.approx(filtered["residual_means"][-1])
        assert node["filtered_variance"] == pytest.approx(filtered["variances"][-1])
        logliks.append(filtered["loglik"])
    np.testing.assert_allclose(weights, np.exp(logliks - logsumexp(logliks)), atol=1e-14)
    assert 0 <= grid["summary"]["edge_mass"] <= 1


def test_fixed_support_and_only_declared_training_prefix():
    train = np.array([.1, -.3, .4, .2, -.2, .6])
    full = np.concatenate([train, [10000., -10000.]])
    grid = noise_grid(full[:len(train)], np.zeros(len(train)), .7)
    variance = np.var(train)
    assert grid["n_training"] == len(train)
    assert grid["training_residual_variance"] == pytest.approx(variance)
    np.testing.assert_allclose(grid["q_values"], np.geomspace(1e-4, 1., 9))
    np.testing.assert_allclose(grid["r_values"], np.geomspace(.005, 1., 9))
    # Including future observations changes likelihood weights, never prior support.
    contaminated = noise_grid(full, np.zeros(len(full)), .7)
    assert contaminated["training_residual_variance"] > 1e6 * variance
    np.testing.assert_array_equal(grid["q_values"], contaminated["q_values"])
    np.testing.assert_array_equal(grid["r_values"], contaminated["r_values"])
    assert not np.allclose([node["weight"] for node in grid["nodes"]],
                           [node["weight"] for node in contaminated["nodes"]])


def test_smoke_grid_size_alias():
    grid = noise_grid([.1, .2, -.3], [0., 0., 0.], .8, n_nodes=3)
    assert len(grid["nodes"]) == 9
    np.testing.assert_allclose(grid["q_values"], [1e-4, .01, 1.])


def test_exact_weighted_second_moment_identity():
    nodes = [
        {"weight": .25, "q": .1, "r": .2, "filtered_variance": .04,
         "filtered_residual_mean": .5},
        {"weight": .75, "q": .03, "r": .4, "filtered_variance": .12,
         "filtered_residual_mean": -.3},
    ]
    a, horizons, seasonal = .8, np.array([1, 4, 13]), np.array([.3, -.2, .8])
    forecast = parameter_forecast({"nodes": nodes}, seasonal, horizons, a)
    weights = np.array([node["weight"] for node in nodes])
    means = np.array([seasonal + a**horizons * node["filtered_residual_mean"] for node in nodes])
    variances = np.array([forecast_components(a, node["q"], node["r"],
                                              node["filtered_variance"], horizons)["total"]
                          for node in nodes])
    np.testing.assert_allclose(forecast["mu"], weights @ means)
    np.testing.assert_allclose(forecast["total"], weights @ (variances + means**2)
                               - forecast["mu"]**2, atol=1e-14)
    np.testing.assert_allclose(forecast["total"], forecast["aleatoric"]
                               + forecast["state"] + forecast["parameter"])
    assert np.all(forecast["parameter"] > 0)
    assert np.all(forecast["state"] >= 0)
    assert np.all(forecast["aleatoric"] >= 0)


def test_single_parameter_matches_known_gaussian_control():
    a, q, r = .75, .04, .2
    y, seasonal = np.array([.3, -.2, .1, .7, .4]), np.zeros(5)
    filtered = filter_history(y, seasonal, a, q, r)
    node = {"weight": 1., "q": q, "r": r,
            "filtered_variance": filtered["variances"][-1],
            "filtered_residual_mean": filtered["residual_means"][-1]}
    horizons = np.arange(1, 8)
    forecast = parameter_forecast({"nodes": [node]}, np.ones(7), horizons, a)
    oracle = forecast_components(a, q, r, filtered["variances"][-1], horizons)
    np.testing.assert_array_equal(forecast["parameter"], 0.)
    np.testing.assert_allclose(forecast["mu"], 1 + a**horizons * filtered["residual_means"][-1])
    np.testing.assert_allclose(forecast["state"], oracle["epistemic"])
    np.testing.assert_allclose(forecast["aleatoric"], oracle["aleatoric"])
    np.testing.assert_allclose(forecast["total"], oracle["total"])


def test_static_grid_has_only_r_nodes_and_exact_constant_forecast():
    y = np.linspace(.1, .6, 16)
    grid = noise_grid(y, np.zeros(16), 1.)
    assert len(grid["nodes"]) == 9
    assert grid["q_values"] == [0.]
    assert grid["summary"]["q_mean"] == 0.
    assert grid["summary"]["q_lower_mass"] == 0.
    for node in grid["nodes"]:
        variance = 1 / (1 + len(y) / node["r"])
        mean = variance * y.sum() / node["r"]
        assert node["filtered_variance"] == pytest.approx(variance)
        assert node["filtered_residual_mean"] == pytest.approx(mean)
    forecast = parameter_forecast(grid, np.zeros(3), [1, 4, 52], 1.)
    for key in ("mu", "aleatoric", "state", "parameter", "total"):
        np.testing.assert_allclose(forecast[key], forecast[key][0])
    assert forecast["aleatoric"][0] == pytest.approx(grid["summary"]["r_mean"])


def test_weighted_quantiles_use_discrete_mass():
    assert weighted_quantiles([3., 1., 2.], [.1, .2, .7]) == [1., 2., 3.]


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        parameter_forecast({"nodes": []}, [0.], [1, 2], .8)
