"""Analytical controls for posterior propagation and variance separation."""

import numpy as np
import pytest

from posterior import (filter_history, fit_noise, forecast_components,
                       gaussian_condition, sample_history)


def simulate(a, q, r, size, rng):
    residual = np.empty(size)
    residual[0] = rng.normal(0, np.sqrt(q / (1 - a * a)))
    for t in range(1, size):
        residual[t] = a * residual[t - 1] + rng.normal(0, np.sqrt(q))
    return residual + rng.normal(0, np.sqrt(r), size)


@pytest.mark.parametrize("a,q,r", [(.8, .12, .3), (-.65, .05, .2), (0., .1, .4)])
def test_joint_posterior_matches_direct_gaussian_conditioning(a, q, r):
    seasonal = .5 * np.sin(np.arange(9) * .4)
    y = seasonal + simulate(a, q, r, 9, np.random.default_rng(1))
    oracle = gaussian_condition(y, seasonal, a, q, r)
    sampled = sample_history(y, seasonal, a, q, r, 30000,
                             np.random.default_rng(43), covariance=True)
    np.testing.assert_allclose(sampled["mean"], oracle["mean"], atol=2e-14)
    np.testing.assert_allclose(sampled["covariance"], oracle["covariance"], atol=2e-14)
    sampling_mean_error = 5 * np.sqrt(np.diag(oracle["covariance"]) / 30000)
    assert np.all(np.abs(sampled["draws"].mean(axis=0) - oracle["mean"]) < sampling_mean_error)
    # Gaussian covariance SE includes both diagonal products and covariance².
    covariance_error = 5 * np.sqrt((np.outer(np.diag(oracle["covariance"]),
                                            np.diag(oracle["covariance"]))
                                    + oracle["covariance"]**2) / 29999)
    assert np.all(np.abs(np.cov(sampled["draws"], rowvar=False)
                         - oracle["covariance"]) < covariance_error)


def test_nonstationary_prior_and_known_initial_mean():
    seasonal, y = np.zeros(6), np.arange(6) / 5
    oracle = gaussian_condition(y, seasonal, 1., .03, .2,
                                initial_mean=.7, initial_variance=.8)
    sampled = sample_history(y, seasonal, 1., .03, .2, 100,
                             np.random.default_rng(9), initial_mean=.7,
                             initial_variance=.8, covariance=True)
    np.testing.assert_allclose(sampled["mean"], oracle["mean"], atol=1e-14)
    np.testing.assert_allclose(sampled["covariance"], oracle["covariance"], atol=1e-14)


def test_static_offset_posterior_and_joint_draws():
    size, prior_mean, prior_variance, r = 12, -.2, 1.4, .3
    seasonal = np.cos(np.arange(size) * .3)
    y = seasonal + np.linspace(.1, .5, size)
    variance = 1 / (1 / prior_variance + size / r)
    mean = variance * (prior_mean / prior_variance + np.sum(y - seasonal) / r)
    sampled = sample_history(y, seasonal, 1., 0., r, 30000,
                             np.random.default_rng(67), initial_mean=prior_mean,
                             initial_variance=prior_variance, covariance=True)
    np.testing.assert_allclose(sampled["mean"] - seasonal, mean, atol=1e-14)
    np.testing.assert_allclose(sampled["covariance"], variance, atol=1e-14)
    residual_draws = sampled["draws"] - seasonal
    np.testing.assert_allclose(residual_draws, residual_draws[:, :1] + np.zeros(size), atol=4e-8)
    components = forecast_components(1., 0., r, variance, [1, 4, 12])
    np.testing.assert_allclose(components["epistemic"], variance)
    np.testing.assert_allclose(components["aleatoric"], r)


def test_linear_future_mixture_recovers_oracle():
    a, q, r = .8, .05, .2
    seasonal = .4 * np.sin(np.arange(16) / 3)
    y = seasonal + simulate(a, q, r, 16, np.random.default_rng(16))
    sampled = sample_history(y, seasonal, a, q, r, 30000, np.random.default_rng(18))
    horizons = np.arange(1, 13)
    mean_draws = (sampled["draws"][:, -1] - seasonal[-1])[:, None] * a**horizons
    oracle = forecast_components(a, q, r, sampled["filtered_variance"], horizons)
    np.testing.assert_allclose(np.var(mean_draws, axis=0), oracle["epistemic"], rtol=.025)
    # Nested future draws verify E[conditional variance] + Var[conditional mean].
    rng = np.random.default_rng(30)
    futures = mean_draws + rng.normal(size=mean_draws.shape) * np.sqrt(oracle["aleatoric"])
    np.testing.assert_allclose(np.var(futures, axis=0), oracle["total"], rtol=.035)
    np.testing.assert_allclose(oracle["total"], oracle["epistemic"] + oracle["aleatoric"])


def test_independent_marginal_sampling_loses_history_covariance():
    size, a, q, r = 16, .95, .02, .5
    sampled = sample_history(np.zeros(size), np.zeros(size), a, q, r, 30000,
                             np.random.default_rng(23), covariance=True)
    weight = np.ones(size) / size
    exact = weight @ sampled["covariance"] @ weight
    joint = np.var(sampled["draws"] @ weight)
    independent = np.random.default_rng(24).normal(
        sampled["mean"], np.sqrt(sampled["variances"]), (30000, size))
    independent_exact = np.sum(weight**2 * sampled["variances"])
    assert joint == pytest.approx(exact, rel=.025)
    assert np.var(independent @ weight) == pytest.approx(independent_exact, rel=.025)
    assert independent_exact < exact / 4


def test_a_zero_likelihood_identifies_only_q_plus_r():
    y = np.random.default_rng(3).normal(0, np.sqrt(.4), 120)
    logliks = [filter_history(y, np.zeros(120), 0., q, .4 - q)["loglik"]
               for q in (.03, .12, .25, .37)]
    np.testing.assert_allclose(logliks, logliks[0], atol=1e-12)
    fitted = fit_noise(y, np.zeros(120), 0.)
    assert fitted["a_zero_nonidentifiable"]
    sums = [fit["q"] + fit["r"] for fit in fitted["fits"]]
    np.testing.assert_allclose(sums, np.mean(y**2), rtol=2e-5)
    assert np.ptp([fit["q"] for fit in fitted["fits"]]) > .15


def test_ar_likelihood_can_identify_training_noise():
    a, q, r = .8, .08, .2
    y = simulate(a, q, r, 5000, np.random.default_rng(138))
    fitted = fit_noise(y, np.zeros(len(y)), a)
    assert fitted["success"]
    assert not fitted["a_zero_nonidentifiable"]
    assert fitted["n_observations"] == len(y)
    assert fitted["q"] == pytest.approx(q, rel=.14)
    assert fitted["r"] == pytest.approx(r, rel=.14)
    np.testing.assert_allclose([fit["q"] for fit in fitted["fits"]], fitted["q"], rtol=.001)
    np.testing.assert_allclose([fit["r"] for fit in fitted["fits"]], fitted["r"], rtol=.001)
    # Keeping Q+R fixed no longer keeps the likelihood fixed when a != 0.
    alternative = filter_history(y, np.zeros(len(y)), a, .20, .08)["loglik"]
    assert fitted["loglik"] - alternative > 100


def test_forecast_components_geometric_sum_and_horizon_validation():
    a, q, r, p = .75, .03, .4, .2
    horizons = np.arange(1, 11)
    components = forecast_components(a, q, r, p, horizons)
    np.testing.assert_allclose(components["aleatoric"],
                               r + q * (1 - a**(2 * horizons)) / (1 - a * a))
    np.testing.assert_allclose(components["epistemic"], a**(2 * horizons) * p)
    with pytest.raises(ValueError):
        forecast_components(a, q, r, p, [0])


def test_single_observation_and_zero_process_variance():
    sampled = sample_history([.2], [0.], .8, .03, .1, 10,
                             np.random.default_rng(1), covariance=True)
    assert sampled["draws"].shape == (10, 1)
    assert sampled["covariance"].shape == (1, 1)
    deterministic = sample_history([.2, .3], [0., 0.], .8, 0., .1, 10,
                                   np.random.default_rng(1), covariance=True)
    np.testing.assert_array_equal(deterministic["draws"], 0.)
    np.testing.assert_array_equal(deterministic["covariance"], 0.)


def test_zero_observation_noise_fixes_all_past_latents():
    y, seasonal = np.array([.2, -.1, .8]), np.array([.1, .2, .3])
    sampled = sample_history(y, seasonal, .8, .03, 0., 300,
                             np.random.default_rng(9), covariance=True)
    np.testing.assert_allclose(sampled["draws"], y + np.zeros((300, 3)), atol=1e-14)
    np.testing.assert_allclose(sampled["mean"], y, atol=1e-14)
    np.testing.assert_array_equal(sampled["covariance"], 0.)
    components = forecast_components(.8, .03, 0., sampled["filtered_variance"], [1, 2, 3])
    np.testing.assert_array_equal(components["epistemic"], 0.)
    assert np.all(components["aleatoric"] > 0)
