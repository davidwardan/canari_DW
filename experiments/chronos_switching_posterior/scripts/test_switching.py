import numpy as np

from switching import SwitchingParticleFilter, mixture_scores, systematic_resample


def identity_transition(histories):
    return np.asarray(histories)[:, -1]


def make_filter(observation_seed=3, n=128):
    return SwitchingParticleFilter(
        identity_transition,
        np.zeros((n, 3)),
        np.zeros(n, dtype=int),
        np.array([[.99, .01], [.10, .90]]),
        np.array([0., 1.]),
        np.array([.01, .01]),
        np.array([.04, .04]),
        np.random.default_rng(observation_seed),
    )


def test_systematic_resample_is_reproducible_and_valid():
    first = systematic_resample(np.array([.25, .5, .25]), np.random.default_rng(4))
    second = systematic_resample(np.array([.25, .5, .25]), np.random.default_rng(4))
    assert np.array_equal(first, second)
    assert len(first) == 3
    assert np.all((first >= 0) & (first < 3))


def test_observation_favoring_abnormal_moves_regime_probability():
    particle_filter = make_filter()
    prediction = particle_filter.predict()
    posterior = particle_filter.update(.9, prediction)
    assert posterior["posterior_regime_probability"][1] > prediction["regime_probability"][1]
    np.testing.assert_allclose(posterior["posterior_regime_probability"].sum(), 1.)


def test_variance_terms_reconstruct_total():
    particle_filter = make_filter()
    prediction = particle_filter.predict()
    assert prediction["aleatoric"] >= 0
    assert prediction["state"] >= 0
    assert prediction["regime"] >= 0
    np.testing.assert_allclose(
        prediction["total"], prediction["aleatoric"] + prediction["state"] + prediction["regime"])


def test_conditional_update_is_between_forecast_and_observation():
    particle_filter = make_filter()
    prediction = particle_filter.predict()
    posterior = particle_filter.update(.9, prediction)
    assert posterior["posterior_mean"] > prediction["mean"]
    assert np.isfinite(posterior["posterior_mean"])


def test_switching_update_keeps_particle_population_fixed():
    particle_filter = make_filter(n=5)
    for observation in (.2, .8, .1):
        prediction = particle_filter.predict()
        particle_filter.update(observation, prediction)
        assert len(particle_filter.histories) == 5
        assert len(particle_filter.regimes) == 5
        assert len(particle_filter.weights) == 5


def test_mixture_scores_are_finite():
    nll, crps = mixture_scores(.5, np.array([0., 1.]), np.array([.05, .05]), np.array([.5, .5]))
    assert np.isfinite(nll)
    assert np.isfinite(crps)
