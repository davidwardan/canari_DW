import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "chronos_switching_posterior" / "scripts"))

from switching import SwitchingParticleFilter


def transition(histories):
    return np.asarray(histories)[:, -1]


def test_particle_count_stays_fixed_with_two_regimes():
    n = 7
    particle_filter = SwitchingParticleFilter(
        transition, np.zeros((n, 4)), np.zeros(n, dtype=int),
        np.array([[.995, .005], [.08, .92]]), np.array([0., 2.5]),
        np.array([.1, .1]), np.array([.05, .05]), np.random.default_rng(5))
    for observation in (0.0, 2.5, 0.0):
        prediction = particle_filter.predict()
        particle_filter.update(observation, prediction)
        assert particle_filter.histories.shape == (n, 4)
        assert particle_filter.regimes.shape == (n,)


def test_injected_level_moves_abnormal_posterior():
    particle_filter = SwitchingParticleFilter(
        transition, np.zeros((32, 4)), np.zeros(32, dtype=int),
        np.array([[.995, .005], [.08, .92]]), np.array([0., 2.5]),
        np.array([.1, .1]), np.array([.05, .05]), np.random.default_rng(6))
    prediction = particle_filter.predict()
    posterior = particle_filter.update(2.5, prediction)
    assert posterior["posterior_regime_probability"][1] > .5


def test_variance_identity_is_exact_to_roundoff():
    particle_filter = SwitchingParticleFilter(
        transition, np.zeros((16, 4)), np.zeros(16, dtype=int),
        np.array([[.995, .005], [.08, .92]]), np.array([0., 2.5]),
        np.array([.1, .1]), np.array([.05, .05]), np.random.default_rng(7))
    prediction = particle_filter.predict()
    np.testing.assert_allclose(
        prediction["total"], prediction["aleatoric"] + prediction["state"] + prediction["regime"])
