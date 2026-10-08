"""Fully adapted scalar Gaussian observation update retaining context ancestry."""

import numpy as np
from scipy.special import logsumexp


def adapted_update(histories, transition_means, observation, q, r, rng):
    if q < 0 or r < 0 or q+r <= 0:
        raise ValueError("Require nonnegative Q,R and positive Q+R")
    histories, transition_means = np.asarray(histories), np.asarray(transition_means)
    n = len(histories)
    log_weights = -.5 * (observation-transition_means)**2 / (q+r)
    weights = np.exp(log_weights-logsumexp(log_weights))
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.
    indices = np.searchsorted(cumulative, (rng.random()+np.arange(n))/n)
    means = transition_means[indices] + q/(q+r)*(observation-transition_means[indices])
    draws = means + rng.normal(0, np.sqrt(q*r/(q+r)), n)
    updated = np.column_stack([histories[indices, 1:], draws])
    return updated, indices, {"ess": float(1/np.sum(weights**2)),
                              "unique_ancestors": int(len(np.unique(indices))),
                              "posterior_mean": float(draws.mean()),
                              "posterior_sd": float(draws.std())}
