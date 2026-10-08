"""Switching particle filtering around a deterministic Chronos transition."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.special import logsumexp
from scipy.stats import norm


def systematic_resample(weights: np.ndarray, rng: np.random.Generator, sample_size=None) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0 or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be a finite nonempty vector")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise ValueError("weights must be nonnegative and sum to one")
    n = len(weights) if sample_size is None else int(sample_size)
    if n <= 0:
        raise ValueError("sample_size must be positive")
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(cumulative, positions)


def _weighted_moments(values, weights):
    mean = float(np.sum(weights * values))
    variance = float(np.sum(weights * (values - mean) ** 2))
    return mean, variance


def gaussian_crps(observation, mean, variance):
    sigma = float(np.sqrt(max(variance, 1e-15)))
    z = (observation - mean) / sigma
    return float(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))


def _abs_normal_moment(delta, variance):
    sigma = np.sqrt(np.maximum(variance, 1e-15))
    z = delta / sigma
    return sigma * np.sqrt(2 / np.pi) * np.exp(-0.5 * z * z) + delta * (2 * ndtr(z) - 1)


def mixture_scores(observation, means, variances, weights):
    """Exact finite Gaussian-mixture NLL and CRPS for scalar components."""
    means, variances, weights = map(np.asarray, (means, variances, weights))
    std = np.sqrt(np.maximum(variances, 1e-15))
    log_density = norm.logpdf(observation, means, std) + np.log(weights)
    nll = float(-logsumexp(log_density))
    first = np.sum(weights * _abs_normal_moment(observation - means, variances))
    pair_delta = means[:, None] - means[None, :]
    pair_var = variances[:, None] + variances[None, :]
    second = 0.5 * np.sum(weights[:, None] * weights[None, :] *
                           _abs_normal_moment(pair_delta, pair_var))
    return nll, float(first - second)


class SwitchingParticleFilter:
    """Particle filter with a Markov regime and exact scalar Gaussian update.

    ``transition`` maps a batch of latent histories to the frozen Chronos
    transition mean. Each particle carries its full history and its current
    discrete regime. Regime-specific shifts, Q, and R are explicit model
    parameters; the Chronos weights remain frozen.
    """

    def __init__(self, transition, initial_histories, initial_regimes,
                 transition_matrix, regime_shifts, q, r, rng):
        self.transition = transition
        self.histories = np.asarray(initial_histories, dtype=float).copy()
        self.regimes = np.asarray(initial_regimes, dtype=int).copy()
        self.transition_matrix = np.asarray(transition_matrix, dtype=float)
        self.shifts = np.asarray(regime_shifts, dtype=float)
        self.q = np.asarray(q, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.rng = rng
        self.weights = np.full(len(self.histories), 1 / len(self.histories))
        n_regimes = len(self.shifts)
        if self.histories.ndim != 2 or len(self.regimes) != len(self.histories):
            raise ValueError("histories and regimes have incompatible shapes")
        if self.transition_matrix.shape != (n_regimes, n_regimes):
            raise ValueError("transition matrix shape must match regimes")
        if np.any(self.transition_matrix < 0) or not np.allclose(
                self.transition_matrix.sum(axis=1), 1.):
            raise ValueError("transition matrix rows must be probability vectors")
        if np.any(self.q < 0) or np.any(self.r < 0) or np.any(self.q + self.r <= 0):
            raise ValueError("regime noise variances must be valid")

    def _candidates(self):
        n = len(self.histories)
        k = len(self.shifts)
        old_index = np.repeat(np.arange(n), k)
        new_regime = np.tile(np.arange(k), n)
        old_regime = self.regimes[old_index]
        prior = self.weights[old_index] * self.transition_matrix[old_regime, new_regime]
        histories = self.histories[old_index]
        # The same latent history is paired with every candidate regime. Query
        # Chronos once per history, then apply regime-specific residual shifts.
        base_by_history = np.asarray(self.transition(self.histories), dtype=float).reshape(-1)
        means = base_by_history[old_index] + self.shifts[new_regime]
        variances = self.q[new_regime] + self.r[new_regime]
        return old_index, old_regime, new_regime, prior, histories, means, variances

    def predict(self):
        """Return the prior predictive mixture and its three variance terms."""
        old_index, old_regime, new_regime, prior, histories, means, variances = self._candidates()
        alpha = prior / prior.sum()
        n_regimes = len(self.shifts)
        regime_probability = np.bincount(new_regime, weights=alpha, minlength=n_regimes)
        regime_mean = np.zeros(n_regimes)
        regime_state = np.zeros(n_regimes)
        for regime in range(n_regimes):
            mask = new_regime == regime
            regime_mean[regime], regime_state[regime] = _weighted_moments(
                means[mask], alpha[mask] / regime_probability[regime])
        mean = float(np.sum(alpha * means))
        aleatoric = float(np.sum(alpha * variances))
        state = float(np.sum(regime_probability * regime_state))
        regime = float(np.sum(regime_probability * (regime_mean - mean) ** 2))
        total = aleatoric + state + regime
        return {
            "old_index": old_index, "old_regime": old_regime,
            "new_regime": new_regime, "histories": histories,
            "means": means, "variances": variances, "weights": alpha,
            "regime_probability": regime_probability,
            "regime_mean": regime_mean, "regime_state": regime_state,
            "mean": mean, "aleatoric": aleatoric, "state": state,
            "regime": regime, "total": total,
        }

    def update(self, observation, prediction=None):
        """Assimilate one observation and return posterior regime diagnostics."""
        prediction = self.predict() if prediction is None else prediction
        log_likelihood = norm.logpdf(
            observation, prediction["means"], np.sqrt(prediction["variances"]))
        log_weights = np.log(prediction["weights"]) + log_likelihood
        normalized = np.exp(log_weights - logsumexp(log_weights))
        posterior_regime_probability = np.bincount(
            prediction["new_regime"], weights=normalized,
            minlength=len(self.shifts))
        pair_switch_probability = float(np.sum(
            normalized[prediction["old_regime"] != prediction["new_regime"]]))
        # Keep the particle population fixed even though the switching
        # proposal has N_particles x N_regimes candidates.
        selected = systematic_resample(normalized, self.rng, len(self.histories))
        regime = prediction["new_regime"][selected]
        means = prediction["means"][selected]
        q = self.q[regime]
        r = self.r[regime]
        total_noise = q + r
        conditional_mean = means + q / total_noise * (observation - means)
        conditional_variance = q * r / total_noise
        draws = conditional_mean + self.rng.normal(size=len(selected)) * np.sqrt(conditional_variance)
        self.histories = np.column_stack([prediction["histories"][selected, 1:], draws])
        self.regimes = regime.copy()
        self.weights = np.full(len(selected), 1 / len(selected))
        return {
            "posterior_regime_probability": posterior_regime_probability,
            "switch_probability": pair_switch_probability,
            "ess": float(1 / np.sum(normalized ** 2)),
            "unique_ancestors": int(len(np.unique(selected))),
            "posterior_mean": float(draws.mean()),
            "posterior_sd": float(draws.std()),
        }
