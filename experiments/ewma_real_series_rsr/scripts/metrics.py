"""Metric helpers for the real-series EWMA experiment. See ../planning/metrics.md."""

import numpy as np
from scipy.stats import norm

PERIOD = 52


def slope_reading(fast, slow, fast_span: float, slow_span: float, dt: float = 1.0):
    """s_t = (z^{M_f} - z^{M_s}) / (slow lag - fast lag), as in `EwmaConstraint`."""

    rhos = np.exp(-dt / np.asarray([fast_span, slow_span], dtype=float))
    lags = dt * rhos / (1 - rhos)
    return (np.asarray(fast) - np.asarray(slow)) / (lags[1] - lags[0])


def ewma_readings(source, fast_span: float, slow_span: float, dt: float = 1.0):
    """Slow memory z^{M_s} and slope reading s_t of a source, memories started at zero."""

    rhos = np.exp(-dt / np.asarray([fast_span, slow_span], dtype=float))
    memories = np.zeros(2)
    history = np.empty((len(source), 2))
    for index, value in enumerate(source):
        memories = rhos * memories + (1 - rhos) * value
        history[index] = memories
    return history[:, 1], slope_reading(history[:, 0], history[:, 1], fast_span, slow_span, dt)


def leakage(cycle, fast_span: float, slow_span: float, num_cycles: int = 60):
    """Peak steady-state |z^{M_s}| and |s_t| of a periodic source."""

    period = len(cycle)
    mean, slope = ewma_readings(np.tile(cycle, num_cycles), fast_span, slow_span)
    return float(np.abs(mean[-period:]).max()), float(np.abs(slope[-period:]).max())


def rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def score(obs, mean, std) -> dict:
    """One-step predictive scores on the observed weeks."""

    observed = np.isfinite(obs)
    error = obs[observed] - mean[observed]
    std = std[observed]
    z = error / std
    log_lik = -0.5 * np.log(2 * np.pi * std**2) - 0.5 * z**2
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return {
        "rmse": rms(error),
        "ll": float(np.mean(log_lik)),
        "crps": float(np.mean(crps)),
        "coverage_95": float(np.mean(np.abs(z) <= 1.959963984540054)),
        "n_obs": int(observed.sum()),
    }
