"""Metric helpers for the EWMA drift-attribution toy. See ../planning/metrics.md."""

import numpy as np
from scipy.stats import norm

PERIOD = 52


def block_slope(values, start_time: int, period: int = PERIOD) -> float:
    """Slope of the complete-cycle means of `values`, per time step.

    A zero-mean `period`-periodic part averages out exactly, so a ramp plus such a
    pattern returns the ramp's slope.
    """

    count = len(values) // period
    if count < 2:
        return float("nan")
    means = np.asarray(values[: count * period], dtype=float).reshape(count, period)
    centres = start_time + period * np.arange(count) + (period - 1) / 2
    return float(np.polyfit(centres, means.mean(axis=1), 1)[0])


def slope_reading(fast, slow, fast_span: float, slow_span: float, dt: float = 1.0):
    """(fast - slow) / (slow lag - fast lag), as in `EwmaConstraint`."""

    rhos = np.exp(-dt / np.asarray([fast_span, slow_span], dtype=float))
    lags = dt * rhos / (1 - rhos)
    return (np.asarray(fast) - np.asarray(slow)) / (lags[1] - lags[0])


def ewma_readings(source, fast_span: float, slow_span: float, dt: float = 1.0):
    """Mean and slope readings of the two EWMA memories, started at zero."""

    rhos = np.exp(-dt / np.asarray([fast_span, slow_span], dtype=float))
    memories = np.zeros(2)
    history = np.empty((len(source), 2))
    for index, value in enumerate(source):
        memories = rhos * memories + (1 - rhos) * value
        history[index] = memories
    return history[:, 1], slope_reading(history[:, 0], history[:, 1], fast_span, slow_span, dt)


def leakage(pattern_cycle, fast_span: float, slow_span: float, num_cycles: int = 60):
    """Peak steady-state |mean| and |slope| readings of a periodic source."""

    period = len(pattern_cycle)
    mean, slope = ewma_readings(np.tile(pattern_cycle, num_cycles), fast_span, slow_span)
    return float(np.abs(mean[-period:]).max()), float(np.abs(slope[-period:]).max())


def rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def score(obs, mean, std) -> dict:
    """Accuracy and Gaussian predictive-distribution scores."""

    error = obs - mean
    z = error / std
    log_lik = -0.5 * np.log(2 * np.pi * std**2) - 0.5 * z**2
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return {
        "rmse": rms(error),
        "mae": float(np.mean(np.abs(error))),
        "ll": float(np.mean(log_lik)),
        "crps": float(np.mean(crps)),
        "coverage_95": float(np.mean(np.abs(z) <= 1.959963984540054)),
    }


def crps(obs, mean, std) -> float:
    return score(obs, mean, std)["crps"]


def adaptation_kept(excess_arm: float, excess_frozen: float, excess_free: float) -> float:
    """eta: 1 recovers like unconstrained RSR, 0 like a frozen LSTM."""

    return (excess_frozen - excess_arm) / (excess_frozen - excess_free)
