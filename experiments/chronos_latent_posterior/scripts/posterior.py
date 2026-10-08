"""Exact Gaussian latent-history inference for the controlled experiment.

The model is z_t = a z_(t-1) + w_t, x_t = seasonal_t + z_t,
y_t = x_t + v_t, with independent w_t ~ N(0,q), v_t ~ N(0,r).
Inference sees only the supplied observations; callers pass a past-only prefix.
The stationary prior is used for |a| < 1. A static a=1,q=0 model instead
needs an explicit proper prior (initial_variance defaults to 1).
"""

import numpy as np
from scipy.optimize import minimize


def filter_history(y, seasonal, a, q, r, *, initial_mean=0., initial_variance=None):
    """Return exact filtering moments and the observation log likelihood.

    ``means`` describe the full latent signal x; ``residual_means`` describe z.
    The prior refers to the first supplied latent residual, before y[0].
    """
    y = np.asarray(y, dtype=float)
    seasonal = np.asarray(seasonal, dtype=float)
    if y.ndim != 1 or len(y) == 0 or seasonal.shape != y.shape:
        raise ValueError("y and seasonal must be matching nonempty vectors")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(seasonal)):
        raise ValueError("observations and seasonal means must be finite")
    if not np.isfinite(a) or q < 0 or r < 0 or q + r <= 0 or not np.isfinite(q + r):
        raise ValueError("require finite a, nonnegative q/r, and q + r > 0")
    if initial_variance is None:
        if abs(a) < 1:
            initial_variance = q / (1 - a * a)
        elif a == 1 and q == 0:
            initial_variance = 1.
        else:
            raise ValueError("nonstationary models need an initial_variance")
    if initial_variance < 0 or not np.isfinite(initial_variance):
        raise ValueError("initial_variance must be finite and nonnegative")

    size = len(y)
    means, variances = np.empty(size), np.empty(size)
    predicted_means, predicted_variances = np.empty(size), np.empty(size)
    innovation_variances, innovations = np.empty(size), np.empty(size)
    mean, variance = float(initial_mean), float(initial_variance)
    for t in range(size):
        if t:
            mean, variance = a * means[t - 1], a * a * variances[t - 1] + q
        predicted_means[t], predicted_variances[t] = mean, variance
        innovation = y[t] - seasonal[t] - mean
        innovation_variance = variance + r
        gain = variance / innovation_variance
        means[t] = mean + gain * innovation
        # This form preserves positivity when the observation is very precise.
        variances[t] = variance * r / innovation_variance
        innovations[t], innovation_variances[t] = innovation, innovation_variance
    loglik = -.5 * np.sum(np.log(2 * np.pi * innovation_variances)
                          + innovations**2 / innovation_variances)
    return {
        "means": means + seasonal,
        "residual_means": means,
        "variances": variances,
        "predicted_means": predicted_means + seasonal,
        "predicted_residual_means": predicted_means,
        "predicted_variances": predicted_variances,
        "innovations": innovations,
        "innovation_variances": innovation_variances,
        "loglik": float(loglik),
        "initial_variance": float(initial_variance),
    }


def smooth_history(filtered, seasonal, a, *, covariance=False):
    """Exact posterior moments of the entire supplied history (RTS smoother).

    Smoothing the past supplied history is allowed: no observation after its
    last element is used. Off-diagonal covariances preserve time dependence.
    """
    mean = filtered["residual_means"].copy()
    variance = filtered["variances"].copy()
    size = len(mean)
    gains = np.zeros(max(size - 1, 0))
    for t in range(size - 2, -1, -1):
        predicted_variance = filtered["predicted_variances"][t + 1]
        gain = a * filtered["variances"][t] / predicted_variance if predicted_variance else 0.
        gains[t] = gain
        mean[t] += gain * (mean[t + 1] - a * filtered["residual_means"][t])
        variance[t] += gain**2 * (variance[t + 1] - predicted_variance)
    result = {"mean": mean + np.asarray(seasonal), "variances": variance, "gains": gains}
    if covariance:
        cov = np.diag(variance)
        for t in range(size - 2, -1, -1):
            cov[t, t + 1:] = gains[t] * cov[t + 1, t + 1:]
            cov[t + 1:, t] = cov[t, t + 1:]
        result["covariance"] = cov
    return result


def sample_history(y, seasonal, a, q, r, n_draws, rng, *, initial_mean=0.,
                   initial_variance=None, covariance=False):
    """Draw JOINT histories using forward filtering/backward sampling.

    Returns a dict: ``draws`` is [n_draws, history_length], ``mean`` is the
    exact smoothed history, ``variances`` its marginal variances, and optional
    ``covariance`` is the exact joint posterior covariance. The terminal
    ``filtered_variance`` is the uncertainty entering an oracle forecast.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    filtered = filter_history(y, seasonal, a, q, r,
                              initial_mean=initial_mean, initial_variance=initial_variance)
    moments = smooth_history(filtered, seasonal, a, covariance=covariance)
    draws = np.empty((n_draws, len(y)))
    draws[:, -1] = rng.normal(filtered["residual_means"][-1],
                              np.sqrt(filtered["variances"][-1]), n_draws)
    for t in range(len(y) - 2, -1, -1):
        gain = moments["gains"][t]
        mean = filtered["residual_means"][t] + gain * (
            draws[:, t + 1] - a * filtered["residual_means"][t])
        predicted_variance = filtered["predicted_variances"][t + 1]
        # Pf * Q / Ppred equals Pf - gain² Ppred, without cancellation at Q=0.
        variance = filtered["variances"][t] * q / predicted_variance if predicted_variance else 0.
        draws[:, t] = mean + rng.normal(0, np.sqrt(max(variance, 0.)), n_draws)
    return {
        **moments,
        "draws": draws + np.asarray(seasonal),
        "filtered_mean": float(filtered["means"][-1]),
        "filtered_variance": float(filtered["variances"][-1]),
        "loglik": filtered["loglik"],
    }


def forecast_components(a, q, r, filtered_variance, horizons):
    """Exact future-noise and current-state components at positive horizons.

    Epistemic here means current-state uncertainty, not frozen-weight model
    uncertainty. Q/R and a are known, so no parameter uncertainty is present.
    """
    horizons = np.asarray(horizons, dtype=int)
    if np.any(horizons < 1):
        raise ValueError("forecast horizons must be positive")
    epistemic = a**(2 * horizons) * filtered_variance
    aleatoric = np.array([r + q * np.sum(a**(2 * np.arange(h))) for h in horizons.flat])
    aleatoric = aleatoric.reshape(horizons.shape)
    return {"epistemic": epistemic, "aleatoric": aleatoric,
            "total": aleatoric + epistemic}


def gaussian_condition(y, seasonal, a, q, r, *, initial_mean=0., initial_variance=None):
    """Direct Gaussian conditioning used as an independent numerical oracle."""
    y, seasonal = np.asarray(y), np.asarray(seasonal)
    filtered = filter_history(y, seasonal, a, q, r,
                              initial_mean=initial_mean, initial_variance=initial_variance)
    size = len(y)
    prior_mean = a**np.arange(size) * initial_mean
    prior_variance = np.empty(size)
    prior_variance[0] = filtered["initial_variance"]
    for t in range(1, size):
        prior_variance[t] = a * a * prior_variance[t - 1] + q
    prior_cov = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            prior_cov[i, j] = a**abs(i - j) * prior_variance[min(i, j)]
    observed_cov = prior_cov + r * np.eye(size)
    mean = seasonal + prior_mean + prior_cov @ np.linalg.solve(
        observed_cov, y - seasonal - prior_mean)
    covariance = prior_cov - prior_cov @ np.linalg.solve(observed_cov, prior_cov)
    return {"mean": mean, "covariance": covariance}


def fit_noise(y, seasonal, a):
    """Fit positive Q/R by training-prefix innovation likelihood, with fixed a.

    Callers enforce the chronological training boundary. Multiple deterministic
    starts reveal the Q/R ridge at a=0 rather than hiding it with one optimum.
    Q/R fitting is restricted to the stationary AR model used in this study.
    """
    if abs(a) >= 1:
        raise ValueError("noise fitting requires abs(a) < 1")
    scale = max(float(np.var(np.asarray(y) - np.asarray(seasonal))), 1e-8)
    bounds = [(np.log(scale * 1e-6), np.log(scale * 10))] * 2
    fits = []
    for fraction in (.05, .25, .5, .75, .95):
        start = np.log([scale * fraction * (1 - a * a), scale * (1 - fraction)])
        result = minimize(lambda value: -filter_history(
            y, seasonal, a, *np.exp(value))["loglik"], start,
            method="L-BFGS-B", bounds=bounds)
        q, r = np.exp(result.x)
        fits.append({"q": float(q), "r": float(r), "loglik": float(-result.fun),
                     "success": bool(result.success)})
    best = max(fits, key=lambda fit: fit["loglik"])
    return {**best, "fits": fits, "n_observations": len(y),
            "a_fixed": float(a), "a_zero_nonidentifiable": bool(a == 0)}
