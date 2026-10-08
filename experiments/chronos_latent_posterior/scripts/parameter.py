"""Finite-grid Bayesian Q/R uncertainty under a prespecified discrete prior.

Equal prior mass is assigned to each logarithmically spaced node. The support
is fixed in benchmark amplitude-squared units before observing data. Posterior
calculations are exact on this declared coarse grid, not on a continuous prior.
"""

import numpy as np
from scipy.special import logsumexp

from posterior import filter_history, forecast_components


def weighted_quantiles(values, weights, probabilities=(.05, .5, .95)):
    """Discrete posterior quantiles (no interpolation between grid nodes)."""
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    indices = np.searchsorted(np.cumsum(weights) / np.sum(weights), probabilities)
    return values[np.minimum(indices, len(values) - 1)].tolist()


def noise_grid(y_train, seasonal_train, a, nodes=9, *, n_nodes=None):
    """Infer Q/R on the supplied training prefix and retain its terminal state.

    Stationary AR: fixed Q support [1e-4, 1], fixed R support [.005, 1], in
    benchmark amplitude-squared units. Static a=1,Q=0: only R is inferred.
    No subsequent validation/test observation updates weights or state here.
    """
    y_train = np.asarray(y_train, dtype=float)
    seasonal_train = np.asarray(seasonal_train, dtype=float)
    if n_nodes is not None:
        nodes = n_nodes
    if nodes < 1 or nodes != int(nodes):
        raise ValueError("nodes must be a positive integer")
    q_values = np.array([0.]) if a == 1 else np.geomspace(1e-4, 1., nodes)
    r_values = np.geomspace(.005, 1., nodes)
    estimates = []
    for q in q_values:
        for r in r_values:
            filtered = filter_history(y_train, seasonal_train, a, q, r)
            estimates.append({
                "q": float(q), "r": float(r), "loglik": filtered["loglik"],
                "filtered_variance": float(filtered["variances"][-1]),
                "Pfiltered": float(filtered["variances"][-1]),
                "filtered_residual_mean": float(filtered["residual_means"][-1]),
            })
    likelihoods = np.array([estimate["loglik"] for estimate in estimates])
    weights = np.exp(likelihoods - logsumexp(likelihoods))
    for estimate, weight in zip(estimates, weights):
        estimate["weight"] = float(weight)
    q = np.array([estimate["q"] for estimate in estimates])
    r = np.array([estimate["r"] for estimate in estimates])
    q_lower = q == q_values[0] if len(q_values) > 1 else np.zeros(len(q), dtype=bool)
    q_upper = q == q_values[-1] if len(q_values) > 1 else np.zeros(len(q), dtype=bool)
    r_lower, r_upper = r == r_values[0], r == r_values[-1]
    summary = {
        "q_mean": float(weights @ q), "r_mean": float(weights @ r),
        "quantile_probabilities": [.05, .5, .95],
        "q_quantiles": weighted_quantiles(q, weights),
        "r_quantiles": weighted_quantiles(r, weights),
        "edge_mass": float(weights[q_lower | q_upper | r_lower | r_upper].sum()),
        "q_lower_mass": float(weights[q_lower].sum()),
        "q_upper_mass": float(weights[q_upper].sum()),
        "r_lower_mass": float(weights[r_lower].sum()),
        "r_upper_mass": float(weights[r_upper].sum()),
    }
    return {
        "nodes": estimates, "summary": summary,
        "training_residual_variance": float(np.var(y_train - seasonal_train)),
        "n_training": len(y_train),
        "q_values": q_values.tolist(), "r_values": r_values.tolist(),
        "prior": "equal mass per logarithmic grid node",
        "initial_variance": 1. if a == 1 else "stationary Q/(1-a^2)",
    }


def parameter_forecast(grid, seasonal_future, horizons, a):
    """Exact mixture forecast from TRAIN end, split into three variance terms.

    Future noise = E_theta[R + Q sum_j a^(2j)]; state = E_theta[a^(2h) P_theta];
    parameter = Var_theta[seasonal_future + a^h m_theta]. Terminal m/P come
    from the same training observations used for the parameter posterior.
    """
    horizons = np.asarray(horizons, dtype=int)
    seasonal_future = np.asarray(seasonal_future, dtype=float)
    if horizons.ndim != 1 or seasonal_future.shape != horizons.shape:
        raise ValueError("horizons and seasonal_future must be matching vectors")
    weights = np.array([node["weight"] for node in grid["nodes"]])
    weights = weights / weights.sum()
    conditional_means, conditional_a, conditional_state = [], [], []
    for node in grid["nodes"]:
        conditional_means.append(seasonal_future + a**horizons * node["filtered_residual_mean"])
        components = forecast_components(a, node["q"], node["r"],
                                         node["filtered_variance"], horizons)
        conditional_a.append(components["aleatoric"])
        conditional_state.append(components["epistemic"])
    conditional_means = np.array(conditional_means)
    mu = weights @ conditional_means
    aleatoric = weights @ np.array(conditional_a)
    state = weights @ np.array(conditional_state)
    parameter = weights @ (conditional_means - mu)**2
    return {"mu": mu, "aleatoric": aleatoric, "state": state,
            "parameter": parameter, "total": aleatoric + state + parameter}
