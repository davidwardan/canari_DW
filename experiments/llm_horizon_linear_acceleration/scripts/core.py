"""Data and state updates for the constant-acceleration H experiment."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = EXP_DIR.parent / "llm_horizon_models" / "results" / "run_20260924_142849" / "config.json"
WARMUP = 52
HORIZONS = [1, 3, 6, 13, 26, 52, 104, 208]
ACCELERATION_PER_YEAR2 = 0.05
ACCELERATION_PER_WEEK2 = ACCELERATION_PER_YEAR2 / 52**2
SIGMA_V = 0.1


def prepare_series(frame):
    """Scale the fixed benchmark on warmup only and add a quadratic trend."""
    values = frame["values"].to_numpy(dtype=float)
    mean = float(np.nanmean(values[:WARMUP]))
    std = float(np.nanstd(values[:WARMUP]))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("The warmup must have positive finite standard deviation.")
    residual = (values - mean) / std
    t = np.arange(len(values), dtype=float)
    acceleration = ACCELERATION_PER_WEEK2
    trend = 0.5 * acceleration * t**2
    transformed = pd.DataFrame({
        "date": frame.index.to_numpy(),
        "y": residual + trend,
        "residual": residual,
        "synthetic_acceleration_trend": trend,
        "synthetic_slope": acceleration * t,
        "synthetic_acceleration": np.full(len(values), acceleration),
    })
    return transformed, {
        "warmup_mean": mean,
        "warmup_std": std,
        "acceleration_per_year2": ACCELERATION_PER_YEAR2,
        "acceleration_per_week2": acceleration,
    }


def load_data():
    """Load the exact ten series selected by the preceding model comparison."""
    path = EXP_DIR.parent / "llm_horizon_degradation" / "scripts" / "common.py"
    spec = importlib.util.spec_from_file_location("acceleration_benchmark_common", path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    records = json.loads(SOURCE_CONFIG.read_text())["series"]
    data, manifest = {}, []
    for record in records:
        series = record["series"]
        transformed, scale = prepare_series(common.load_series(series))
        data[series] = transformed
        manifest.append({**record, **scale})
    return data, manifest


def build_model(y_warmup, predict_fn, horizon, with_aux=True):
    """Initialize LocalAcceleration from a warmup quadratic OLS fit."""
    from canari import Model
    from canari.component import Auxiliary, LocalAcceleration, WhiteNoise

    y = np.asarray(y_warmup, dtype=float).reshape(-1)
    time = np.arange(len(y), dtype=float) - (len(y) - 1)
    design = np.column_stack((np.ones(len(y)), time, 0.5 * time**2))
    observed = np.isfinite(y)
    if observed.sum() <= 3:
        raise ValueError("Quadratic initialization needs at least four observations.")
    fit_design, fit_y = design[observed], y[observed]
    coef = np.linalg.lstsq(fit_design, fit_y, rcond=None)[0]
    residual = y - design @ coef
    residual_variance = max(
        float(np.sum(residual[observed] ** 2) / (observed.sum() - 3)), SIGMA_V**2
    )
    covariance = residual_variance * np.linalg.inv(fit_design.T @ fit_design)
    components = [LocalAcceleration(std_error=0, mu_states=coef.tolist())]
    if with_aux:
        components.append(Auxiliary(predict_fn=predict_fn, horizon=horizon, context=residual))
    components.append(WhiteNoise(std_error=SIGMA_V))
    model = Model(*components)
    initial_covariance = model.var_states.copy()
    initial_covariance[:3, :3] = covariance
    model.set_states(model.mu_states, initial_covariance)
    return model


def advance(model, y):
    """Forecast before y, assimilate y, and append posterior residual state."""
    mu, variance, prior, prior_covariance = model.forward()
    level = model.get_states_index("level")
    slope = model.get_states_index("trend")
    acceleration = model.get_states_index("acceleration")
    white = model.get_states_index("white noise")
    auxiliary = model.get_states_index("auxiliary")
    result = {
        "mu": float(mu.item()),
        "var": float(variance.item()),
        "level_prior": float(prior[level, 0]),
        "slope_prior": float(prior[slope, 0]),
        "acceleration_prior": float(prior[acceleration, 0]),
        "trend_variance": float(prior_covariance[level, level]),
        "aux_mu": 0.0 if auxiliary is None else float(prior[auxiliary, 0]),
        "aux_variance": 0.0 if auxiliary is None else float(prior_covariance[auxiliary, auxiliary]),
        "white_variance": float(prior_covariance[white, white]),
    }
    component_variance = result["trend_variance"] + result["aux_variance"] + result["white_variance"]
    if not np.isclose(result["var"], component_variance, rtol=1e-10, atol=1e-12):
        raise AssertionError("Predictive variance does not equal component sum.")
    if np.isnan(y):
        posterior, posterior_covariance = prior.copy(), prior_covariance.copy()
        model.mu_states_posterior = posterior
        model.var_states_posterior = posterior_covariance
    else:
        _, _, posterior, posterior_covariance = model.backward(float(y))
    if model.aux_component is not None:
        model.update_aux_context(posterior)
    model.set_states(posterior, posterior_covariance)
    result.update(
        level_posterior=float(posterior[level, 0]),
        slope_posterior=float(posterior[slope, 0]),
        acceleration_posterior=float(posterior[acceleration, 0]),
    )
    return result
