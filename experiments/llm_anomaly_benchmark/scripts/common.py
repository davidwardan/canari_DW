"""Shared pieces of the LLM anomaly-detection benchmark.

Mirrors the upstream benchmark (Bayes-Works/canari @ feature/global-model,
`experiments/anomaly_detection_lstm.py` and `experiments/utils.py`) with the LSTM
slot filled by a frozen foundation model through canari's `Auxiliary` component.
Everything else -- the sigma_v grid search, the SKF parameter search, the
multi-realization testing scheme and the anomaly magnitudes -- follows upstream.

Two upstream helpers are reimplemented here rather than called, because the
installed canari predates them:

- `SKF.detect_synthetic_anomaly` upstream accepts `synthetic_data`, `n_jobs` and
  `test_only` and returns time-to-detection. `detect_synthetic_anomaly` below
  reproduces its counting rules exactly, including the false-alarm definition.
- `SKF(likelihood_covariance_floor=...)` does not exist locally. Upstream defaults
  it to 0.0, so omitting it is equivalent to the default.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm as _norm

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EXP_DIR.parents[1]
DEFAULT_CONFIG = EXP_DIR / "config" / "benchmark.yaml"


def _load_forecast_common():
    """Reuse the forecasting experiment's series selection and Chronos predictor.

    Both experiments call their shared module `common`, so a path-based import is
    what keeps `import common` from resolving to this file.
    """

    path = EXP_DIR.parent / "llm_horizon_degradation" / "scripts" / "common.py"
    spec = importlib.util.spec_from_file_location("forecast_common", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["forecast_common"] = module
    spec.loader.exec_module(module)
    return module


base = _load_forecast_common()

WEEKS_PER_YEAR = 52.0


def load_config(path=None):
    """Read a config, resolving anything that must adapt to the host machine."""

    import os

    with open(path or DEFAULT_CONFIG, "r") as handle:
        config = yaml.safe_load(handle)
    if not config.get("num_workers"):
        # Leave two cores for the OS; the foundation model is memory-bandwidth
        # bound, so oversubscribing buys nothing.
        config["num_workers"] = max(1, (os.cpu_count() or 4) - 2)
    return config


def resolve_series(config):
    """The series to benchmark: the config list, or the shared selection of 10."""

    if config.get("series"):
        return list(config["series"])
    return base.select_series()[0]


# ---- Data ---------------------------------------------------------------------
def prepare_dataset(series, config):
    """Warmup context, train/validation/test splits and standardization.

    Follows upstream `prepare_dataset`: the standardization constants come from the
    training region of the *original* frame (warmup included), the warmup rows are
    then dropped from the modelled frame, and the splits are pinned by timestamp.
    Upstream also attaches a `week_of_year` covariate for the LSTM; the `Auxiliary`
    interface takes no covariates, so none is added here.
    """

    from canari import DataProcess

    frame = base.load_series(series)
    warmup_len = int(config["global_warmup_lookback_len"])
    if len(frame) <= warmup_len + 2:
        raise ValueError(f"{series}: too short for a {warmup_len}-step warmup")

    num_rows = len(frame)
    validation_idx = int(num_rows * (1.0 - config["validation_ratio"] - config["test_ratio"]))
    test_idx = int(num_rows * (1.0 - config["test_ratio"]))
    if validation_idx <= warmup_len:
        raise ValueError(f"{series}: training window ends before the warmup does")
    validation_start = frame.index[validation_idx]
    test_start = frame.index[test_idx]

    scale_processor = DataProcess(
        data=frame,
        validation_start=validation_start,
        test_start=test_start,
        output_col=[0],
    )

    modelled = frame.iloc[warmup_len:].copy()
    data_processor = DataProcess(
        data=modelled,
        validation_start=validation_start,
        test_start=test_start,
        output_col=[0],
        scale_const_mean=scale_processor.scale_const_mean,
        scale_const_std=scale_processor.scale_const_std,
    )
    train_data, validation_data, test_data, all_data = data_processor.get_splits()
    train_val = data_processor.get_splits(split="train_val")

    warmup_values = frame.iloc[:warmup_len, 0].to_numpy(dtype=float)
    warmup_context = (
        warmup_values - scale_processor.scale_const_mean[0]
    ) / scale_processor.scale_const_std[0]

    return {
        "series": series,
        "data_processor": data_processor,
        "train_data": train_data,
        "validation_data": validation_data,
        "test_data": test_data,
        "all_data": all_data,
        "train_val": train_val,
        "warmup_context": warmup_context,
    }


def years_spanned(data_processor, end_index):
    """Calendar years from the first modelled step to an exclusive split bound."""

    index = data_processor.data.index
    last = index[min(max(end_index - 1, 0), len(index) - 1)]
    return (last - index[data_processor.train_start]).days / 365.25


# ---- Model --------------------------------------------------------------------
def build_model(predict_fn, sigma_v, dataset, config):
    """Normal model: `LocalTrend` + `Auxiliary` + `WhiteNoise`.

    Upstream builds `LocalTrend() + LstmNetwork() + WhiteNoise(sigma_v)` and then
    forces the trend state to zero after initializing the baseline. That zeroing is
    load-bearing when the recurrent slot holds a component that can represent any
    signal: otherwise the trend keeps the slope fitted to the initialization window,
    nothing ever corrects it, and the level integrates it into an unbounded drift.
    """

    from canari import Model
    from canari.component import Auxiliary, LocalTrend, WhiteNoise

    auxiliary = Auxiliary(
        predict_fn=predict_fn,
        horizon=int(config["llm_horizon"]),
        context=dataset["warmup_context"],
    )
    model = Model(LocalTrend(), auxiliary, WhiteNoise(std_error=sigma_v))
    model.auto_initialize_baseline_states(
        dataset["train_data"]["y"][0 : int(config["baseline_init_len"])]
    )
    model.mu_states[model.get_states_index("trend")] = 0.0
    return model, auxiliary


def build_skf(predict_fn, param, dataset, config):
    """Switching model, with the same `Auxiliary` instance in both regimes.

    The SKF queries the auxiliary once per time step and hands the same forecast to
    all four transition models, so the component must be shared, not duplicated.
    """

    from canari import Model, SKF
    from canari.component import Auxiliary, LocalAcceleration, LocalTrend, WhiteNoise

    auxiliary = Auxiliary(
        predict_fn=predict_fn,
        horizon=int(config["llm_horizon"]),
        context=dataset["warmup_context"],
    )
    sigma_v = float(param["sigma_v"])
    norm_model = Model(LocalTrend(), auxiliary, WhiteNoise(std_error=sigma_v))
    abnorm_model = Model(LocalAcceleration(), auxiliary, WhiteNoise(std_error=sigma_v))
    norm_model.auto_initialize_baseline_states(
        dataset["train_data"]["y"][0 : int(config["baseline_init_len"])]
    )
    norm_model.mu_states[norm_model.get_states_index("trend")] = 0.0

    skf = SKF(
        norm_model=norm_model,
        abnorm_model=abnorm_model,
        std_transition_error=float(param["std_transition_error"]),
        norm_to_abnorm_prob=float(param["norm_to_abnorm_prob"]),
        abnorm_to_norm_prob=float(param["abnorm_to_norm_prob"]),
    )
    skf.save_initial_states()
    return skf, auxiliary


def reset(skf, auxiliary):
    """Return the filter to its post-warmup state.

    `Model.set_memory` does not cover the `Auxiliary` context, so restoring the SKF
    states alone would leave the previous realization's context in place and let it
    grow across runs. Upstream's `load_initial_states` is enough for an LSTM because
    `set_memory` does restore `lstm_output_history`.
    """

    skf.load_initial_states()
    auxiliary.reset()


# ---- Foundation model ---------------------------------------------------------
_PREDICT_FN = None
_MODEL_ID = None


def set_model_id(model_id):
    """Select the foundation model for this process, reloading if it changed.

    Ray trial processes and pool workers are started without the config, so each
    entry point sets this from `config["model_id"]` before asking for a predictor.
    Without it every process would silently fall back to the shared default.
    """

    global _MODEL_ID, _PREDICT_FN
    if model_id and model_id != _MODEL_ID:
        _MODEL_ID = model_id
        _PREDICT_FN = None


def predict_fn():
    """The process-local predictor, loading it on first use."""

    global _PREDICT_FN
    if _PREDICT_FN is None:
        import torch

        torch.set_num_threads(1)
        _PREDICT_FN = base.make_predict_fn(base.load_chronos(_MODEL_ID))
    return _PREDICT_FN


def init_worker(model_id=None):
    """Pool initializer: one pipeline per process, loaded up front."""

    set_model_id(model_id)
    return predict_fn()


# ---- Metrics ------------------------------------------------------------------
def crps_gaussian(mu, std, obs):
    """Mean CRPS of a Gaussian predictive distribution, as upstream computes it."""

    z = (obs - mu) / std
    return float(
        np.nanmean(
            std * (z * (2 * _norm.cdf(z) - 1) + 2 * _norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        )
    )


def validation_metrics(mu, std, observations, data_processor):
    """Validation log-likelihood, RMSE and CRPS in the original units."""

    from pytagi import Normalizer as normalizer

    output_col = data_processor.output_col
    mu = normalizer.unstandardize(
        mu,
        data_processor.scale_const_mean[output_col],
        data_processor.scale_const_std[output_col],
    )
    std = normalizer.unstandardize_std(std, data_processor.scale_const_std[output_col])
    finite = np.isfinite(observations)
    return {
        "validation_log_likelihood": float(
            np.nanmean(_norm.logpdf(observations[finite], mu[finite], std[finite]))
        ),
        "validation_rmse": float(np.sqrt(np.nanmean((mu - observations) ** 2))),
        "validation_crps": crps_gaussian(mu, std, observations),
    }


# ---- Testing scheme -----------------------------------------------------------
def make_realizations(data, magnitude_per_year, num_samples, anomaly_start, anomaly_end):
    """One anomaly per realization, injected both up and down at each magnitude.

    `DataProcess.add_synthetic_anomaly` seeds numpy with a fixed value, so the same
    series and magnitude always yield the same onsets -- realizations are shared
    across whatever is being compared.
    """

    from canari import DataProcess

    slope = magnitude_per_year / WEEKS_PER_YEAR
    return DataProcess.add_synthetic_anomaly(
        data,
        num_samples=int(num_samples),
        slope=[slope, -slope],
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end,
    )


def detect_synthetic_anomaly(
    skf,
    auxiliary,
    data,
    synthetic_data,
    threshold,
    max_timestep_to_detect,
    test_only=False,
    anomaly_start=0.0,
):
    """Reproduces upstream `SKF.detect_synthetic_anomaly`'s counting rules.

    Note two upstream conventions that differ from the usual ones:

    - **False alarms are counted per time step**, not per alarm episode: a run of
      40 consecutive steps above the threshold counts as 40 false alarms. The rate
      is that count divided by the calendar span of the data.
    - **Any step above the threshold inside the detection window counts**, including
      an alarm that was already running when the anomaly began.

    Returns `(detection_rate, num_false_alarms, (ttd_mean, ttd_std))`, with the time
    to detection in steps.
    """

    num_timesteps = len(data["y"])

    clean_prob, _ = skf.filter(data=data)
    reset(skf, auxiliary)
    if test_only:
        clean_prob = clean_prob[int(anomaly_start * len(clean_prob)) :]
    num_false_alarms = int(np.sum(clean_prob > threshold))

    num_detected = 0
    time_to_detection = []
    for realization in synthetic_data:
        prob, _ = skf.filter(data=realization)
        window_start = int(realization["anomaly_timestep"])
        window_end = (
            num_timesteps
            if max_timestep_to_detect is None
            else window_start + int(max_timestep_to_detect)
        )
        hits = np.flatnonzero(prob[window_start:window_end] > threshold)
        if hits.size:
            num_detected += 1
            time_to_detection.append(int(hits[0]))
        reset(skf, auxiliary)

    detection_rate = num_detected / len(synthetic_data)
    if time_to_detection:
        values = np.asarray(time_to_detection, dtype=float)
        moments = (float(np.nanmean(values)), float(np.nanstd(values)))
    else:
        moments = (float("nan"), float("nan"))
    return detection_rate, num_false_alarms, moments
