"""Shared pieces of the LLM anomaly-detection benchmark.

Mirrors the upstream benchmark (Bayes-Works/canari @ feature/global-model,
`experiments/anomaly_detection_lstm.py` and `experiments/utils.py`) with the LSTM
slot filled by a frozen foundation model through canari's `Auxiliary` component.
Everything else -- the sigma_v grid search, the SKF parameter search, the
multi-realization testing scheme and the anomaly magnitudes -- follows upstream.
With `condition: global_finetune` the slot holds upstream's global LSTM instead,
on the same data.

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
MIN_SPLIT_YEARS = 4  # minimum length of train+validation and of test


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


def uses_lstm(config):
    """True when the recurrent slot holds upstream's global LSTM, not a foundation model."""

    return config.get("condition", "llm") == "global_finetune"


def resolve_series(config):
    """The series to benchmark: the config list, or the shared selection of 10."""

    if config.get("series"):
        return list(config["series"])
    return base.select_series()[0]


def evaluation_magnitudes(config):
    """Magnitudes of the multi-realization evaluation.

    Defaults to `slope_search_space`, so that the SKF search can be restricted to
    the slopes the objective can reward while evaluation still covers all of them.
    """

    return config.get("evaluation_magnitudes") or config["slope_search_space"]


# ---- Data ---------------------------------------------------------------------
def prepare_dataset(series, config):
    """Warmup context, train/validation/test splits and standardization.

    Follows upstream `prepare_dataset`: the standardization constants come from the
    training region of the *original* frame (warmup included), the warmup rows are
    then dropped from the modelled frame, and the splits are pinned by timestamp.
    Upstream also attaches a `week_of_year` covariate for the LSTM; the `Auxiliary`
    interface takes no covariates, so the LLM configs set none. The global-LSTM
    configs set `time_covariates: [week_of_year]`, which adds a column to `x` and
    leaves `y`, the splits and the standardization of `y` unchanged.
    """

    from canari import DataProcess

    frame = base.load_series(series)
    warmup_len = int(config["global_warmup_lookback_len"])
    if len(frame) <= warmup_len + 2:
        raise ValueError(f"{series}: too short for a {warmup_len}-step warmup")

    # Ratios are fractions of the modelled rows (after the warmup), so that
    # test_ratio = 0.5 makes train+validation and test the same length.
    num_rows = len(frame) - warmup_len
    validation_idx = warmup_len + int(num_rows * (1.0 - config["validation_ratio"] - config["test_ratio"]))
    test_idx = warmup_len + int(num_rows * (1.0 - config["test_ratio"]))
    if validation_idx <= warmup_len:
        raise ValueError(f"{series}: training window ends before the warmup does")
    min_steps = MIN_SPLIT_YEARS * WEEKS_PER_YEAR
    if test_idx - warmup_len < min_steps or len(frame) - test_idx < min_steps:
        raise ValueError(f"{series}: train+validation and test need {MIN_SPLIT_YEARS} years each")
    validation_start = frame.index[validation_idx]
    test_start = frame.index[test_idx]

    time_covariates = config.get("time_covariates")
    scale_processor = DataProcess(
        data=frame,
        time_covariates=time_covariates,
        validation_start=validation_start,
        test_start=test_start,
        output_col=[0],
    )

    modelled = frame.iloc[warmup_len:].copy()
    data_processor = DataProcess(
        data=modelled,
        time_covariates=time_covariates,
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
def _maybe_zero_variance(predict_fn, config):
    """Ablation `zero_llm_variance`: keep the forecaster's mean, drop its variance.

    With a zero auxiliary prior variance the auxiliary state has no covariance with
    the observation, so its posterior equals its prior: the context fed back to the
    forecaster becomes its own past forecasts, and level/trend/white noise absorb
    every innovation.
    """

    if not config.get("zero_llm_variance"):
        return predict_fn

    def zero_variance(context, horizon):
        mu, var = predict_fn(context, horizon)
        return mu, np.zeros_like(np.asarray(var, dtype=float))

    return zero_variance


def _maybe_feed_residual(model, config):
    """Ablation `feed_observed_residual`: context gets `c_post + w_post`, not `c_post`.

    The observation is `level + auxiliary + white noise` with no separate observation
    noise, so the posterior satisfies `c_post + w_post = y - level_post` and the
    forecaster keeps seeing the observed residual even when the auxiliary state is
    not updated (e.g. under `zero_llm_variance`).
    """

    if not config.get("feed_observed_residual"):
        return

    def update_aux_context(mu_states):
        value = (
            mu_states[model.get_states_index("auxiliary")]
            + mu_states[model.get_states_index("white noise")]
        )
        model.aux_component.update_context(value)

    model.update_aux_context = update_aux_context


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
        predict_fn=_maybe_zero_variance(predict_fn, config),
        horizon=int(config["llm_horizon"]),
        context=dataset["warmup_context"],
    )
    model = Model(LocalTrend(), auxiliary, WhiteNoise(std_error=sigma_v))
    model.auto_initialize_baseline_states(
        dataset["train_data"]["y"][0 : int(config["baseline_init_len"])]
    )
    model.mu_states[model.get_states_index("trend")] = 0.0
    _maybe_feed_residual(model, config)
    return model, auxiliary


# Upstream's variance for the warmup lookback handed to the LSTM output history.
LSTM_WARMUP_VAR = 0.1


def global_lstm_params(config):
    """Weights file for this case, a relative path being from the repository root.

    A `{seed}` placeholder selects one file per seed, as upstream keeps them.
    """

    path = str(config["lstm_global_params"]).format(seed=int(config["seed"]))
    return str(REPO_DIR / path)


def set_lstm_warmup(model, dataset):
    """Hand the standardized warmup to the LSTM as its lookback, as upstream does."""

    context = dataset["warmup_context"]
    model.lstm_output_history.set(context, np.full_like(context, LSTM_WARMUP_VAR))


def build_lstm_model(sigma_v, dataset, config):
    """Normal model for the `global_finetune` condition, ready to be trained.

    `LocalTrend` + `LstmNetwork` + `WhiteNoise`, as upstream. The installed canari
    predates upstream's `LstmNetwork(increase_output_variance=True)`, so it is
    reproduced here: the pretrained weight and bias *means* are kept and the
    *variances* are the fresh initialization, which lets training move the network
    away from the global solution.
    """

    from canari import Model
    from canari.component import LocalTrend, LstmNetwork, WhiteNoise

    lstm = LstmNetwork(
        look_back_len=int(config["lstm_look_back_len"]),
        num_features=int(config["lstm_num_features"]),
        num_layer=int(config["lstm_num_layer"]),
        num_hidden_unit=int(config["num_hidden_unit"]),
        device="cpu",
        num_thread=1,
        manual_seed=int(config["seed"]),
        smoother=False,
    )
    model = Model(LocalTrend(), lstm, WhiteNoise(std_error=sigma_v))

    initial = model.lstm_net.state_dict()
    model.lstm_net.load(filename=global_lstm_params(config))
    loaded = model.lstm_net.state_dict()
    model.lstm_net.load_state_dict(
        {
            name: (loaded[name][0], initial[name][1], loaded[name][2], initial[name][3])
            for name in initial
        }
    )

    model.auto_initialize_baseline_states(
        dataset["train_data"]["y"][0 : int(config["baseline_init_len"])]
    )
    model.mu_states[model.get_states_index("trend")] = 0.0
    return model


def _build_lstm_skf(param, dataset, config):
    """Switching model around the trained LSTM, as upstream's `_build_skf`.

    The normal model is the one saved at the optimal epoch, including its states
    at t=0 (smoothed during training), so it is loaded as is rather than rebuilt.
    """

    import pickle

    from canari import Model, SKF
    from canari.component import LocalAcceleration, LstmNetwork, WhiteNoise

    with open(config["trained_model_path"], "rb") as handle:
        norm_model = Model.load_dict(pickle.load(handle))
    # The LSTM here only gives the abnormal model its state layout; the SKF runs
    # the normal model's network for every transition.
    abnorm_model = Model(
        LocalAcceleration(), LstmNetwork(), WhiteNoise(std_error=float(param["sigma_v"]))
    )
    skf = SKF(
        norm_model=norm_model,
        abnorm_model=abnorm_model,
        std_transition_error=float(param["std_transition_error"]),
        norm_to_abnorm_prob=float(param["norm_to_abnorm_prob"]),
        abnorm_to_norm_prob=float(param["abnorm_to_norm_prob"]),
    )
    set_lstm_warmup(skf.model["norm_norm"], dataset)
    skf.save_initial_states()
    return skf, None


def build_skf(predict_fn, param, dataset, config):
    """Switching model, with the same `Auxiliary` instance in both regimes.

    The SKF queries the auxiliary once per time step and hands the same forecast to
    all four transition models, so the component must be shared, not duplicated.
    """

    from canari import Model, SKF
    from canari.component import Auxiliary, LocalAcceleration, LocalTrend, WhiteNoise

    if uses_lstm(config):
        return _build_lstm_skf(param, dataset, config)

    auxiliary = Auxiliary(
        predict_fn=_maybe_zero_variance(predict_fn, config),
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
    # The SKF feeds the context through norm_norm, a copy of norm_model.
    _maybe_feed_residual(skf.model["norm_norm"], config)
    skf.save_initial_states()
    return skf, auxiliary


def reset(skf, auxiliary):
    """Return the filter to its post-warmup state.

    `Model.set_memory` does not cover the `Auxiliary` context, so restoring the SKF
    states alone would leave the previous realization's context in place and let it
    grow across runs. Upstream's `load_initial_states` is enough for an LSTM because
    `set_memory` does restore `lstm_output_history`; `auxiliary` is then None.
    """

    skf.load_initial_states()
    if auxiliary is not None:
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


def recurrent_predictor(config):
    """The foundation model's predictor, or None when the slot holds the LSTM."""

    if uses_lstm(config):
        return None
    set_model_id(config.get("model_id"))
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


def save_skf_figure(skf, auxiliary, dataset, realization, threshold, path):
    """Filter one realization and save its SKF states and Pr(abnormal) as a PDF.

    Diagnostic only (not a paper figure): standardized space, the injected series
    in place of the clean one, the anomaly onset dashed, the threshold dotted.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from canari import plot_skf_states

    prob, states = skf.filter(data=realization)
    reset(skf, auxiliary)
    fig, axes = plot_skf_states(
        data_processor=dataset["data_processor"],
        states=states,
        model_prob=prob,
        standardization=True,
        plot_observation=False,
    )
    time = dataset["data_processor"].data.index[: len(prob)]
    axes[0].plot(time, realization["y"].ravel(), color="tab:red", lw=0.6, zorder=0)
    onset = time[int(realization["anomaly_timestep"])]
    for ax in axes:
        ax.axvline(onset, color="0.4", ls="--", lw=0.8)
    axes[-1].axhline(threshold, color="0.4", ls=":", lw=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


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
