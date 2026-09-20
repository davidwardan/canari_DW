"""Shared pieces of the `H`-sweep experiment: data, model, metrics, plot style."""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EXP_DIR.parents[1]
WEEKLY_DIR = REPO_DIR / "data" / "hq_benchmark_data" / "weekly"

# ---- Experiment configuration -------------------------------------------------
WARMUP = 52  # weeks handed to Chronos-2 as context, never filtered, never scored
HORIZONS = [1, 3, 6, 13, 26, 52, 104, 208]
SIGMA_V = 0.1  # WhiteNoise std, warmup-standardized units, same for every run
MODEL_ID = "amazon/chronos-2"
SEED = 0

# Selection rule for the 10 series (see planning/README.md)
MIN_SPAN = 500
MAX_GAPS = 5
MAX_ABS_CORR = 0.9
NUM_SERIES = 10

# Quantile levels bracketing +/- 1 sigma of a Gaussian
QUANTILE_LEVELS = [0.1587, 0.8413]

# ---- Plot style ---------------------------------------------------------------
SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (6.5, 3.5)


def use_paper_style():
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
            "lines.linewidth": 1,
            "figure.figsize": SINGLE_COL,
            "font.size": 9,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig, name, figures_dir=EXP_DIR / "figures"):
    """Save to both .pdf and .pgf, as the project style requires."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".pgf"):
        fig.savefig(figures_dir / f"{name}{suffix}", bbox_inches="tight")


# ---- Data ---------------------------------------------------------------------
def _read_weekly():
    """The processed benchmark files: detrended and globally standardized.

    The global standardization is harmless here -- every series is re-standardized
    on its own warmup below, and that is invariant to any affine map of the input,
    so the file's constants cancel exactly.
    """

    values = pd.read_csv(WEEKLY_DIR / "weekly_values.csv")
    times = pd.read_csv(WEEKLY_DIR / "weekly_datetimes.csv")
    return values, times


def _spans(values):
    """Per-series first/last observed row and number of internal missing values."""

    rows = []
    for name in values.columns:
        observed = np.flatnonzero(values[name].notna().to_numpy())
        if observed.size == 0:
            continue
        start, end = observed[0], observed[-1] + 1
        rows.append((name, start, end, end - start, end - start - observed.size))
    return pd.DataFrame(rows, columns=["series", "start", "end", "span", "gaps"])


def select_series():
    """The 10 series of the experiment: longest first, near-duplicates dropped."""

    values, _ = _read_weekly()
    spans = _spans(values)
    eligible = spans[(spans["span"] >= MIN_SPAN) & (spans["gaps"] <= MAX_GAPS)]
    eligible = eligible.sort_values("span", ascending=False)

    corr = values[eligible["series"]].corr().abs()
    selected = []
    for name in eligible["series"]:
        if all(corr.loc[name, other] < MAX_ABS_CORR for other in selected):
            selected.append(name)
        if len(selected) == NUM_SERIES:
            break
    if len(selected) < NUM_SERIES:
        raise RuntimeError(f"only {len(selected)} series pass the selection rule")

    manifest = eligible.set_index("series").loc[selected].reset_index()
    return selected, manifest


def load_series(name):
    """One series as a DataFrame indexed by date, trimmed to its observed span."""

    values, times = _read_weekly()
    observed = np.flatnonzero(values[name].notna().to_numpy())
    start, end = observed[0], observed[-1] + 1
    frame = pd.DataFrame(
        {"values": values[name].to_numpy()[start:end]},
        index=pd.to_datetime(times[name]).to_numpy()[start:end],
    )
    frame.index.name = "date_time"
    return frame


def make_data_processor(frame):
    """Standardize on the warmup only, and split warmup / evaluation at `WARMUP`."""

    from canari import DataProcess

    warmup = frame["values"].to_numpy()[:WARMUP]
    return DataProcess(
        data=frame,
        validation_start=frame.index[WARMUP],
        test_start=frame.index[WARMUP],
        output_col=[0],
        standardization=True,
        scale_const_mean=[np.nanmean(warmup)],
        scale_const_std=[np.nanstd(warmup)],
    )


# ---- Foundation model ---------------------------------------------------------
def load_chronos(model_id=None):
    from chronos import BaseChronosPipeline

    torch.manual_seed(SEED)
    return BaseChronosPipeline.from_pretrained(
        model_id or MODEL_ID, device_map="cpu", dtype=torch.float32
    )


def make_predict_fn(pipeline, call_counter=None):
    """`(context, horizon) -> (mu, var)` for the `Auxiliary` component."""

    def predict_fn(context, horizon):
        if call_counter is not None:
            call_counter.append(len(context))
        quantiles, mean = pipeline.predict_quantiles(
            inputs=[torch.tensor(np.asarray(context), dtype=torch.float32)],
            prediction_length=horizon,
            quantile_levels=QUANTILE_LEVELS,
        )
        # Chronos-2 returns a list of (n_variates, horizon) tensors; Chronos-Bolt
        # returns a (batch, horizon) tensor. Squeeze the variate axis only when
        # it is there, so one predictor works for both families.
        mu, q = mean[0], quantiles[0]
        if mu.ndim == 2:
            mu, q = mu[0], q[0]
        mu = mu.cpu().numpy()  # (horizon,)
        q = q.cpu().numpy()  # (horizon, 2)
        std = (q[:, 1] - q[:, 0]) / 2.0
        return mu, np.maximum(std**2, 1e-6)

    return predict_fn


def build_model(predict_fn, horizon, context):
    from canari import Model
    from canari.component import Auxiliary, WhiteNoise

    return Model(
        Auxiliary(predict_fn=predict_fn, horizon=horizon, context=context),
        WhiteNoise(std_error=SIGMA_V),
    )


# ---- Metrics ------------------------------------------------------------------
def gaussian_crps(mu, std, y):
    """Closed-form CRPS of a Gaussian predictive distribution, per observation."""

    z = (y - mu) / std
    return std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def score(mu, std, y):
    """Point and distributional scores, ignoring missing observations."""

    observed = ~np.isnan(y)
    mu, std, y = mu[observed], std[observed], y[observed]
    error = mu - y
    return {
        "num_scored": int(observed.sum()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "log_lik": float(np.mean(norm.logpdf(y, loc=mu, scale=std))),
        "crps": float(np.mean(gaussian_crps(mu, std, y))),
    }


# ---- Filter driver ------------------------------------------------------------
def run_filter(frame, horizon, predict_fn):
    """Filter one series with re-forecast interval `horizon`.

    The first `WARMUP` observations seed the `Auxiliary` context and are not
    filtered. Returns the evaluation segment with the one-step-ahead predictive
    moments and, per step, its lead time within the current forecast block.
    """

    from canari import DataProcess

    data_processor = make_data_processor(frame)
    _, _, _, all_data = data_processor.get_splits()
    context_data, filter_data = DataProcess.split_at(all_data, WARMUP)

    model = build_model(predict_fn, horizon, context_data["y"].flatten())
    mu_preds, std_preds, _ = model.filter(data=filter_data)

    return pd.DataFrame(
        {
            "date": filter_data["time"],
            "y": filter_data["y"].flatten(),
            "mu": mu_preds,
            "std": std_preds,
            "lead": np.arange(len(mu_preds)) % horizon + 1,
        }
    )
