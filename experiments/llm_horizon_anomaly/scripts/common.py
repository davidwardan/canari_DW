"""Shared pieces of the anomaly-detection H sweep: SKF, anomalies, detection metrics.

The data layer (series selection, warmup standardization, the Chronos-2 predictor)
and the plot style are reused from the forecasting experiment next door, so both
experiments are guaranteed to run on identical series with identical scaling.
"""

import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]


def _load_forecast_common():
    """Load the sibling experiment's `common.py` under an unambiguous name.

    Both experiments call their shared module `common`, so a path-based import is
    what keeps `import common` from resolving to this file.
    """

    path = EXP_DIR.parent / "llm_horizon_degradation" / "scripts" / "common.py"
    spec = importlib.util.spec_from_file_location("forecast_common", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["forecast_common"] = module  # so worker processes can import it too
    spec.loader.exec_module(module)
    return module


base = _load_forecast_common()

WARMUP = base.WARMUP
HORIZONS = base.HORIZONS
SEED = base.SEED
SIGMA_V = base.SIGMA_V

# ---- Experiment configuration -------------------------------------------------
NUM_REALIZATIONS = 10  # anomalous realizations per series, identical across H
MAX_WEEKS_TO_DETECT = 156  # three years
ALARM_THRESHOLD = 0.5
ANOMALY_WINDOW = (0.33, 0.66)  # fractional position of the anomaly onset
WEEKS_PER_YEAR = 52

# ---- Tuning, all on held-out series that are never scored ---------------------
TUNE_SERIES = ["ts43", "ts69", "ts53"]
TUNE_REALIZATIONS = 4
# H=1 has no operating point at all: the only settings that fire, fire ~1.6-5.5
# times a year on clean data whatever the threshold, and every quieter setting
# never detects. So the configuration is tuned at the *lowest horizon that has a
# real operating point*, found by scanning upward, and then frozen for every H.
SCAN_HORIZONS = [1, 3, 6, 13, 26]
MIN_EXCESS_POD = 0.3  # a detector has to beat its own false-alarm process by this
# `norm_to_abnorm_prob` is the decisive lever on false alarms, and it has to reach
# well below canari's 1e-4 default: at 1e-4 and above the filter fires ~1-2 times a
# year on clean data, while 1e-6..1e-5 gives a real operating point. The abnormal
# model's acceleration noise sets sensitivity -- 1e-2 detects fast but raises the
# false-alarm rate sharply at small H.
PARAM_GRID = {
    "norm_to_abnorm_prob": [1e-6, 1e-5, 1e-4],
    "local_trend_std": [1e-4],
    "local_accel_std": [1e-3, 1e-2],
    "std_transition_error": [1e-5, 1e-4],
}
TREND_INIT_VAR = 1e-10  # pins the normal regime's slope at zero; see build_skf
SCREEN_HORIZON = 13  # cheap pre-screen to narrow the grid before tuning at H=1
NUM_FINALISTS = 4
PROVISIONAL_SLOPE_PER_YEAR = 2.0
MAGNITUDE_GRID_PER_YEAR = [1.0, 2.0, 3.0, 4.0]
TARGET_POD = 0.8
# Relaxed from 0.1 after no configuration met that at any horizon tested;
# 0.2/year is ~3 false alarms over the ~17-year record.
MAX_FALSE_ALARMS_PER_YEAR = 0.2

TUNED_PARAMS_FILE = EXP_DIR / "data" / "tuned_params.json"
NUM_WORKERS = 6


# ---- Model --------------------------------------------------------------------
def build_skf(predict_fn, horizon, context, params):
    """Switching model: normal keeps a slowly varying slope, abnormal can change it.

    The trend state is **reset to zero** after initialization. This matters more
    than it looks. `initialize_from_context` seeds the trend with the slope fitted
    to the warmup, and because the auxiliary component can represent any signal, no
    residual ever pushes that slope back: the level integrates it for the rest of
    the record. On `ts43` the warmup slope is +0.0149/week and the level duly ran
    away to +12 over 896 clean steps while the auxiliary went to -12, summing to
    the right observation, so forecasts looked perfect while the decomposition was
    meaningless. On `ts53`, whose warmup slope happens to be -0.00000/week, the
    same model never drifted at all -- which is what identified the cause.

    Worse for this experiment, the runaway rate depended on `H` (the variable under
    study) and at ~0.7 sigma/year it was a sizeable fraction of the anomalies being
    injected. Starting the normal regime from "no drift" is also the right prior
    for a benchmark that is already detrended.
    """

    from canari import Model, SKF
    from canari.component import Auxiliary, LocalAcceleration, LocalTrend, WhiteNoise

    auxiliary = Auxiliary(predict_fn=predict_fn, horizon=horizon)
    noise = WhiteNoise(std_error=SIGMA_V)
    skf = SKF(
        norm_model=Model(LocalTrend(std_error=params["local_trend_std"]), auxiliary, noise),
        abnorm_model=Model(
            LocalAcceleration(std_error=params["local_accel_std"]), auxiliary, noise
        ),
        std_transition_error=params["std_transition_error"],
        norm_to_abnorm_prob=params["norm_to_abnorm_prob"],
    )
    skf.initialize_from_context(context)

    norm_model = skf.model["norm_norm"]
    trend_index = norm_model.get_states_index("trend")
    norm_model.mu_states[trend_index] = 0.0
    norm_model.var_states[trend_index, trend_index] = TREND_INIT_VAR

    skf.save_initial_states()
    return skf, auxiliary


def reset(skf, auxiliary):
    """Return the filter to its post-warmup state.

    `Model.set_memory` does not cover the `Auxiliary` context, so restoring the
    SKF states alone would leave the previous realization's context in place and
    let it grow across runs.
    """

    skf.load_initial_states()
    auxiliary.reset()


# ---- Data ---------------------------------------------------------------------
def split_series(series):
    """Warmup context and the evaluation segment of one series."""

    from canari import DataProcess

    frame = base.load_series(series)
    _, _, _, all_data = base.make_data_processor(frame).get_splits()
    return DataProcess.split_at(all_data, WARMUP)


def make_realizations(filter_data, slope_per_week, num_realizations):
    """One anomaly per realization, drawn identically for every horizon.

    `DataProcess.add_synthetic_anomaly` seeds numpy with a fixed value and draws
    the onsets from the segment length alone, so the same series always yields the
    same onsets whatever `H` is -- the comparison across `H` is paired by design.
    """

    from canari import DataProcess

    return DataProcess.add_synthetic_anomaly(
        filter_data,
        num_samples=num_realizations,
        slope=[slope_per_week],
        anomaly_start=ANOMALY_WINDOW[0],
        anomaly_end=ANOMALY_WINDOW[1],
    )


# ---- Detection metrics --------------------------------------------------------
def alarm_upcrossings(prob, threshold=ALARM_THRESHOLD):
    """Indices where the anomaly probability crosses the threshold from below.

    Consecutive steps above the threshold are one alarm, not many.
    """

    above = prob > threshold
    return np.flatnonzero(above & ~np.concatenate(([False], above[:-1])))


def score_realization(prob, onset, threshold=ALARM_THRESHOLD):
    """Detection and time to detection within three years of the onset.

    An alarm counts only if it *starts* inside the detection window. An alarm that
    was already running when the anomaly began cannot be attributed to it, and
    counting it would score the background false-alarm process as a detection.

    Even so, detection probability has to be read against the chance level: with a
    false-alarm rate `r` and a three-year window, `1 - exp(-3r)` of realizations
    would show an alarm with no anomaly present at all. `summarize` reports that
    chance level and the excess over it.
    """

    window_end = onset + MAX_WEEKS_TO_DETECT
    starts = alarm_upcrossings(prob, threshold)
    in_window = starts[(starts >= onset) & (starts < window_end)]
    return {
        "onset": int(onset),
        "detected": bool(in_window.size),
        "weeks_to_detect": int(in_window[0] - onset) if in_window.size else np.nan,
        "pre_onset_alarm": bool((prob[:onset] > threshold).any()),
        "alarm_active_at_onset": bool(onset > 0 and prob[onset - 1] > threshold),
    }


def score_clean(prob, threshold=ALARM_THRESHOLD):
    """False alarms raised on a series with no injected anomaly."""

    events = alarm_upcrossings(prob, threshold)
    years = len(prob) / WEEKS_PER_YEAR
    return {
        "num_false_alarms": int(events.size),
        "years": round(years, 2),
        "false_alarms_per_year": events.size / years,
        "any_false_alarm": bool(events.size),
    }


def summarize(detections, false_alarms, by="horizon"):
    """Detection probability, time to detection and false-alarm rate per horizon.

    `chance_pod` is what the measured false-alarm rate alone would produce over the
    three-year detection window; `excess_pod` is the detector's contribution above
    it, and is the quantity worth comparing across horizons.
    """

    window_years = MAX_WEEKS_TO_DETECT / WEEKS_PER_YEAR
    false_alarms = false_alarms.copy()
    false_alarms["chance_pod"] = 1 - np.exp(
        -false_alarms["false_alarms_per_year"] * window_years
    )

    detected = detections[detections["detected"]]
    pod = detections.groupby(by)["detected"].mean()
    chance = false_alarms.groupby(by)["chance_pod"].mean()
    summary = pd.DataFrame(
        {
            "pod": pod,
            "chance_pod": chance,
            "excess_pod": pod - chance,
            "num_realizations": detections.groupby(by)["detected"].size(),
            "median_weeks_to_detect": detected.groupby(by)["weeks_to_detect"].median(),
            "mean_weeks_to_detect": detected.groupby(by)["weeks_to_detect"].mean(),
            "pre_onset_alarm_rate": detections.groupby(by)["pre_onset_alarm"].mean(),
            "alarm_active_at_onset_rate": detections.groupby(by)["alarm_active_at_onset"].mean(),
            "false_alarms_per_year": false_alarms.groupby(by)["false_alarms_per_year"].mean(),
            "series_with_false_alarm": false_alarms.groupby(by)["any_false_alarm"].mean(),
        }
    )
    return summary


# ---- Parallel execution -------------------------------------------------------
_PREDICT_FN = None


def _init_worker():
    """One Chronos-2 pipeline per process; the model barely uses more than a core."""

    import torch

    torch.set_num_threads(1)
    global _PREDICT_FN
    _PREDICT_FN = base.make_predict_fn(base.load_chronos())


def make_task(series, horizon, params, slope_per_week, num_realizations, **tags):
    """One unit of work: one series at one horizon with one parameter set."""

    return {
        "series": series,
        "horizon": horizon,
        "params": params,
        "slope_per_week": slope_per_week,
        "num_realizations": num_realizations,
        "tags": tags,
    }


def run_case(task):
    """Filter one case: the clean series first, then every anomalous realization."""

    identity = {"series": task["series"], "horizon": task["horizon"], **task["tags"]}
    context_data, filter_data = split_series(task["series"])
    skf, auxiliary = build_skf(
        _PREDICT_FN, task["horizon"], context_data["y"], task["params"]
    )

    prob, _ = skf.filter(data=filter_data)
    clean = {**identity, **score_clean(prob)}
    reset(skf, auxiliary)

    rows = []
    realizations = make_realizations(
        filter_data, task["slope_per_week"], task["num_realizations"]
    )
    for index, realization in enumerate(realizations):
        prob, _ = skf.filter(data=realization)
        rows.append(
            {
                **identity,
                "realization": index,
                **score_realization(prob, realization["anomaly_timestep"]),
            }
        )
        reset(skf, auxiliary)

    return rows, clean


def run_all(tasks, num_workers=NUM_WORKERS, progress=True, checkpoint_dir=None):
    """Run (series, horizon) cases across processes; each case is independent.

    With `checkpoint_dir`, every case is appended to disk as it finishes. A sweep
    is hours long and holds nothing else in durable form, so without this a failure
    on the last case would discard the whole run.
    """

    detections, false_alarms = [], []
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(name, rows):
        if checkpoint_dir is None:
            return
        path = checkpoint_dir / name
        frame = pd.DataFrame(rows)
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)

    with mp.get_context("spawn").Pool(num_workers, initializer=_init_worker) as pool:
        for done, (rows, clean) in enumerate(pool.imap_unordered(run_case, tasks), 1):
            detections.extend(rows)
            false_alarms.append(clean)
            checkpoint("detections.partial.csv", rows)
            checkpoint("false_alarms.partial.csv", [clean])
            if progress:
                pod = np.mean([row["detected"] for row in rows]) if rows else float("nan")
                print(
                    f"[{done:>3d}/{len(tasks)}] {clean['series']:>16s} "
                    f"H={clean['horizon']:>3d}  POD {pod:.2f}  "
                    f"FA/yr {clean['false_alarms_per_year']:.3f}",
                    flush=True,
                )
    return pd.DataFrame(detections), pd.DataFrame(false_alarms)
