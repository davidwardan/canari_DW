"""Real weekly series with an added trend: frozen vs RSR, with and without EWMA.

See `../planning/README.md`. Stage 1 trains one snapshot per series, seed and snapshot
type (no constraint, or EWMA at one tolerance ratio); every epoch starts from the true
level and trend. Stage 2 runs each arm over the whole series, saves the states and
scores the test split.

Run from the repository root, with the source tree that has `EwmaConstraint`:

    PYTHONPATH=/Users/davidwardan/canari_DW/src python \
        experiments/ewma_real_series_rsr/scripts/run_experiment.py --workers 6
"""

import os

# One thread per worker: the runs are parallelized over processes.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import copy
import hashlib
import json
import logging
import multiprocessing
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pytagi import metric

import canari
from canari import DataProcess, Model
from canari.component import EwmaConstraint, LocalTrend, LstmNetwork, WhiteNoise
from metrics import PERIOD, leakage, rms, score

EXP_DIR = Path(__file__).resolve().parents[1]
# Data are git-ignored: use the first checkout above this one that has them.
DATA_DIR = next(path / "data/hq_benchmark_data/weekly" for path in EXP_DIR.parents
                if (path / "data/hq_benchmark_data/weekly").is_dir())

SERIES = ("ts67", "ts66", "ts57", "ts25", "ts26")
SEEDS = (1, 2, 3, 4, 5)
SLOPE = 0.005  # added trend per week, in the units of weekly_values.csv
TRAIN_LEN = 4 * PERIOD
VALIDATION_LEN = 2 * PERIOD
TEST_START = TRAIN_LEN + VALIDATION_LEN
SIGMA_V_GRID = np.linspace(0.01, 0.3, 5)
MAX_EPOCHS = 50
PATIENCE = 10
NUM_HIDDEN_UNIT = 50
WINDOW_LEN = PERIOD
STRATEGY = "replay_smoothed"
FAST_SPAN, SLOW_SPAN = 2 * PERIOD, 4 * PERIOD
KAPPAS = (4, 2, 1, 0.5, 0.25, 0.125)
LEVEL_VAR, TREND_VAR = 1e-4, 1e-6  # canari's auto-initialization variances
TOY_NUM_STEPS = 900  # `--series toy`: pipeline check with a known pattern
TOY_STD_NOISE = 0.1

SNAPSHOTS = ("none", *(f"ewma_k{kappa:g}" for kappa in KAPPAS))
ARMS = {
    "frozen": {"snapshot": "none", "online": "frozen"},
    "rsr": {"snapshot": "none", "online": "rsr"},
    **{f"frozen_ewma_k{kappa:g}": {"snapshot": f"ewma_k{kappa:g}", "online": "frozen"}
       for kappa in KAPPAS},
    **{f"rsr_ewma_k{kappa:g}": {"snapshot": f"ewma_k{kappa:g}", "online": "rsr"}
       for kappa in KAPPAS},
}
SOURCE_FILES = ("model.py", "rsr.py", "common.py", "component/ewma_component.py",
                "component/lstm_component.py")

log = logging.getLogger("experiment")


def detrended_series(name: str) -> pd.DataFrame:
    """The provider's detrended series, or the toy pattern plus noise."""

    if name == "toy":
        time_step = np.arange(TOY_NUM_STEPS)
        values = (np.sin(2 * np.pi * time_step / PERIOD)
                  + 0.5 * np.sin(4 * np.pi * time_step / PERIOD + np.pi / 4)
                  + np.random.default_rng(0).normal(0, TOY_STD_NOISE, TOY_NUM_STEPS))
        index = pd.date_range("2000-01-02", periods=TOY_NUM_STEPS, freq="7D")
        return pd.DataFrame({"values": values}, index=pd.Index(index, name="date_time"))
    values = pd.read_csv(DATA_DIR / "weekly_values.csv")[name]
    datetimes = pd.read_csv(DATA_DIR / "weekly_datetimes.csv")[name]
    data = pd.DataFrame({"date_time": pd.to_datetime(datetimes), "values": values})
    data = data.dropna(subset=["date_time"]).set_index("date_time")
    if not data.index.is_monotonic_increasing or (data.index.diff()[1:].days != 7).any():
        raise ValueError(f"{name}: expected regular weekly timestamps")
    return data


def with_trend(detrended: pd.DataFrame) -> pd.DataFrame:
    return detrended.assign(values=detrended["values"] + SLOPE * np.arange(len(detrended)))


def true_baseline(detrended: pd.DataFrame) -> tuple:
    """Training mean of the detrended part, and b_t = m + delta t."""

    level = float(np.nanmean(detrended["values"].to_numpy()[:TRAIN_LEN]))
    return level, level + SLOPE * np.arange(len(detrended))


def make_processor(data: pd.DataFrame) -> DataProcess:
    data_processor = DataProcess(
        data=data,
        time_covariates=["week_of_year"],
        validation_start=data.index[TRAIN_LEN],
        test_start=data.index[TEST_START],
        output_col=[0],
    )
    assert data_processor.train_end == TRAIN_LEN
    assert data_processor.test_start == TEST_START
    return data_processor


def scale(data_processor: DataProcess) -> tuple:
    return data_processor.scale_const_mean[0], data_processor.scale_const_std[0]


def initial_baseline(level: float, data_processor: DataProcess) -> tuple:
    """Normalized level and trend one step before week 0 (the first prediction moves it)."""

    y_mean, y_std = scale(data_processor)
    return (level - SLOPE - y_mean) / y_std, SLOPE / y_std


def leak_of(detrended: pd.DataFrame, level: float, data_processor: DataProcess) -> tuple:
    """Leakage of the average training cycle of y - b, in normalized units."""

    cycles = detrended["values"].to_numpy()[:TRAIN_LEN].reshape(-1, PERIOD) - level
    return leakage(np.nanmean(cycles, axis=0) / scale(data_processor)[1], FAST_SPAN, SLOW_SPAN)


def taus_of(snapshot: str, leak: tuple):
    if snapshot == "none":
        return None
    kappa = float(snapshot.removeprefix("ewma_k"))
    return kappa * leak[0], kappa * leak[1]


def build_model(sigma_v: float, seed: int, taus=None) -> Model:
    """`taus=None` builds the model without the constraint component."""

    components = [
        LocalTrend(std_error=0.0),
        LstmNetwork(
            look_back_len=PERIOD,
            num_features=2,
            infer_len=PERIOD,
            num_hidden_unit=NUM_HIDDEN_UNIT,
            num_layer=1,
            manual_seed=seed,
        ),
        WhiteNoise(std_error=sigma_v),
    ]
    if taus is not None:
        components.append(EwmaConstraint(source="lstm", fast_span=FAST_SPAN,
                                         slow_span=SLOW_SPAN, tau_mean=taus[0],
                                         tau_slope=taus[1]))
    return Model(*components)


def set_baseline(model: Model, baseline: tuple) -> None:
    """Known level and trend, independent of the other states."""

    for name, mean, variance in (("level", baseline[0], LEVEL_VAR),
                                 ("trend", baseline[1], TREND_VAR)):
        index = model.get_states_index(name)
        model.mu_states[index] = mean
        model.var_states[index, :] = 0
        model.var_states[:, index] = 0
        model.var_states[index, index] = variance
    model._mu_local_level = baseline[0]


def train_offline(data_processor, sigma_v: float, seed: int, taus, baseline: tuple,
                  max_epochs: int):
    """Train with validation early stopping; every epoch starts from `baseline`."""

    train_data, validation_data, _, _ = data_processor.get_splits()
    validation_obs = validation_data["y"].flatten()
    model = build_model(sigma_v, seed, taus)
    set_baseline(model, baseline)

    history = []
    snapshot = None
    status = {"status": "complete"}
    for epoch in range(max_epochs):
        try:
            mean, std, states = model.lstm_train(
                train_data=train_data,
                validation_data=validation_data,
                data_processor=data_processor,
            )
            log_lik = float(metric.log_likelihood(mean, validation_obs, std))
            if not np.isfinite(log_lik):
                raise FloatingPointError("nonfinite validation log-likelihood")
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
            status = {"status": "numerical_failure", "failure_epoch": epoch,
                      "error": str(error)}
            log.warning("seed %d sigma_v=%.4f epoch=%d: %s", seed, sigma_v, epoch, error)
            break
        history.append(log_lik)
        model.early_stopping(evaluate_metric=log_lik, current_epoch=epoch,
                             max_epoch=max_epochs, mode="max", patience=PATIENCE)
        model.set_memory(states=states, time_step=0)
        set_baseline(model, baseline)  # instead of the smoothed trend of this epoch
        if epoch == model.optimal_epoch and np.isfinite(model.early_stop_metric):
            # Parameters, initial states and LSTM look-back of the best epoch
            snapshot = copy.deepcopy(model.get_dict())
        if model.stop_training:
            break
    best_log_lik = model.early_stop_metric if snapshot is not None else -np.inf
    return best_log_lik, model.optimal_epoch, snapshot, history, status


def start_log(name: str):
    handler = logging.FileHandler(EXP_DIR / "logs" / f"{name}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    return handler


def stop_log(handler) -> None:
    log.removeHandler(handler)
    handler.close()


def run_training(job: dict) -> dict:
    """Stage 1: one snapshot for a (series, snapshot type, seed)."""

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)
    handler = start_log(out_dir.name)
    (out_dir / "config.json").write_text(json.dumps(job, indent=2))
    log.info("START %s", json.dumps(job))
    series, seed = job["series"], job["seed"]
    detrended = detrended_series(series)
    data_processor = make_processor(with_trend(detrended))
    level, _ = true_baseline(detrended)
    baseline = initial_baseline(level, data_processor)
    leak = leak_of(detrended, level, data_processor)
    taus = taus_of(job["snapshot"], leak)

    start_time = time.time()
    candidates, candidate_status = [], []
    for sigma_v in SIGMA_V_GRID:
        best_log_lik, optimal_epoch, snapshot, history, status = train_offline(
            data_processor, float(sigma_v), seed, taus, baseline, job["max_epochs"])
        candidates.append((best_log_lik, float(sigma_v), optimal_epoch, snapshot))
        candidate_status.append({"sigma_v": float(sigma_v), **status})
        pd.DataFrame({"validation_ll": history}).to_csv(
            out_dir / f"validation_history_sigma{sigma_v:.4f}.csv", index_label="epoch")
        log.info("sigma_v=%.4f epoch=%s validation_ll=%s status=%s",
                 sigma_v, optimal_epoch, best_log_lik, status["status"])
    valid = [candidate for candidate in candidates if candidate[3] is not None]
    if not valid:
        raise RuntimeError(f"{out_dir.name}: no valid noise-grid candidate")
    best_log_lik, sigma_v, optimal_epoch, snapshot = max(valid, key=lambda c: c[0])
    selection = {
        "selected_sigma_v": sigma_v, "optimal_epoch": optimal_epoch,
        "validation_ll": best_log_lik, "true_level": level, "initial_baseline": baseline,
        "leak": leak, "taus": taus, "training_seconds": time.time() - start_time,
        "candidates": [{"sigma_v": c[1], "validation_ll": c[0] if np.isfinite(c[0]) else None,
                        "optimal_epoch": c[2]} for c in candidates],
        "candidate_status": candidate_status,
    }
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    with (out_dir / "snapshot.pkl").open("wb") as stream:
        pickle.dump(snapshot, stream)
    log.info("COMPLETE %s", json.dumps(selection))
    stop_log(handler)
    return {"name": out_dir.name, "series": series, "snapshot": job["snapshot"], "seed": seed,
            **{key: value for key, value in selection.items()
               if key not in ("candidates", "candidate_status")}}


def run_full(snapshot: dict, online: str, all_data: dict):
    """Predictions and filtered states over the whole series.

    RSR: the frozen warm-up before the first replay window is reconstructed exactly as
    `rsr_filter` runs it internally, then RSR returns that window and the test split.
    """

    model = Model.load_dict(snapshot)
    if online == "frozen":
        model.lstm_net.num_samples = len(all_data["y"])
        model.lstm_net.eval()
        parts = [model.filter(all_data, train_lstm=False)]
    else:
        prefix_end = TEST_START - WINDOW_LEN
        prefix_model = Model.load_dict(snapshot)
        prefix_model.lstm_net.num_samples = max(prefix_end, WINDOW_LEN + 1)
        prefix_model.lstm_net.eval()
        prefix = {key: values[:prefix_end] for key, values in all_data.items()}
        parts = [prefix_model.filter(prefix, train_lstm=False),
                 model.rsr_filter(data=all_data, start=TEST_START, window_len=WINDOW_LEN,
                                  strategy=STRATEGY, include_warmup=True)]
    hidden = {name: tuple(np.concatenate([getattr(states, getter)(name, "posterior")
                                          for _, _, states in parts])
                          for getter in ("get_mean", "get_std"))
              for name in model.states_name}
    return (np.concatenate([part[0] for part in parts]),
            np.concatenate([part[1] for part in parts]), hidden)


def run_evaluation(job: dict) -> dict:
    """Stage 2: one (series, arm, seed) from its stage-1 snapshot."""

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)
    handler = start_log(out_dir.name)
    (out_dir / "config.json").write_text(json.dumps(job, indent=2))
    log.info("START %s", json.dumps(job))
    series, arm, seed = job["series"], job["arm"], job["seed"]
    detrended = detrended_series(series)
    data = with_trend(detrended)
    data_processor = make_processor(data)
    _, _, _, all_data = data_processor.get_splits()
    y_mean, y_std = scale(data_processor)
    _, baseline = true_baseline(detrended)
    with Path(job["snapshot_path"]).open("rb") as stream:
        snapshot = pickle.load(stream)

    row = {"series": series, "arm": arm, "seed": seed, "run": out_dir.name,
           "status": "complete"}
    start_time = time.time()
    try:
        mean, std, hidden = run_full(snapshot, ARMS[arm]["online"], all_data)
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
        log.warning("FAILED %s", error)
        row.update(status="numerical_failure", error=str(error))
        (out_dir / "metrics.json").write_text(json.dumps(row, indent=2))
        stop_log(handler)
        return row
    row["seconds"] = time.time() - start_time

    # Back to data units; the level carries the standardization mean.
    states = pd.DataFrame({"obs": data["values"].to_numpy(), "mean": mean * y_std + y_mean,
                           "std": std * y_std, "baseline": baseline}, index=data.index)
    for name, (mu, sd) in hidden.items():
        column = name.replace(" ", "_")
        states[f"{column}_mean"] = mu * y_std + (y_mean if name == "level" else 0.0)
        states[f"{column}_std"] = sd * y_std
    states.insert(0, "split", np.where(np.arange(len(states)) < TRAIN_LEN, "train",
                                       np.where(np.arange(len(states)) < TEST_START,
                                                "validation", "test")))
    states.to_csv(out_dir / "full_states.csv")

    test = states.iloc[TEST_START:]
    row["level_rmse"] = rms(test["level_mean"] - test["baseline"])
    row["trend_rmse"] = rms(test["trend_mean"] - SLOPE)
    row.update(score(test["obs"].to_numpy(), test["mean"].to_numpy(), test["std"].to_numpy()))
    (out_dir / "metrics.json").write_text(json.dumps(row, indent=2))
    log.info("COMPLETE %s", json.dumps(row))
    stop_log(handler)
    return row


def source_hashes() -> dict:
    root = Path(canari.__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--series", nargs="+", choices=(*SERIES, "toy"), default=list(SERIES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--tag-suffix", default="", help="e.g. _smoke")
    args = parser.parse_args()

    run_tag = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y%m%d-%H%M%S")
    run_tag += ("_toy" if args.series == ["toy"] else "") + args.tag_suffix
    batch_dir = EXP_DIR / "results" / run_tag
    batch_dir.mkdir(parents=True, exist_ok=False)
    (EXP_DIR / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(EXP_DIR / "logs" / f"{run_tag}.log"),
                                  logging.StreamHandler(sys.stdout)])
    config = {
        "run_tag": run_tag, "series": args.series, "seeds": args.seeds,
        "arms": {arm: ARMS[arm] for arm in args.arms}, "slope": SLOPE, "period": PERIOD,
        "train_len": TRAIN_LEN, "validation_len": VALIDATION_LEN, "test_start": TEST_START,
        "sigma_v_grid": SIGMA_V_GRID.tolist(), "max_epochs": args.max_epochs,
        "patience": PATIENCE, "num_hidden_unit": NUM_HIDDEN_UNIT, "local_trend_std": 0.0,
        "initial_variances": {"level": LEVEL_VAR, "trend": TREND_VAR},
        "rsr_window_len": WINDOW_LEN, "rsr_strategy": STRATEGY, "fast_span": FAST_SPAN,
        "slow_span": SLOW_SPAN, "kappas": KAPPAS, "data_dir": str(DATA_DIR),
        "toy_std_noise": TOY_STD_NOISE, "canari_path": str(Path(canari.__file__).parent),
        "source_sha256": source_hashes(),
    }
    (batch_dir / "config.json").write_text(json.dumps(config, indent=2))
    log.info("Run %s: %s", run_tag, json.dumps(config))

    snapshots = sorted({ARMS[arm]["snapshot"] for arm in args.arms})
    run_dir = EXP_DIR / "results"
    training_jobs = [
        {"run_tag": run_tag, "series": series, "snapshot": kind, "seed": seed,
         "max_epochs": args.max_epochs,
         "out_dir": str(run_dir / f"{run_tag}_train_{series}_{kind}_seed{seed}")}
        for series in args.series for kind in snapshots for seed in args.seeds
    ]
    evaluation_jobs = [
        {"run_tag": run_tag, "series": series, "arm": arm, "seed": seed,
         "snapshot_path": str(run_dir / f"{run_tag}_train_{series}_{ARMS[arm]['snapshot']}"
                                        f"_seed{seed}" / "snapshot.pkl"),
         "out_dir": str(run_dir / f"{run_tag}_{series}_{arm}_seed{seed}")}
        for series in args.series for arm in args.arms for seed in args.seeds
    ]
    (batch_dir / "jobs.json").write_text(json.dumps(
        {"training": training_jobs, "evaluation": evaluation_jobs}, indent=2))
    context = multiprocessing.get_context("spawn")
    with context.Pool(min(args.workers, len(training_jobs))) as pool:
        selections = list(pool.imap_unordered(run_training, training_jobs))
    pd.DataFrame(selections).to_csv(batch_dir / "training.csv", index=False)

    rows = []
    with context.Pool(min(args.workers, len(evaluation_jobs))) as pool:
        for row in pool.imap_unordered(run_evaluation, evaluation_jobs):
            rows.append(row)
            log.info("%s %-18s seed %d  %s  level RMSE=%.4f  trend RMSE=%.2e  CRPS=%.4f",
                     row["series"], row["arm"], row["seed"], row["status"],
                     row.get("level_rmse", np.nan), row.get("trend_rmse", np.nan),
                     row.get("crps", np.nan))
    pd.DataFrame(rows).sort_values(["series", "arm", "seed"]).to_csv(
        batch_dir / "metrics.csv", index=False)
    log.info("Saved %s", batch_dir / "metrics.csv")


if __name__ == "__main__":
    main()
