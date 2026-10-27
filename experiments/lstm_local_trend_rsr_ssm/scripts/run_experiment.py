"""Local-trend RSR benchmark: unconstrained, EWMA- and auxiliary-SSM-constrained LSTM.

Copied from `lstm_local_trend_rsr_ewma`; see `../planning/README.md` for the protocol.
For every series, seed and arm, the LSTM is trained offline on 4 cycles with validation
early stopping and a `sigma_v` grid search, then the test split is filtered either with
frozen LSTM parameters or with Recurrent Smoothing and Replay (RSR), for each strategy.

Run from the repository root:

    python experiments/lstm_local_trend_rsr_ssm/scripts/run_experiment.py --toy
    python experiments/lstm_local_trend_rsr_ssm/scripts/run_experiment.py
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
from scipy.stats import norm

import canari
from canari import DataProcess, Model
from canari.component import EwmaConstraint, LocalTrend, LstmNetwork, WhiteNoise
from ssm_constraint import LocalTrendConstraint, detrended_std, std_error_for_response_time

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
DATA_DIR = ROOT / "data/hq_benchmark_data/weekly"

SERIES = ("ts67", "ts66", "ts57", "ts25", "ts26", "ts27", "ts36", "ts61", "ts62", "ts58")
SEEDS = (1, 2, 3, 4, 5)
ARMS = ("unconstrained", "ewma", "ssm")
SLOPE = 0.005  # added drift per week, in the units of weekly_values.csv
PERIOD = 52
TRAIN_LEN = 4 * PERIOD
VALIDATION_LEN = 2 * PERIOD
SIGMA_V_GRID = np.linspace(0.01, 0.3, 5)
MAX_EPOCHS = 50
PATIENCE = 10
NUM_HIDDEN_UNIT = 50
LOCAL_TREND_STD = 0.0  # source build_model actually used LocalTrend() (zero)
WINDOW_LEN = PERIOD
TREND_DEVIATION_CYCLES = 3
METHODS = ("offline", "rsr_replay_filtered", "rsr_replay_smoothed")
CONSTRAINT = {"fast_span": 2 * PERIOD, "slow_span": 4 * PERIOD,
              "tau_mean": 0.05, "tau_slope": 0.0005 * 24 / PERIOD}
# Auxiliary level's unit-step response reaches 1 - 1/e at the EWMA slow span
SSM_RESPONSE_TIME = 4 * PERIOD
SSM_STD_RATIO = std_error_for_response_time(1.0, SSM_RESPONSE_TIME)

# Toy check: known periodic signal plus the same drift
TOY_NUM_STEPS = 900
TOY_STD_NOISE = 0.1

log = logging.getLogger("experiment")


def load_series(name: str) -> pd.DataFrame:
    """One weekly series with the synthetic constant slope added."""

    values = pd.read_csv(DATA_DIR / "weekly_values.csv")[name]
    datetimes = pd.read_csv(DATA_DIR / "weekly_datetimes.csv")[name]
    data = pd.DataFrame({"date_time": pd.to_datetime(datetimes), "values": values})
    data = data.dropna(subset=["date_time"]).set_index("date_time")
    if not data.index.is_monotonic_increasing or (data.index.diff()[1:].days != 7).any():
        raise ValueError(f"{name}: expected regular weekly timestamps")
    data["values"] += SLOPE * np.arange(len(data))
    return data


def toy_series() -> pd.DataFrame:
    """Synthetic weekly series: two yearly harmonics, white noise, and the drift."""

    rng = np.random.default_rng(0)
    time_step = np.arange(TOY_NUM_STEPS)
    periodic = np.sin(2 * np.pi * time_step / PERIOD) + 0.5 * np.sin(
        4 * np.pi * time_step / PERIOD + np.pi / 4
    )
    values = (
        periodic + SLOPE * time_step + rng.normal(0, TOY_STD_NOISE, TOY_NUM_STEPS)
    )
    index = pd.date_range("2000-01-02", periods=TOY_NUM_STEPS, freq="7D")
    return pd.DataFrame({"values": values}, index=pd.Index(index, name="date_time"))


def make_processor(data: pd.DataFrame) -> DataProcess:
    data_processor = DataProcess(
        data=data,
        time_covariates=["week_of_year"],
        validation_start=data.index[TRAIN_LEN],
        test_start=data.index[TRAIN_LEN + VALIDATION_LEN],
        output_col=[0],
    )
    assert data_processor.train_end == TRAIN_LEN
    assert data_processor.test_start == TRAIN_LEN + VALIDATION_LEN
    return data_processor


def build_model(
    sigma_v: float, seed: int, arm: str = "unconstrained", std_residual: float = 1.0
) -> Model:
    """`std_residual` is the auxiliary residual std of the `ssm` arm only."""

    components = (
        LocalTrend(std_error=LOCAL_TREND_STD),
        LstmNetwork(
            look_back_len=PERIOD,
            num_features=2,
            infer_len=PERIOD,
            num_hidden_unit=NUM_HIDDEN_UNIT,
            num_layer=1,
            manual_seed=seed,
        ),
        WhiteNoise(std_error=sigma_v),
    )
    if arm == "unconstrained":
        return Model(*components)
    if arm == "ewma":
        return Model(*components, EwmaConstraint(source="lstm", **CONSTRAINT))
    if arm == "ssm":
        return Model(*components, LocalTrendConstraint(
            source="lstm", std_error=SSM_STD_RATIO * std_residual, std_residual=std_residual
        ))
    raise ValueError(f"Unknown arm: {arm}")


def load_model(snapshot):
    # Constraints are components, so the snapshot rebuilds them
    return Model.load_dict(copy.deepcopy(snapshot))


def train_offline(data_processor, sigma_v: float, seed: int, arm: str = "unconstrained",
                  std_residual: float = 1.0):
    """Train with validation early stopping; return the best epoch's model snapshot."""

    train_data, validation_data, _, _ = data_processor.get_splits()
    validation_obs = validation_data["y"].flatten()
    model = build_model(sigma_v, seed, arm, std_residual)
    model.auto_initialize_baseline_states(train_data["y"])

    history = []
    snapshot = None
    status = {"status": "complete"}
    for epoch in range(MAX_EPOCHS):
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
            # Constrained CANARI models reject a diverged LSTM's nonfinite moments
            # with a ValueError (LinAlgError is also a ValueError subclass)
            if not isinstance(error, (np.linalg.LinAlgError, FloatingPointError)) \
                    and "must be finite" not in str(error):
                raise
            status = {
                "status": "numerical_failure", "failure_epoch": epoch,
                "error": str(error), "retained_best_epoch": model.optimal_epoch
                if snapshot is not None else None,
            }
            log.warning("seed %d sigma_v=%.4f epoch=%d: %s", seed, sigma_v, epoch, error)
            break
        history.append(log_lik)
        model.early_stopping(
            evaluate_metric=log_lik,
            current_epoch=epoch,
            max_epoch=MAX_EPOCHS,
            mode="max",
            patience=PATIENCE,
        )
        model.set_memory(states=states, time_step=0)
        if epoch == model.optimal_epoch and np.isfinite(model.early_stop_metric):
            # Parameters, initial states and LSTM look-back of the best epoch
            snapshot = copy.deepcopy(model.get_dict())
        if model.stop_training:
            break
    best_log_lik = model.early_stop_metric if snapshot is not None else -np.inf
    return best_log_lik, model.optimal_epoch, snapshot, history, status


def predict_test(
    snapshot: dict, all_data: dict, test_start: int, method: str,
    include_history: bool = False,
):
    """Predictions and filtered states, optionally including train/validation history.

    The prefix uses the selected offline snapshot with frozen parameters; it is a
    diagnostic reconstruction, not the evolving states from the training epochs.
    RSR's first replay window supplies the last WINDOW_LEN validation steps.
    """

    model = load_model(snapshot)
    if method == "offline":
        # Frozen parameters; the state-space states are still filtered.
        model.lstm_net.num_samples = len(all_data["y"])
        model.lstm_net.eval()
        mean, std, states = model.filter(all_data, train_lstm=False)
        test = slice(None) if include_history else slice(test_start, None)
    else:
        mean, std, states = model.rsr_filter(
            data=all_data,
            start=test_start,
            window_len=WINDOW_LEN,
            strategy=method.removeprefix("rsr_"),
            include_warmup=include_history,
        )
        test = slice(None)
    hidden_states = {}
    for name in model.states_name:
        hidden_states[f"{name}_mean"] = states.get_mean(name, "posterior")[test]
        hidden_states[f"{name}_std"] = states.get_std(name, "posterior")[test]
    if include_history and method != "offline":
        prefix_end = test_start - WINDOW_LEN
        if prefix_end:
            prefix_model = load_model(snapshot)
            prefix_model.lstm_net.num_samples = max(prefix_end, WINDOW_LEN + 1)
            prefix_model.lstm_net.eval()
            prefix = {key: values[:prefix_end] for key, values in all_data.items()}
            prefix_mean, prefix_std, prefix_states = prefix_model.filter(
                prefix, train_lstm=False
            )
            mean = np.concatenate([prefix_mean, mean])
            std = np.concatenate([prefix_std, std])
            for name in prefix_model.states_name:
                for moment, getter in (("mean", "get_mean"), ("std", "get_std")):
                    key = f"{name}_{moment}"
                    hidden_states[key] = np.concatenate([
                        getattr(prefix_states, getter)(name, "posterior"),
                        hidden_states[key],
                    ])
    return mean[test], std[test], hidden_states


def score(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    """Accuracy and Gaussian predictive-distribution scores on the finite observations."""

    finite = np.isfinite(obs)
    error = obs[finite] - mean[finite]
    std = std[finite]
    z = error / std
    log_lik = -0.5 * np.log(2 * np.pi * std**2) - 0.5 * z**2
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "ll_mean": float(np.mean(log_lik)),
        "ll_total": float(np.sum(log_lik)),
        "crps": float(np.mean(crps)),
        "coverage_95": float(np.mean(np.abs(z) <= 1.959963984540054)),
        "n_obs": int(finite.sum()),
    }


def trend_deviation(trend: np.ndarray) -> dict:
    """RMS deviation of the filtered trend from the added slope, early and late in test."""

    num_steps = TREND_DEVIATION_CYCLES * PERIOD
    deviation = trend - SLOPE
    return {
        "trend_rmse_first": float(np.sqrt(np.mean(deviation[:num_steps] ** 2))),
        "trend_rmse_last": float(np.sqrt(np.mean(deviation[-num_steps:] ** 2))),
    }


def run_job(job: dict) -> list:
    """Offline training for one series and seed, then every test method."""

    series, seed, arm = job["series"], job["seed"], job["arm"]
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(EXP_DIR / "logs" / f"{out_dir.name}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    (out_dir / "config.json").write_text(json.dumps(job, indent=2))
    log.info("START %s", json.dumps(job))
    data = toy_series() if series == "toy" else load_series(series)
    data.to_csv(out_dir / "series_with_slope.csv")
    data_processor = make_processor(data)
    _, _, _, all_data = data_processor.get_splits()
    test_start = data_processor.test_start
    y_mean = data_processor.scale_const_mean[0]
    y_std = data_processor.scale_const_std[0]
    # Auxiliary residual std from the standardized training observations only
    std_residual = detrended_std(data_processor.get_splits()[0]["y"])

    # Offline training and sigma_v selection on the validation split only
    start_time = time.time()
    candidates = []
    candidate_status = []
    for sigma_v in SIGMA_V_GRID:
        best_log_lik, optimal_epoch, snapshot, history, status = train_offline(
            data_processor, float(sigma_v), seed, arm, std_residual
        )
        candidates.append((best_log_lik, float(sigma_v), optimal_epoch, snapshot))
        candidate_status.append({"sigma_v": float(sigma_v), **status})
        pd.DataFrame({"validation_ll": history}).to_csv(
            out_dir / f"validation_history_sigma{sigma_v:.4f}.csv",
            index_label="epoch",
        )
        log.info("sigma_v=%.4f selected epoch=%s validation_ll=%s status=%s",
                 sigma_v, optimal_epoch, best_log_lik, status["status"])
    valid_candidates = [candidate for candidate in candidates if candidate[3] is not None]
    if not valid_candidates:
        (out_dir / "training_failures.json").write_text(json.dumps(candidate_status, indent=2))
        raise RuntimeError(f"{series} seed {seed}: no valid noise-grid candidate; see {out_dir}")
    best_log_lik, sigma_v, optimal_epoch, snapshot = max(
        valid_candidates, key=lambda candidate: candidate[0]
    )
    selection = {
        "sigma_v_grid": [candidate[1] for candidate in candidates],
        "best_validation_ll": [candidate[0] if np.isfinite(candidate[0]) else None
                               for candidate in candidates],
        "optimal_epoch": [candidate[2] for candidate in candidates],
        "selected_sigma_v": sigma_v,
        "ssm_std_residual": std_residual if arm == "ssm" else None,
        "training_seconds": time.time() - start_time,
        "candidate_status": candidate_status,
    }
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    with (out_dir / "selected_snapshot.pkl").open("wb") as stream:
        pickle.dump(snapshot, stream)

    rows = []
    obs = data["values"].to_numpy()
    splits = np.full(len(data), "test", dtype=object)
    splits[:TRAIN_LEN] = "train"
    splits[TRAIN_LEN:test_start] = "validation"
    for method in METHODS:
        start_time = time.time()
        mean, std, hidden_states = predict_test(
            snapshot, all_data, test_start, method, include_history=True
        )
        seconds = time.time() - start_time
        if len(mean) != len(obs):
            raise RuntimeError(f"{method}: {len(mean)} predictions for {len(obs)} steps")
        # Back to the units of weekly_values.csv
        # (the level carries the standardization mean; the other states are additive)
        mean = mean * y_std + y_mean
        std = std * y_std
        hidden_states = {key: value * y_std for key, value in hidden_states.items()}
        hidden_states["level_mean"] += y_mean
        predictions = pd.DataFrame(
            {"split": splits, "obs": obs, "mean": mean, "std": std, **hidden_states},
            index=data.index,
        )
        predictions.to_csv(out_dir / f"full_history_{method}.csv")
        predictions.iloc[test_start:].drop(columns="split").to_csv(
            out_dir / f"predictions_{method}.csv"
        )
        test_trend = hidden_states["trend_mean"][test_start:]
        rows.append(
            {
                "series": series,
                "seed": seed,
                "arm": arm,
                "run_dir": str(out_dir),
                "method": method,
                "sigma_v": sigma_v,
                "ssm_std_residual": std_residual if arm == "ssm" else np.nan,
                "optimal_epoch": optimal_epoch,
                "validation_ll": best_log_lik,
                **score(obs[test_start:], mean[test_start:], std[test_start:]),
                "trend_mean": float(np.mean(test_trend)),
                **trend_deviation(test_trend),
                "lstm_drift_slope": lstm_drift(hidden_states["lstm_mean"][test_start:]),
                "seconds": seconds,
            }
        )
        log.info("%s %s", method, json.dumps(rows[-1]))
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    log.info("COMPLETE %s", out_dir)
    log.removeHandler(handler)
    handler.close()
    return rows


def canari_fingerprint() -> dict:
    """Location and content hash of the imported CANARI sources."""

    package = Path(canari.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return {"path": str(package), "sha256": digest.hexdigest()}


def lstm_drift(values):
    count = len(values) // PERIOD
    if count < 2:
        return float("nan")
    block_means = np.asarray(values[:count * PERIOD]).reshape(count, PERIOD).mean(axis=1)
    return float(np.polyfit(np.arange(count) * PERIOD, block_means, 1)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--toy", action="store_true", help="run the toy check only")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="initialization seeds")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--series", nargs="+", choices=SERIES, default=list(SERIES))
    args = parser.parse_args()

    run_tag = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y%m%d-%H%M%S") + ("_toy" if args.toy else "_benchmark")
    results_dir = EXP_DIR / "results" / run_tag
    results_dir.mkdir(parents=True, exist_ok=False)
    (EXP_DIR / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(EXP_DIR / "logs" / f"{run_tag}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    series_names = ("toy",) if args.toy else tuple(args.series)
    seeds = tuple(args.seeds)
    config = {
        "run_tag": run_tag,
        "series": list(series_names),
        "seeds": list(seeds),
        "arms": list(args.arms),
        "ewma_constraint": CONSTRAINT,
        "ssm_constraint": {
            "response_time": SSM_RESPONSE_TIME,
            "std_error_over_std_residual": SSM_STD_RATIO,
            "std_residual": "detrended std of standardized training observations",
            "pseudo_observation_noise": "steady-state auxiliary posterior covariance",
        },
        "canari": canari_fingerprint(),
        "source_experiment": "lstm_local_trend_rsr_ewma",
        "slope": SLOPE,
        "period": PERIOD,
        "train_len": TRAIN_LEN,
        "validation_len": VALIDATION_LEN,
        "sigma_v_grid": SIGMA_V_GRID.tolist(),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "num_hidden_unit": NUM_HIDDEN_UNIT,
        "local_trend_std": LOCAL_TREND_STD,
        "rsr_window_len": WINDOW_LEN,
        "methods": list(METHODS),
        "trend_deviation_cycles": TREND_DEVIATION_CYCLES,
        "history_scope": "full_series_selected_snapshot",
        "numerical_failure_policy": "retain_best_valid_epoch_or_exclude_candidate",
        "toy_std_noise": TOY_STD_NOISE if args.toy else None,
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2))
    log.info("Run %s: %s", run_tag, json.dumps(config))

    jobs = [
        {**config, "series": name, "seed": seed, "arm": arm,
         "out_dir": str(EXP_DIR / "results" / f"{run_tag}_{name}_{arm}_seed{seed}")}
        for name in series_names
        for seed in seeds
        for arm in args.arms
    ]
    (results_dir / "jobs.json").write_text(json.dumps(jobs, indent=2))
    rows = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(min(args.workers, len(jobs))) as pool:
        for job_rows in pool.imap_unordered(run_job, jobs):
            rows.extend(job_rows)
            for row in job_rows:
                log.info(
                    "%-5s seed %d %-22s sigma_v=%.4f epoch=%2d  RMSE=%.4f  MAE=%.4f  "
                    "LL=%.3f  CRPS=%.4f  trend=%.5f  (%.0f s)",
                    row["series"] + "/" + row["arm"], row["seed"], row["method"], row["sigma_v"],
                    row["optimal_epoch"], row["rmse"], row["mae"], row["ll_mean"],
                    row["crps"], row["trend_mean"], row["seconds"],
                )
    pd.DataFrame(rows).sort_values(["series", "seed", "arm", "method"]).to_csv(
        results_dir / "metrics.csv", index=False
    )
    log.info("Saved %s", results_dir / "metrics.csv")


if __name__ == "__main__":
    main()
