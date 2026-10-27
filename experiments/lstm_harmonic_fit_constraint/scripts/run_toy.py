"""Toy: harmonic-fit vs EWMA constraint on online RSR learning, with known components.

See `../planning/README.md`. Per seed: one unconstrained offline training with
validation selection, frozen filtering up to the first RSR window, then for each arm
the constraint is added with warm-started memories and RSR filters the test split.

Run from the repository root:

    python experiments/lstm_harmonic_fit_constraint/scripts/run_toy.py
"""

import os

# One thread per worker: the runs are parallelized over processes.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import copy
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

EXP_DIR = Path(__file__).resolve().parents[1]
# Toy series, offline training and scores are shared with the SSM-constraint experiment
sys.path.insert(0, str(EXP_DIR.parent / "lstm_local_trend_rsr_ssm" / "scripts"))

import run_experiment as base  # noqa: E402
from canari import Model  # noqa: E402
from canari.common import create_block_diag  # noqa: E402
from canari.component import EwmaConstraint  # noqa: E402
from harmonic_constraint import HarmonicTrendConstraint, discounted_fit  # noqa: E402

SEEDS = (1, 2, 3, 4, 5)
ARMS = ("unconstrained", "ewma", "harmonic")
METHODS = ("rsr_replay_filtered", "rsr_replay_smoothed")
HARMONIC = {"period": base.PERIOD, "num_harmonics": 2, "span": 4 * base.PERIOD,
            "tau_level": base.CONSTRAINT["tau_mean"], "tau_slope": base.CONSTRAINT["tau_slope"]}
LAST = 3 * base.PERIOD  # final weeks for tracking errors and readings

log = logging.getLogger("toy")


def true_components() -> dict:
    """Data-unit components of `base.toy_series()`."""

    time_step = np.arange(base.TOY_NUM_STEPS)
    rng = np.random.default_rng(0)
    components = {
        "level": base.SLOPE * time_step,
        "trend": np.full(len(time_step), base.SLOPE),
        "seasonal": np.sin(2 * np.pi * time_step / base.PERIOD)
        + 0.5 * np.sin(4 * np.pi * time_step / base.PERIOD + np.pi / 4),
        "noise": rng.normal(0, base.TOY_STD_NOISE, len(time_step)),
    }
    total = components["level"] + components["seasonal"] + components["noise"]
    assert np.allclose(total, base.toy_series()["values"])
    return components


def make_constraint(arm: str):
    if arm == "ewma":
        return EwmaConstraint(source="lstm", **base.CONSTRAINT)
    return HarmonicTrendConstraint(source="lstm", **HARMONIC)


def frozen_carry(snapshot: dict, data: dict, end: int):
    """Frozen filtering of steps [0, end); returns the carried snapshot and LSTM means."""

    model = Model.load_dict(copy.deepcopy(snapshot))
    model.lstm_net.num_samples = end
    model.lstm_net.eval()
    _, _, states = model.filter({key: value[:end] for key, value in data.items()},
                                train_lstm=False)
    # Resume from the recurrent state of the last carried step, as rsr_filter does
    model.lstm_net.set_lstm_states(model.lstm_net.get_lstm_states(end - 1))
    return model.get_dict(), states.get_mean("lstm", "posterior")


def add_constraint(carried: dict, arm: str, lstm_history: np.ndarray) -> dict:
    """Append the arm's constraint, with memories warm-started from the LSTM history."""

    snapshot = copy.deepcopy(carried)
    if arm == "unconstrained":
        return snapshot
    component = make_constraint(arm)
    if arm == "ewma":
        memory = np.zeros(2)
        for value in lstm_history:
            memory = component.rhos * memory + component.weights * value
    else:
        memory = discounted_fit(lstm_history, component.period, component.num_harmonics,
                                component.rho)
    snapshot["components"][component.component_name] = component
    snapshot["mu_states"] = np.vstack([snapshot["mu_states"], memory.reshape(-1, 1)])
    # Known memories: zero variance and zero covariance with every other state
    snapshot["var_states"] = create_block_diag(
        snapshot["var_states"], np.zeros((len(memory), len(memory)))
    )
    return snapshot


def readings(states, model: Model) -> dict:
    """Moments of the constrained level and slope readings."""

    if not model.ewma_constraints:
        return {}
    rows, _ = model.ewma_constraints[0].constraint_observation(model.states_name)
    mean = np.array(states.mu_posterior)[:, :, 0] @ rows.T
    var = np.einsum("ij,tjk,ik->ti", rows, np.array(states.var_posterior), rows)
    return {"reading_level": (mean[:, 0], np.sqrt(var[:, 0])),
            "reading_slope": (mean[:, 1], np.sqrt(var[:, 1]))}


def ripple(values: np.ndarray) -> float:
    """Mean within-cycle standard deviation over the last complete cycles."""

    return float(np.asarray(values[-LAST:]).reshape(-1, base.PERIOD).std(axis=1).mean())


def run_seed(job: dict) -> list:
    seed, out_dir = job["seed"], Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(EXP_DIR / "logs" / f"{out_dir.name}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    (out_dir / "config.json").write_text(json.dumps(job, indent=2))
    log.info("START %s", json.dumps(job))

    data_processor = base.make_processor(base.toy_series())
    _, _, _, all_data = data_processor.get_splits()
    test_start = data_processor.test_start
    first_window = test_start - base.WINDOW_LEN
    y_mean = data_processor.scale_const_mean[0]
    y_std = data_processor.scale_const_std[0]
    truth = true_components()

    # Unconstrained offline training, shared by every arm
    candidates, statuses = [], []
    for sigma_v in base.SIGMA_V_GRID:
        best_ll, epoch, snapshot, _, status = base.train_offline(
            data_processor, float(sigma_v), seed, "unconstrained"
        )
        statuses.append({"sigma_v": float(sigma_v), **status})
        if snapshot is not None:
            candidates.append((best_ll, float(sigma_v), epoch, snapshot))
        log.info("sigma_v=%.4f epoch=%s validation_ll=%s status=%s",
                 sigma_v, epoch, best_ll, status["status"])
    best_ll, sigma_v, epoch, snapshot = max(candidates, key=lambda candidate: candidate[0])
    (out_dir / "selection.json").write_text(json.dumps(
        {"selected_sigma_v": sigma_v, "optimal_epoch": epoch, "validation_ll": best_ll,
         "candidate_status": statuses}, indent=2))
    with (out_dir / "selected_snapshot.pkl").open("wb") as stream:
        pickle.dump(snapshot, stream)

    carried, lstm_history = frozen_carry(snapshot, all_data, first_window)
    online_data = {key: value[first_window:] for key, value in all_data.items()}
    test = slice(test_start, None)
    obs = base.toy_series()["values"].to_numpy()[test]
    rows = []
    for arm in ARMS:
        for method in METHODS:
            start_time = time.time()
            model = Model.load_dict(add_constraint(carried, arm, lstm_history),
                                    use_smoothed_look_back=False)
            mean, std, states = model.rsr_filter(
                online_data, start=base.WINDOW_LEN, window_len=base.WINDOW_LEN,
                strategy=method.removeprefix("rsr_"),
            )
            seconds = time.time() - start_time
            # Data units: the level carries the standardization mean
            frame = {"obs": obs, "mean": mean * y_std + y_mean, "std": std * y_std}
            for name in model.states_name:
                frame[f"{name}_mean"] = states.get_mean(name, "posterior") * y_std
                frame[f"{name}_std"] = states.get_std(name, "posterior") * y_std
            frame["level_mean"] += y_mean
            for name, (reading_mean, reading_std) in readings(states, model).items():
                frame[f"{name}_mean"] = reading_mean * y_std
                frame[f"{name}_std"] = reading_std * y_std
            if arm == "harmonic":
                for k in range(1, HARMONIC["num_harmonics"] + 1):
                    frame[f"amplitude{k}"] = np.hypot(frame[f"lstm harmonic cos{k}_mean"],
                                                      frame[f"lstm harmonic sin{k}_mean"])
            for name, values in truth.items():
                frame[f"true_{name}"] = values[test]
            frame = pd.DataFrame(frame, index=base.toy_series().index[test])
            frame.to_csv(out_dir / f"{arm}_{method}.csv")

            tail = slice(-LAST, None)
            row = {
                "seed": seed, "arm": arm, "method": method, "sigma_v": sigma_v,
                "optimal_epoch": epoch, "run_dir": str(out_dir),
                **base.score(obs, frame["mean"].to_numpy(), frame["std"].to_numpy()),
                "trend_mean": float(frame["trend_mean"].mean()),
                **base.trend_deviation(frame["trend_mean"].to_numpy()),
                "lstm_drift_slope": base.lstm_drift(frame["lstm_mean"].to_numpy()),
                "level_rmse_last": float(np.sqrt(np.mean(
                    (frame["level_mean"] - frame["true_level"])[tail] ** 2))),
                "seasonal_rmse_last": float(np.sqrt(np.mean(
                    (frame["lstm_mean"] - frame["true_seasonal"])[tail] ** 2))),
                "seconds": seconds,
            }
            if arm != "unconstrained":
                for name in ("reading_level", "reading_slope"):
                    values = frame[f"{name}_mean"].to_numpy()
                    row[f"{name}_mean_last"] = float(values[tail].mean())
                    row[f"{name}_ripple_last"] = ripple(values)
            rows.append(row)
            log.info("%s %s %s", arm, method, json.dumps(row))
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    log.info("COMPLETE %s", out_dir)
    log.removeHandler(handler)
    handler.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    tag = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y%m%d-%H%M%S") + "_toy"
    batch_dir = EXP_DIR / "results" / tag
    batch_dir.mkdir(parents=True, exist_ok=False)
    (EXP_DIR / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(EXP_DIR / "logs" / f"{tag}.log"),
                  logging.StreamHandler(sys.stdout)],
    )
    config = {
        "run_tag": tag, "seeds": args.seeds, "arms": list(ARMS), "methods": list(METHODS),
        "ewma_constraint": base.CONSTRAINT, "harmonic_constraint": HARMONIC,
        "memory_start": "warm start from frozen LSTM posterior means before the first window",
        "constraint_in_offline_training": False,
        "slope": base.SLOPE, "period": base.PERIOD, "train_len": base.TRAIN_LEN,
        "validation_len": base.VALIDATION_LEN, "toy_num_steps": base.TOY_NUM_STEPS,
        "toy_std_noise": base.TOY_STD_NOISE, "sigma_v_grid": base.SIGMA_V_GRID.tolist(),
        "max_epochs": base.MAX_EPOCHS, "patience": base.PATIENCE,
        "num_hidden_unit": base.NUM_HIDDEN_UNIT, "rsr_window_len": base.WINDOW_LEN,
        "canari": base.canari_fingerprint(),
    }
    (batch_dir / "config.json").write_text(json.dumps(config, indent=2))
    log.info("Run %s: %s", tag, json.dumps(config))
    jobs = [{**config, "seed": seed,
             "out_dir": str(EXP_DIR / "results" / f"{tag[:-4]}_toy_seed{seed}")}
            for seed in args.seeds]
    (batch_dir / "jobs.json").write_text(json.dumps(jobs, indent=2))

    rows = []
    with multiprocessing.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for seed_rows in pool.imap_unordered(run_seed, jobs):
            rows.extend(seed_rows)
            for row in seed_rows:
                log.info("seed %d %-13s %-20s RMSE=%.4f LL=%.3f trend=%.5f "
                         "drift=%.2e seasonal_rmse=%.3f (%.0f s)",
                         row["seed"], row["arm"], row["method"], row["rmse"],
                         row["ll_mean"], row["trend_mean"], row["lstm_drift_slope"],
                         row["seasonal_rmse_last"], row["seconds"])
    pd.DataFrame(rows).sort_values(["arm", "method", "seed"]).to_csv(
        batch_dir / "metrics.csv", index=False)
    log.info("Saved %s", batch_dir / "metrics.csv")


if __name__ == "__main__":
    main()
