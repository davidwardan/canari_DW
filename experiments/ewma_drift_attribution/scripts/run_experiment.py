"""EWMA drift attribution on a toy series: who carries a whole-record slope under RSR?

See `../planning/README.md` for the protocol and `../planning/metrics.md` for the
metrics. Stage 1 trains one offline snapshot per scenario family, snapshot type and
seed. Stage 2 runs every (scenario, arm, seed) from those snapshots, either frozen or
with RSR `replay_smoothed`, and saves the test-split states and metrics.

Run from the repository root, with the source tree that has `EwmaConstraint`:

    PYTHONPATH=/Users/davidwardan/canari_DW/src python \
        experiments/ewma_drift_attribution/scripts/run_experiment.py --workers 6
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
from metrics import PERIOD, block_slope, ewma_readings, leakage, rms, score, slope_reading

EXP_DIR = Path(__file__).resolve().parents[1]

NUM_STEPS = 900
TRAIN_LEN = 4 * PERIOD
VALIDATION_LEN = 2 * PERIOD
TEST_START = TRAIN_LEN + VALIDATION_LEN
CHANGE_START = TEST_START + 2 * PERIOD  # scenario D: on a block boundary
PHASE_SHIFT = PERIOD // 4
SLOPE = 0.005  # data units per week
STD_NOISE = 0.1
UNKNOWN_TREND_STD = 0.01  # scenario C trend prior std, normalized units
SEEDS = (1, 2, 3, 4, 5)
SIGMA_V_GRID = np.linspace(0.01, 0.3, 5)
MAX_EPOCHS = 50
PATIENCE = 10
NUM_HIDDEN_UNIT = 50
WINDOW_LEN = PERIOD
STRATEGY = "replay_smoothed"
FAST_SPAN, SLOW_SPAN = 2 * PERIOD, 4 * PERIOD
TRAIN_KAPPA = 1.0
KAPPAS = (None, 8, 4, 2, 1, 0.5, 0.25)  # None: tolerance disabled (kappa = infinity)

SCENARIOS = {
    "A": {"slope": 0.0, "phase_shift": False, "unknown_trend": False, "family": "A"},
    "B": {"slope": SLOPE, "phase_shift": False, "unknown_trend": False, "family": "B"},
    "C": {"slope": SLOPE, "phase_shift": False, "unknown_trend": True, "family": "C"},
    "D": {"slope": SLOPE, "phase_shift": True, "unknown_trend": False, "family": "B"},
}
FAMILIES = ("A", "B", "C")
SNAPSHOTS = ("none", "ewma")


def kappa_label(kappa) -> str:
    return "inf" if kappa is None else f"{kappa:g}"


ARMS = {
    "none": {"snapshot": "none", "online": "rsr", "kappa_mean": None, "kappa_slope": None},
    "frozen": {"snapshot": "ewma", "online": "frozen",
               "kappa_mean": TRAIN_KAPPA, "kappa_slope": TRAIN_KAPPA},
    **{f"rsr_k{kappa_label(kappa)}": {"snapshot": "ewma", "online": "rsr",
                                      "kappa_mean": kappa, "kappa_slope": kappa}
       for kappa in KAPPAS},
    "rsr_mean_k1": {"snapshot": "ewma", "online": "rsr",
                    "kappa_mean": TRAIN_KAPPA, "kappa_slope": None},
    "rsr_slope_k1": {"snapshot": "ewma", "online": "rsr",
                     "kappa_mean": None, "kappa_slope": TRAIN_KAPPA},
}
SOURCE_FILES = ("model.py", "rsr.py", "common.py", "component/ewma_component.py",
                "component/lstm_component.py")

log = logging.getLogger("experiment")


def pattern(time) -> np.ndarray:
    return np.sin(2 * np.pi * time / PERIOD) + 0.5 * np.sin(
        4 * np.pi * time / PERIOD + np.pi / 4
    )


def true_pattern(scenario: str) -> np.ndarray:
    time = np.arange(NUM_STEPS)
    if SCENARIOS[scenario]["phase_shift"]:
        time = np.where(time >= CHANGE_START, time + PHASE_SHIFT, time)
    return pattern(time)


def toy_series(scenario: str, seed: int) -> pd.DataFrame:
    """Slope + seasonal pattern + noise; the noise depends on the seed only."""

    noise = np.random.default_rng(seed).normal(0, STD_NOISE, NUM_STEPS)
    values = SCENARIOS[scenario]["slope"] * np.arange(NUM_STEPS) + true_pattern(scenario)
    index = pd.date_range("2000-01-02", periods=NUM_STEPS, freq="7D")
    return pd.DataFrame({"values": values + noise}, index=pd.Index(index, name="date_time"))


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


def leak_of(data_processor: DataProcess) -> tuple:
    """Seasonal leakage (mean, slope) of the true pattern, in normalized units."""

    cycle = pattern(np.arange(PERIOD)) / data_processor.scale_const_std[0]
    return leakage(cycle, FAST_SPAN, SLOW_SPAN)


def tolerances(kappa_mean, kappa_slope, leak) -> tuple:
    return (None if kappa_mean is None else kappa_mean * leak[0],
            None if kappa_slope is None else kappa_slope * leak[1])


def make_constraint(taus: tuple) -> EwmaConstraint:
    return EwmaConstraint(source="lstm", fast_span=FAST_SPAN, slow_span=SLOW_SPAN,
                          tau_mean=taus[0], tau_slope=taus[1])


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
        components.append(make_constraint(taus))
    return Model(*components)


def arm_model(snapshot: dict, arm: str, leak: tuple) -> Model:
    """Load a snapshot and give it the arm's online tolerances."""

    snapshot = copy.deepcopy(snapshot)
    spec = ARMS[arm]
    if spec["snapshot"] == "ewma":
        constraint = make_constraint(tolerances(spec["kappa_mean"], spec["kappa_slope"], leak))
        snapshot["components"][constraint.name] = constraint
    return Model.load_dict(snapshot)


def train_offline(data_processor, sigma_v: float, seed: int, taus, unknown_trend: bool,
                  max_epochs: int):
    """Train with validation early stopping; return the best epoch's snapshot."""

    train_data, validation_data, _, _ = data_processor.get_splits()
    validation_obs = validation_data["y"].flatten()
    model = build_model(sigma_v, seed, taus)
    model.auto_initialize_baseline_states(train_data["y"])
    if unknown_trend:
        index = model.get_states_index("trend")
        model.mu_states[index] = 0.0
        model.var_states[index, index] = UNKNOWN_TREND_STD**2

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
    """Stage 1: one snapshot for a (family, snapshot type, seed)."""

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)
    handler = start_log(out_dir.name)
    (out_dir / "config.json").write_text(json.dumps(job, indent=2))
    log.info("START %s", json.dumps(job))
    family, seed = job["family"], job["seed"]
    data_processor = make_processor(toy_series(family, seed))
    leak = leak_of(data_processor)
    taus = None if job["snapshot"] == "none" else tolerances(TRAIN_KAPPA, TRAIN_KAPPA, leak)

    start_time = time.time()
    candidates, candidate_status = [], []
    for sigma_v in SIGMA_V_GRID:
        best_log_lik, optimal_epoch, snapshot, history, status = train_offline(
            data_processor, float(sigma_v), seed, taus,
            SCENARIOS[family]["unknown_trend"], job["max_epochs"],
        )
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
        "validation_ll": best_log_lik, "leak": leak, "train_taus": taus,
        "trend_init_normalized": float(
            snapshot["mu_states"][snapshot["states_name"].index("trend"), 0]),
        "training_seconds": time.time() - start_time,
        "candidates": [{"sigma_v": c[1], "validation_ll": c[0] if np.isfinite(c[0]) else None,
                        "optimal_epoch": c[2]} for c in candidates],
        "candidate_status": candidate_status,
    }
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    with (out_dir / "snapshot.pkl").open("wb") as stream:
        pickle.dump(snapshot, stream)
    log.info("COMPLETE %s", json.dumps(selection))
    stop_log(handler)
    return {"name": out_dir.name, "family": family, "snapshot": job["snapshot"],
            "seed": seed, **selection}


def run_online(model: Model, all_data: dict, online: str):
    """Test-split predictions and filtered states."""

    if online == "frozen":
        model.lstm_net.num_samples = len(all_data["y"])
        model.lstm_net.eval()
        mean, std, states = model.filter(all_data, train_lstm=False)
        test = slice(TEST_START, None)
    else:
        mean, std, states = model.rsr_filter(
            data=all_data, start=TEST_START, window_len=WINDOW_LEN, strategy=STRATEGY)
        test = slice(None)
    hidden = {name: (states.get_mean(name, "posterior")[test],
                     states.get_std(name, "posterior")[test]) for name in model.states_name}
    return mean[test], std[test], hidden


def run_full_history(snapshot: dict, arm: str, leak: tuple, all_data: dict):
    """Predictions and filtered states over the whole series, for the state figures.

    For RSR arms the frozen warm-up before the first replay window is reconstructed with
    the same arm model, as `rsr_filter` runs it internally; RSR then returns that window
    and every test step. Before the test split these are the selected snapshot's
    filtered states, not the states of the training epochs.
    """

    model = arm_model(snapshot, arm, leak)
    if ARMS[arm]["online"] == "frozen":
        model.lstm_net.num_samples = len(all_data["y"])
        model.lstm_net.eval()
        parts = [model.filter(all_data, train_lstm=False)]
    else:
        prefix_end = TEST_START - WINDOW_LEN
        prefix_model = arm_model(snapshot, arm, leak)
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


def states_frame(data: pd.DataFrame, scenario: str, start: int, mean, std, hidden: dict,
                 y_mean: float, y_std: float) -> pd.DataFrame:
    """Predictions and hidden states from `start` on, back in data units.

    The level carries the standardization mean; the other states are additive.
    """

    states = pd.DataFrame(
        {"obs": data["values"].to_numpy()[start:], "mean": mean * y_std + y_mean,
         "std": std * y_std, "pattern": true_pattern(scenario)[start:]},
        index=data.index[start:],
    )
    for name, (mu, sd) in hidden.items():
        column = name.replace(" ", "_")
        states[f"{column}_mean"] = mu * y_std + (y_mean if name == "level" else 0.0)
        states[f"{column}_std"] = sd * y_std
    return states


def run_evaluation(job: dict) -> dict:
    """Stage 2: one (scenario, arm, seed) from its stage-1 snapshot."""

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)
    handler = start_log(out_dir.name)
    (out_dir / "config.json").write_text(json.dumps(job, indent=2))
    log.info("START %s", json.dumps(job))
    scenario, arm, seed = job["scenario"], job["arm"], job["seed"]
    data = toy_series(scenario, seed)
    data_processor = make_processor(data)
    _, _, _, all_data = data_processor.get_splits()
    y_mean, y_std = data_processor.scale_const_mean[0], data_processor.scale_const_std[0]
    leak = leak_of(data_processor)
    with Path(job["snapshot_path"]).open("rb") as stream:
        snapshot = pickle.load(stream)
    model = arm_model(snapshot, arm, leak)
    taus = tolerances(ARMS[arm]["kappa_mean"], ARMS[arm]["kappa_slope"], leak)

    row = {"scenario": scenario, "arm": arm, "seed": seed, "run": out_dir.name,
           "tau_mean": taus[0], "tau_slope": taus[1], "status": "complete"}
    start_time = time.time()
    try:
        mean, std, hidden = run_online(model, all_data, ARMS[arm]["online"])
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
        log.warning("FAILED %s", error)
        row.update(status="numerical_failure", error=str(error))
        (out_dir / "metrics.json").write_text(json.dumps(row, indent=2))
        stop_log(handler)
        return row
    row["seconds"] = time.time() - start_time

    states = states_frame(data, scenario, TEST_START, mean, std, hidden, y_mean, y_std)
    states.to_csv(out_dir / "states.csv")
    obs, true = states["obs"].to_numpy(), states["pattern"].to_numpy()

    row.update(score(obs, states["mean"].to_numpy(), states["std"].to_numpy()))
    after = slice(CHANGE_START - TEST_START, None)
    row["crps_after_change"] = score(
        obs[after], states["mean"].to_numpy()[after], states["std"].to_numpy()[after])["crps"]
    components = {"level": states["level_mean"], "lstm": states["lstm_mean"],
                  "noise": states["white_noise_mean"]}
    row["sum_identity_error"] = float(np.max(np.abs(sum(components.values()) - obs)))
    row["lstm_drift"] = block_slope(states["lstm_mean"], TEST_START)
    row["pattern_rmse"] = rms(states["lstm_mean"] - true)
    slope = SCENARIOS[scenario]["slope"]
    if slope:
        for name, values in components.items():
            row[f"share_{name}"] = block_slope(values, TEST_START) / slope
        row["share_sum"] = block_slope(obs, TEST_START) / slope
        row["share_trend"] = float(states["trend_mean"].mean()) / slope
    if "lstm ewma slow" in hidden:
        # Readings of the posterior memories against those of the true pattern alone
        fast, slow = (hidden[f"lstm ewma {key}"][0] for key in ("fast", "slow"))
        true_mean, true_slope = ewma_readings(
            true_pattern(scenario) / y_std, FAST_SPAN, SLOW_SPAN)
        row["reading_ratio_mean"] = rms(slow) / rms(true_mean[TEST_START:])
        row["reading_ratio_slope"] = (
            rms(slope_reading(fast, slow, FAST_SPAN, SLOW_SPAN))
            / rms(true_slope[TEST_START:]))
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
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--tag-suffix", default="", help="e.g. _smoke")
    args = parser.parse_args()

    run_tag = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y%m%d-%H%M%S")
    run_tag += args.tag_suffix
    batch_dir = EXP_DIR / "results" / run_tag
    batch_dir.mkdir(parents=True, exist_ok=False)
    (EXP_DIR / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(EXP_DIR / "logs" / f"{run_tag}.log"),
                                  logging.StreamHandler(sys.stdout)])

    config = {
        "run_tag": run_tag, "seeds": args.seeds, "scenarios": args.scenarios,
        "arms": {arm: ARMS[arm] for arm in args.arms}, "scenario_specs": SCENARIOS,
        "num_steps": NUM_STEPS, "train_len": TRAIN_LEN, "validation_len": VALIDATION_LEN,
        "test_start": TEST_START, "change_start": CHANGE_START, "phase_shift": PHASE_SHIFT,
        "std_noise": STD_NOISE, "unknown_trend_std": UNKNOWN_TREND_STD,
        "sigma_v_grid": SIGMA_V_GRID.tolist(), "max_epochs": args.max_epochs,
        "patience": PATIENCE, "num_hidden_unit": NUM_HIDDEN_UNIT, "local_trend_std": 0.0,
        "rsr_window_len": WINDOW_LEN, "rsr_strategy": STRATEGY,
        "fast_span": FAST_SPAN, "slow_span": SLOW_SPAN, "train_kappa": TRAIN_KAPPA,
        "canari_path": str(Path(canari.__file__).parent), "source_sha256": source_hashes(),
    }
    (batch_dir / "config.json").write_text(json.dumps(config, indent=2))
    log.info("Run %s: %s", run_tag, json.dumps(config))

    families = sorted({SCENARIOS[scenario]["family"] for scenario in args.scenarios})
    snapshot_types = sorted({ARMS[arm]["snapshot"] for arm in args.arms})
    training_jobs = [
        {"run_tag": run_tag, "family": family, "snapshot": kind, "seed": seed,
         "max_epochs": args.max_epochs,
         "out_dir": str(EXP_DIR / "results" / f"{run_tag}_train_{family}_{kind}_seed{seed}")}
        for family in families for kind in snapshot_types for seed in args.seeds
    ]
    context = multiprocessing.get_context("spawn")
    with context.Pool(min(args.workers, len(training_jobs))) as pool:
        selections = list(pool.imap_unordered(run_training, training_jobs))
    pd.DataFrame(selections).drop(columns=["candidates", "candidate_status"]).to_csv(
        batch_dir / "training.csv", index=False)

    evaluation_jobs = []
    for scenario in args.scenarios:
        family = SCENARIOS[scenario]["family"]
        for arm in args.arms:
            kind = ARMS[arm]["snapshot"]
            for seed in args.seeds:
                train_dir = EXP_DIR / "results" / f"{run_tag}_train_{family}_{kind}_seed{seed}"
                evaluation_jobs.append({
                    "run_tag": run_tag, "scenario": scenario, "arm": arm, "seed": seed,
                    "snapshot_path": str(train_dir / "snapshot.pkl"),
                    "out_dir": str(EXP_DIR / "results" / f"{run_tag}_{scenario}_{arm}_seed{seed}"),
                })
    (batch_dir / "jobs.json").write_text(json.dumps(
        {"training": training_jobs, "evaluation": evaluation_jobs}, indent=2))
    rows = []
    with context.Pool(min(args.workers, len(evaluation_jobs))) as pool:
        for row in pool.imap_unordered(run_evaluation, evaluation_jobs):
            rows.append(row)
            log.info("%s %-12s seed %d  %s  CRPS=%.4f  lstm_drift=%.2e  share_lstm=%s",
                     row["scenario"], row["arm"], row["seed"], row["status"],
                     row.get("crps", np.nan), row.get("lstm_drift", np.nan),
                     row.get("share_lstm", "-"))
    pd.DataFrame(rows).sort_values(["scenario", "arm", "seed"]).to_csv(
        batch_dir / "metrics.csv", index=False)
    log.info("Saved %s", batch_dir / "metrics.csv")


if __name__ == "__main__":
    main()
