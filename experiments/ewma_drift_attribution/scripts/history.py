"""Full-series states of selected runs, for the hidden-state figures.

The evaluation runs only saved the test split. This re-runs (scenario, arm, seed) from
the snapshots of a finished run and saves predictions and filtered states over the
whole series in new run folders. The test part must reproduce the original run's
`states.csv`; the largest absolute difference is recorded in `check.json`.

    PYTHONPATH=/Users/davidwardan/canari_DW/src python \
        experiments/ewma_drift_attribution/scripts/history.py --tag RUN_TAG
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import logging
import multiprocessing
import pickle
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import run_experiment as rx

ARMS = ("none", "frozen", *(f"rsr_k{rx.kappa_label(kappa)}" for kappa in rx.KAPPAS))
SEEDS = (1, 5)

log = logging.getLogger("experiment")


def run_history(job: dict) -> dict:
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)
    handler = rx.start_log(out_dir.name)
    (out_dir / "config.json").write_text(json.dumps(job, indent=2))
    log.info("START %s", json.dumps(job))
    scenario, arm, seed = job["scenario"], job["arm"], job["seed"]
    data = rx.toy_series(scenario, seed)
    data_processor = rx.make_processor(data)
    _, _, _, all_data = data_processor.get_splits()
    y_mean, y_std = data_processor.scale_const_mean[0], data_processor.scale_const_std[0]
    with Path(job["snapshot_path"]).open("rb") as stream:
        snapshot = pickle.load(stream)
    mean, std, hidden = rx.run_full_history(snapshot, arm, rx.leak_of(data_processor), all_data)
    states = rx.states_frame(data, scenario, 0, mean, std, hidden, y_mean, y_std)
    states.insert(0, "split", np.where(np.arange(len(states)) < rx.TRAIN_LEN, "train",
                                       np.where(np.arange(len(states)) < rx.TEST_START,
                                                "validation", "test")))
    states.to_csv(out_dir / "full_states.csv")

    original = pd.read_csv(Path(job["original_run"]) / "states.csv", index_col=0)
    test = states.iloc[rx.TEST_START:][original.columns].to_numpy()
    check = {"max_abs_diff_test": float(np.max(np.abs(test - original.to_numpy())))}
    (out_dir / "check.json").write_text(json.dumps(check, indent=2))
    log.info("COMPLETE %s", json.dumps(check))
    rx.stop_log(handler)
    return {"run": out_dir.name, "scenario": scenario, "arm": arm, "seed": seed, **check}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="finished run to reproduce")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scenarios", nargs="+", choices=rx.SCENARIOS, default=list(rx.SCENARIOS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", choices=rx.ARMS, default=list(ARMS))
    args = parser.parse_args()

    history_tag = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y%m%d-%H%M%S") + "_history"
    batch_dir = rx.EXP_DIR / "results" / history_tag
    batch_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(rx.EXP_DIR / "logs" / f"{history_tag}.log"),
                                  logging.StreamHandler(sys.stdout)])
    jobs = []
    for scenario in args.scenarios:
        for arm in args.arms:
            for seed in args.seeds:
                original = rx.EXP_DIR / "results" / f"{args.tag}_{scenario}_{arm}_seed{seed}"
                jobs.append({
                    "source_tag": args.tag, "scenario": scenario, "arm": arm, "seed": seed,
                    "snapshot_path": json.loads((original / "config.json").read_text())["snapshot_path"],
                    "original_run": str(original),
                    "out_dir": str(rx.EXP_DIR / "results" / f"{history_tag}_{scenario}_{arm}_seed{seed}"),
                })
    (batch_dir / "config.json").write_text(json.dumps({"source_tag": args.tag, "jobs": jobs}, indent=2))
    with multiprocessing.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        rows = list(pool.imap_unordered(run_history, jobs))
    checks = pd.DataFrame(rows).sort_values(["scenario", "arm", "seed"])
    checks.to_csv(batch_dir / "checks.csv", index=False)
    log.info("Saved %s; largest test difference %.2e", batch_dir,
             checks["max_abs_diff_test"].max())


if __name__ == "__main__":
    main()
