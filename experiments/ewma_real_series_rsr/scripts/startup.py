"""Start-up diagnostics: how well the LSTM and the baseline begin at week 0.

For each frozen run (`frozen`, `frozen_ewma_k1`) of the given run tags, from its snapshot
and saved full-series states:

- `look_back_rmse` (toy only): RMS error of the inferred look-back, the 52 LSTM outputs
  the snapshot starts from, against the true pattern before week 0 (normalized units);
- `lstm_rmse_year0` (toy only): RMS error of the LSTM output against the true pattern
  over the first 52 weeks (data units);
- `forecast_ratio_year0`: one-step forecast RMSE over the first 52 weeks divided by that
  over weeks 52-207;
- `trend_peak_dev`: largest |trend - 0.005| over the training split (data units/week);
- `level_err_test_start`: level minus the true baseline at week 312 (data units).

    PYTHONPATH=/Users/davidwardan/canari_DW/src python \
        experiments/ewma_real_series_rsr/scripts/startup.py --tags TAG [TAG ...]
"""

import argparse
import json
import pickle

import numpy as np
import pandas as pd

import run_experiment as rx
from canari import Model
from metrics import PERIOD, rms

ARMS = ("frozen", "frozen_ewma_k1")


def diagnose(job: dict) -> dict:
    states = pd.read_csv(f"{job['out_dir']}/full_states.csv")
    with open(job["snapshot_path"], "rb") as stream:
        model = Model.load_dict(pickle.load(stream))
    error = (states["obs"] - states["mean"]).to_numpy()
    row = {
        "forecast_ratio_year0": rms(error[:PERIOD][np.isfinite(error[:PERIOD])])
        / rms(error[PERIOD:rx.TRAIN_LEN][np.isfinite(error[PERIOD:rx.TRAIN_LEN])]),
        "trend_peak_dev": float(np.max(np.abs(states["trend_mean"][:rx.TRAIN_LEN] - rx.SLOPE))),
        "level_err_test_start": float(states["level_mean"][rx.TEST_START]
                                      - states["baseline"][rx.TEST_START]),
    }
    if job["series"] == "toy":
        y_std = rx.scale(rx.make_processor(rx.with_trend(rx.detrended_series("toy"))))[1]
        before = rx.toy_pattern(np.arange(-PERIOD, 0)) / y_std
        row["look_back_rmse"] = rms(np.asarray(model.lstm_output_history.mu) - before)
        row["lstm_rmse_year0"] = rms(states["lstm_mean"][:PERIOD]
                                     - rx.toy_pattern(np.arange(PERIOD)))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tags", nargs="+", required=True)
    args = parser.parse_args()
    rows = []
    for tag in args.tags:
        batch_dir = rx.EXP_DIR / "results" / tag
        infer_len = json.loads((batch_dir / "config.json").read_text()).get("infer_len", PERIOD)
        jobs = json.loads((batch_dir / "jobs.json").read_text())["evaluation"]
        tag_rows = [{"tag": tag, "infer_len": infer_len, "series": job["series"],
                     "arm": job["arm"], "seed": job["seed"], **diagnose(job)}
                    for job in jobs if job["arm"] in ARMS]
        pd.DataFrame(tag_rows).to_csv(batch_dir / "startup.csv", index=False)
        rows += tag_rows
    frame = pd.DataFrame(rows)
    metrics = [c for c in ("look_back_rmse", "lstm_rmse_year0", "forecast_ratio_year0",
                           "trend_peak_dev", "level_err_test_start") if c in frame]
    grouped = frame.groupby(["series", "arm", "infer_len"])[metrics]
    table = grouped.mean().round(4).astype(str) + " ± " + grouped.std().round(4).astype(str)
    print(table.to_string())


if __name__ == "__main__":
    main()
