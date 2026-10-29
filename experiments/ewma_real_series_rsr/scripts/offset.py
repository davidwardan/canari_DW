"""Level-offset test: is a wrong starting level corrected under each setup?

Re-runs the test arms of a finished run from their snapshots over the whole series, with
the week-0 level shifted by +0.5 (data units) and a prior std of 0.5, so the true level is
one std away and can be reached. Each run is paired with the same arm, snapshot and seed
without offset. Because level + LSTM + noise = y exactly, whatever offset the level keeps
is carried, with opposite sign, by the LSTM or the noise:

- `level_kept_year1`, `level_kept_test`: mean level difference (offset minus no-offset run)
  over weeks 52-103 and over the test split, as a fraction of the offset (0: corrected);
- `lstm_took_test`: minus the mean LSTM difference over the test split, as a fraction of
  the offset (0: the LSTM did not take any of it);
- `weeks_to_correct`: first week from which the level difference stays below 10% of the
  offset (the series length if never), and `corrected` (1 if it does);
- level and trend RMSE and predictive scores on the test split, as in the main run.

Runs, summary and figures go to `results/level_offset/` and `figures/level_offset/`.

    PYTHONPATH=/Users/davidwardan/canari_DW/src python \
        experiments/ewma_real_series_rsr/scripts/offset.py --tag RUN_TAG
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

import run_experiment as rx
import summarize as sm  # sets the shared plotting defaults before pyplot loads

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OFFSET, OFFSET_STD = 0.5, 0.5  # data units
SUBDIR = "level_offset"
METRICS = ["level_kept_year1", "level_kept_test", "lstm_took_test", "weeks_to_correct",
           "corrected", "level_rmse", "trend_rmse", "rmse", "ll", "crps", "coverage_95"]

log = logging.getLogger("experiment")


def correction_time(level_kept) -> dict:
    """First week from which |level kept| stays below 10%; the series length if never."""

    uncorrected = np.flatnonzero(np.abs(level_kept) >= 0.1)
    corrected = not uncorrected.size or uncorrected[-1] < len(level_kept) - 1
    weeks = int(uncorrected[-1] + 1) if uncorrected.size else 0
    return {"weeks_to_correct": weeks if corrected else len(level_kept),
            "corrected": float(corrected)}


def run_offset(job: dict) -> dict:
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)
    handler = rx.start_log(f"{SUBDIR}/{out_dir.name}")
    (out_dir / "config.json").write_text(json.dumps(job, indent=2))
    log.info("START %s", json.dumps(job))
    series, arm, seed = job["series"], job["arm"], job["seed"]
    detrended = rx.detrended_series(series)
    data = rx.with_trend(detrended)
    data_processor = rx.make_processor(data)
    _, _, _, all_data = data_processor.get_splits()
    y_mean, y_std = rx.scale(data_processor)
    _, baseline = rx.true_baseline(detrended)
    with Path(job["snapshot_path"]).open("rb") as stream:
        snapshot = pickle.load(stream)

    row = {"series": series, "arm": arm, "seed": seed, "run": out_dir.name,
           "status": "complete"}
    try:
        mean, std, hidden = rx.run_full(snapshot, rx.ARMS[arm]["online"], all_data,
                                        level_shift=(OFFSET / y_std, (OFFSET_STD / y_std) ** 2))
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
        log.warning("FAILED %s", error)
        row.update(status="numerical_failure", error=str(error))
        (out_dir / "metrics.json").write_text(json.dumps(row, indent=2))
        rx.stop_log(handler)
        return row
    states = rx.states_frame(data, baseline, mean, std, hidden, y_mean, y_std)
    states.to_csv(out_dir / "full_states.csv")

    reference = pd.read_csv(Path(job["reference_run"]) / "full_states.csv")
    level = (states["level_mean"].to_numpy() - reference["level_mean"].to_numpy()) / OFFSET
    lstm = (states["lstm_mean"].to_numpy() - reference["lstm_mean"].to_numpy()) / OFFSET
    row.update({
        "level_kept_year1": float(level[rx.PERIOD:2 * rx.PERIOD].mean()),
        "level_kept_test": float(level[rx.TEST_START:].mean()),
        "lstm_took_test": float(-lstm[rx.TEST_START:].mean()),
        **correction_time(level),
        **rx.score_test_split(states),
    })
    (out_dir / "metrics.json").write_text(json.dumps(row, indent=2))
    log.info("COMPLETE %s", json.dumps(row))
    rx.stop_log(handler)
    return row


def summarize(runs: pd.DataFrame, batch_dir: Path, figures_dir: Path, tag: str) -> None:
    runs = runs[runs["status"] == "complete"]
    seed_means = runs.groupby(["series", "arm"], as_index=False)[METRICS].mean()
    seed_means.to_csv(batch_dir / "seed_means.csv", index=False)
    lines = [f"# Level-offset test, run `{tag}`", "",
             f"Week-0 level shifted by +{OFFSET} with prior std {OFFSET_STD} (data units); "
             "fractions of the offset, paired with the run without offset. "
             "Definitions: `scripts/offset.py`.", "",
             sm.table(seed_means, METRICS[:5], "series (each series averaged over seeds)"), "",
             sm.table(seed_means, METRICS[5:], "series (each series averaged over seeds)"), ""]
    (batch_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))

    rows = (("level_kept_year1", "level kept, year 1"), ("level_kept_test", "level kept, test"),
            ("lstm_took_test", "LSTM took, test"), ("level_rmse", "level RMSE"))
    fig, axes = plt.subplots(len(rows), 1, sharex=True, figsize=(3.5, 4.6))
    grouped = seed_means.groupby("arm")
    for ax, (metric, label) in zip(axes, rows):
        sm.draw(ax, grouped[metric].mean(), grouped[metric].std())
        ax.set_ylabel(label)
    axes[-1].set_xlabel(r"tolerance ratio $\kappa = \tau/\lambda$")
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.4)
    sm.legend(fig, ncol=2)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sm.save(fig, figures_dir, f"{tag}_offset_vs_kappa")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="finished run whose snapshots are reused")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    tag = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y%m%d-%H%M%S") + "_offset"
    results_dir = rx.EXP_DIR / "results" / SUBDIR
    batch_dir = results_dir / tag
    batch_dir.mkdir(parents=True, exist_ok=False)
    (rx.EXP_DIR / "logs" / SUBDIR).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(rx.EXP_DIR / "logs" / SUBDIR / f"{tag}.log"),
                                  logging.StreamHandler(sys.stdout)])
    source = json.loads((rx.EXP_DIR / "results" / args.tag / "jobs.json").read_text())
    jobs = [{**job, "source_tag": args.tag, "level_offset": OFFSET,
             "level_offset_std": OFFSET_STD, "reference_run": job["out_dir"],
             "out_dir": str(results_dir / f"{tag}_{job['series']}_{job['arm']}_seed{job['seed']}")}
            for job in source["evaluation"]]
    (batch_dir / "config.json").write_text(json.dumps(
        {"source_tag": args.tag, "level_offset": OFFSET, "level_offset_std": OFFSET_STD,
         "source_config": json.loads((rx.EXP_DIR / "results" / args.tag / "config.json").read_text())},
        indent=2))
    (batch_dir / "jobs.json").write_text(json.dumps({"evaluation": jobs}, indent=2))
    with multiprocessing.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        rows = list(pool.imap_unordered(run_offset, jobs))
    runs = pd.DataFrame(rows).sort_values(["series", "arm", "seed"])
    runs.to_csv(batch_dir / "metrics.csv", index=False)
    summarize(runs, batch_dir, rx.EXP_DIR / "figures" / SUBDIR, tag)


if __name__ == "__main__":
    main()
