"""Sweep the re-forecast interval H for anomaly detection on 10 weekly series.

The SKF configuration and the anomaly magnitude come from `tune.py`, which chose
them at H=1 on held-out series; nothing here is refitted. For every (series, H)
the clean series is filtered once to measure false alarms, then each anomalous
realization is filtered in turn. Realizations are identical across horizons, so
each H is scored on exactly the same anomalies.

    python scripts/tune.py && python scripts/run_experiment.py
"""

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def load_tuned():
    if not common.TUNED_PARAMS_FILE.exists():
        raise SystemExit(
            f"{common.TUNED_PARAMS_FILE} not found; run scripts/tune.py first."
        )
    return json.loads(common.TUNED_PARAMS_FILE.read_text())


def write_config(run_dir, tuned, series_names, elapsed):
    import canari
    import chronos

    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "seed": common.SEED,
        "warmup": common.WARMUP,
        "horizons": common.HORIZONS,
        "sigma_v": common.SIGMA_V,
        "model_id": common.base.MODEL_ID,
        "series": series_names,
        "num_realizations": common.NUM_REALIZATIONS,
        "max_weeks_to_detect": common.MAX_WEEKS_TO_DETECT,
        "alarm_threshold": common.ALARM_THRESHOLD,
        "anomaly_window": common.ANOMALY_WINDOW,
        "tuned": tuned,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "chronos": chronos.__version__,
            "canari": getattr(canari, "__version__", "unknown"),
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))


def main():
    tuned = load_tuned()
    series_names, _ = common.base.select_series()
    run_dir = common.EXP_DIR / "results" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)

    tasks = [
        common.make_task(
            series,
            horizon,
            tuned["params"],
            tuned["slope_per_week"],
            common.NUM_REALIZATIONS,
        )
        for series in series_names
        for horizon in common.HORIZONS
    ]
    print(
        f"{len(tasks)} cases "
        f"({len(series_names)} series x {len(common.HORIZONS)} horizons), "
        f"{common.NUM_REALIZATIONS + 1} filters each, "
        f"anomaly {tuned['slope_per_year']:.2f} sigma/year"
    )

    started = time.time()
    detections, false_alarms = common.run_all(tasks, checkpoint_dir=run_dir)
    detections.to_csv(run_dir / "detections.csv", index=False)
    false_alarms.to_csv(run_dir / "false_alarms.csv", index=False)

    summary = common.summarize(detections, false_alarms, by="horizon")
    summary.to_csv(run_dir / "summary.csv")
    write_config(run_dir, tuned, series_names, time.time() - started)

    print("\n" + summary.round(3).to_string())
    print(f"\nwrote {run_dir}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
