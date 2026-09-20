"""Sweep the re-forecast interval H over 10 weekly HQ-benchmark series.

For every (series, H) pair, filter the evaluation segment with an `Auxiliary`
component that queries Chronos-2 once every H steps, and record the one-step-ahead
predictive distribution at every step. Results go to a fresh timestamped directory
under `results/`; nothing is overwritten.

    python scripts/run_experiment.py
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


def climatology_scores(observations):
    """Scores of predicting the warmup mean, i.e. 0 in warmup-standardized units."""

    scores = common.score(
        np.zeros_like(observations), np.ones_like(observations), observations
    )
    return {f"{key}_clim": value for key, value in scores.items() if key != "num_scored"}


def write_config(run_dir, manifest, elapsed):
    import canari
    import chronos

    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "seed": common.SEED,
        "warmup": common.WARMUP,
        "horizons": common.HORIZONS,
        "sigma_v": common.SIGMA_V,
        "model_id": common.MODEL_ID,
        "quantile_levels": common.QUANTILE_LEVELS,
        "selection": {
            "min_span": common.MIN_SPAN,
            "max_gaps": common.MAX_GAPS,
            "max_abs_corr": common.MAX_ABS_CORR,
            "num_series": common.NUM_SERIES,
        },
        "series": manifest.to_dict(orient="records"),
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
    np.random.seed(common.SEED)
    torch.manual_seed(common.SEED)

    series_names, manifest = common.select_series()
    run_dir = common.EXP_DIR / "results" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    (run_dir / "raw").mkdir(parents=True)
    manifest.to_csv(common.EXP_DIR / "data" / "selected_series.csv", index=False)

    predict_fn = common.make_predict_fn(common.load_chronos())
    rows = []
    started = time.time()

    for series in series_names:
        frame = common.load_series(series)
        for horizon in common.HORIZONS:
            step_started = time.time()
            predictions = common.run_filter(frame, horizon, predict_fn)
            predictions.to_csv(run_dir / "raw" / f"{series}_H{horizon}.csv", index=False)

            observations = predictions["y"].to_numpy()
            scores = common.score(
                predictions["mu"].to_numpy(), predictions["std"].to_numpy(), observations
            )
            rows.append(
                {"series": series, "horizon": horizon, **scores}
                | climatology_scores(observations)
            )
            print(
                f"{series:>5s}  H={horizon:>3d}  "
                f"RMSE {scores['rmse']:.3f}  CRPS {scores['crps']:.3f}  "
                f"LL {scores['log_lik']:+.3f}  "
                f"({time.time() - step_started:.0f}s)",
                flush=True,
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    write_config(run_dir, manifest, time.time() - started)
    print(f"\nwrote {run_dir}  ({time.time() - started:.0f}s total)")


if __name__ == "__main__":
    main()
