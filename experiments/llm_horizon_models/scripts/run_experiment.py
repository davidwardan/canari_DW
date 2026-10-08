"""Sweep H x series x Chronos model, with the pipeline of `llm_horizon_degradation`.

Results go to a fresh timestamped directory under `results/`; nothing is
overwritten. `metrics.csv` is rewritten after every model, so a partial run is
still usable.

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
from models import EXP_DIR, MODELS, REFERENCE_RUN, common, slug

KEYS = ["rmse", "mae", "log_lik", "crps"]


def climatology_scores(observations):
    """Scores of predicting the warmup mean, i.e. 0 in warmup-standardized units."""

    scores = common.score(
        np.zeros_like(observations), np.ones_like(observations), observations
    )
    return {f"{key}_clim": value for key, value in scores.items() if key != "num_scored"}


def write_config(run_dir, manifest, elapsed):
    import chronos

    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "seed": common.SEED,
        "warmup": common.WARMUP,
        "horizons": common.HORIZONS,
        "sigma_v": common.SIGMA_V,
        "models": [model_id for model_id, *_ in MODELS],
        "quantile_levels": common.QUANTILE_LEVELS,
        "series": manifest.to_dict(orient="records"),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "chronos": chronos.__version__,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))


def check_reference(metrics):
    """The Chronos-2 rows must reproduce the original degradation run."""

    reference = pd.read_csv(REFERENCE_RUN / "metrics.csv").set_index(["series", "horizon"])
    ours = metrics[metrics["model"] == "amazon/chronos-2"].set_index(["series", "horizon"])
    diff = (ours[KEYS] - reference.loc[ours.index, KEYS]).abs().to_numpy().max()
    print(f"Chronos-2 vs {REFERENCE_RUN.name}: max abs difference {diff:.2e}")
    return float(diff)


def main():
    np.random.seed(common.SEED)
    torch.manual_seed(common.SEED)

    series_names, manifest = common.select_series()
    run_dir = EXP_DIR / "results" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    manifest.to_csv(EXP_DIR / "data" / "selected_series.csv", index=False)

    frames = {series: common.load_series(series) for series in series_names}
    rows = []
    started = time.time()

    for model_id, label, _, _ in MODELS:
        raw_dir = run_dir / "raw" / slug(model_id)
        raw_dir.mkdir(parents=True)
        predict_fn = common.make_predict_fn(common.load_chronos(model_id))
        model_started = time.time()

        for series in series_names:
            for horizon in common.HORIZONS:
                predictions = common.run_filter(frames[series], horizon, predict_fn)
                predictions.to_csv(raw_dir / f"{series}_H{horizon}.csv", index=False)

                observations = predictions["y"].to_numpy()
                scores = common.score(
                    predictions["mu"].to_numpy(), predictions["std"].to_numpy(), observations
                )
                rows.append(
                    {"model": model_id, "series": series, "horizon": horizon, **scores}
                    | climatology_scores(observations)
                )
                print(
                    f"{label:>15s}  {series:>5s}  H={horizon:>3d}  "
                    f"RMSE {scores['rmse']:.3f}  LL {scores['log_lik']:+.3f}",
                    flush=True,
                )

        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
        print(f"{label} done in {time.time() - model_started:.0f}s", flush=True)

    metrics = pd.DataFrame(rows)
    write_config(run_dir, manifest, time.time() - started)
    check_reference(metrics)
    print(f"\nwrote {run_dir}  ({time.time() - started:.0f}s total)")


if __name__ == "__main__":
    main()
