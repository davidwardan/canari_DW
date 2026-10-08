"""Repeat the LocalAcceleration H sweep with the Chronos variance forced to zero."""

import argparse
import gc
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import try_to_load_from_cache

EXP_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = EXP_DIR.parent / "llm_horizon_linear_acceleration"
REPO_DIR = EXP_DIR.parents[1]
sys.path.insert(0, str(REFERENCE_DIR / "scripts"))
import run_experiment as reference  # noqa: E402
from core import HORIZONS, WARMUP, load_data  # noqa: E402
from run_experiment import CONTEXT_MAX, MODELS, QUANTILES, check_batch, common, digest, write_group  # noqa: E402

chronos_predict_batch = reference.predict_batch


def predict_batch_zero_variance(pipeline, contexts, horizon):
    """Chronos means and native quantiles, with the variance set to exactly zero."""
    return [(mu, np.zeros_like(variance), q) for mu, variance, q in chronos_predict_batch(pipeline, contexts, horizon)]


# run_group and check_batch look predict_batch up in the reference module.
reference.predict_batch = predict_batch_zero_variance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    frames, manifest = load_data()
    if args.smoke:
        frames = {name: frame.iloc[:WARMUP + 24].copy() for name, frame in list(frames.items())[:2]}
    horizons = [1, 13, 208] if args.smoke else HORIZONS
    run = EXP_DIR / "results" / (datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ") + ("_smoke" if args.smoke else ""))
    (run / "data").mkdir(parents=True, exist_ok=False)
    for name, frame in frames.items():
        frame.to_csv(run / "data" / f"{name}.csv", index=False)
    checkpoints = {}
    for model_id, *_ in MODELS:
        cached = try_to_load_from_cache(model_id, "config.json")
        if not isinstance(cached, str):
            raise FileNotFoundError(f"Checkpoint not cached: {model_id}")
        folder = Path(cached).parent
        architecture = json.loads((folder / "config.json").read_text())["chronos_config"]
        native_length = architecture.get("prediction_length")
        if native_length is None:
            native_length = architecture["output_patch_size"] * architecture["max_output_patches"]
        checkpoints[model_id] = {"path": str(folder), "revision": folder.name,
                                "native_prediction_length": native_length,
                                "sha256": {p.name: digest(p) for p in folder.iterdir() if p.suffix in (".json", ".safetensors", ".bin")}}
    inputs = [common.WEEKLY_DIR / "weekly_values.csv", common.WEEKLY_DIR / "weekly_datetimes.csv", EXP_DIR.parent / "llm_horizon_models/results/run_20260924_142849/config.json"]
    sources = [*Path(__file__).parent.glob("*.py"), *(REFERENCE_DIR / "scripts").glob("*.py"), EXP_DIR / "planning/README.md", EXP_DIR.parent / "llm_horizon_degradation/scripts/common.py", REPO_DIR / "src/canari/model.py", REPO_DIR / "src/canari/common.py", REPO_DIR / "src/canari/component/auxiliary_component.py", REPO_DIR / "src/canari/component/baseline_component.py"]
    config = {"smoke": args.smoke, "created_utc": datetime.now(timezone.utc).isoformat(), "seed": 0, "warmup": WARMUP, "horizons": horizons, "context_max": CONTEXT_MAX, "acceleration_per_year2": .05, "acceleration_per_week2": .05 / 52**2, "sigma_v": .1, "acceleration_process_std": 0., "chronos_variance": "forced to exactly 0; means and native quantiles unchanged", "reference_experiment": str(REFERENCE_DIR.relative_to(REPO_DIR)), "models": [m for m, *_ in MODELS], "series": list(frames), "manifest": manifest, "quantile_levels": QUANTILES, "cross_learning": False, "torch_threads": 1, "batching": "independent series at matching relative forecast origin", "elapsed_seconds_definition": "shared wall time of a whole model/H group, repeated per series; do not sum across series", "checkpoints": checkpoints, "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "torch", "chronos-forecasting", "transformers", "matplotlib")}, "input_sha256": {str(p): digest(p) for p in inputs}, "source_sha256": {str(p.relative_to(REPO_DIR)): digest(p) for p in sources}}
    (run / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Run: {run}", flush=True)
    metric_rows, checks, failures = [], [], []
    write_group(run, "acceleration_only", 0, frames, None, metric_rows)
    from chronos import BaseChronosPipeline
    for model_id, *_ in MODELS:
        pipeline = BaseChronosPipeline.from_pretrained(checkpoints[model_id]["path"], device_map="cpu", dtype=torch.float32, local_files_only=True)
        pipeline.model.eval()
        for row in check_batch(pipeline):
            checks.append({"model": model_id, **row})
        pd.DataFrame(checks).to_csv(run / "batch_checks.csv", index=False)
        for horizon in horizons:
            completed = len(metric_rows)
            try:
                write_group(run, model_id, horizon, frames, pipeline, metric_rows)
            except ValueError as error:
                # A self-fed rollout can overflow float32; record the group as diverged and keep going.
                del metric_rows[completed:]
                pd.DataFrame(metric_rows).to_csv(run / "metrics.csv", index=False)
                failures.append({"model": model_id, "horizon": horizon, "error": str(error)})
                pd.DataFrame(failures).to_csv(run / "failures.csv", index=False)
                print(f"{model_id} H={horizon}: FAILED ({error})", flush=True)
        del pipeline
        gc.collect()
    (run / "COMPLETED").write_text(f"All {len(metric_rows)} model/series/H rows completed; {len(failures)} model/H groups diverged (failures.csv).\n")
    print(f"Completed: {run}", flush=True)


if __name__ == "__main__":
    main()
