"""Compare frozen Chronos models and refresh intervals with LocalAcceleration."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import try_to_load_from_cache
from scipy.stats import norm

from core import HORIZONS, WARMUP, advance, build_model, load_data

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EXP_DIR.parents[1]
sys.path.insert(0, str(EXP_DIR.parent / "llm_horizon_models/scripts"))
from models import MODELS, common, slug

CONTEXT_MAX = 512
QUANTILES = [.1587, .8413]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def predict_batch(pipeline, contexts, horizon):
    from chronos import Chronos2Pipeline

    inputs = [torch.tensor(np.asarray(context)[-CONTEXT_MAX:], dtype=torch.float32) for context in contexts]
    kwargs = {"limit_prediction_length": False}
    if isinstance(pipeline, Chronos2Pipeline):
        kwargs.update(cross_learning=False, batch_size=len(inputs), context_length=CONTEXT_MAX)
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="We recommend keeping prediction length")
        bounds, locations = pipeline.predict_quantiles(
            inputs=inputs, prediction_length=horizon, quantile_levels=QUANTILES, **kwargs,
        )
    forecasts = []
    for location, quantiles in zip(locations, bounds):
        if location.ndim == 2:
            location, quantiles = location[0], quantiles[0]
        mu, q = location.cpu().numpy(), quantiles.cpu().numpy()
        if mu.shape != (horizon,) or q.shape != (horizon, 2):
            raise ValueError("Unexpected forecast shape")
        if not np.isfinite(mu).all() or not np.isfinite(q).all() or np.any(q[:, 1] < q[:, 0]):
            raise ValueError("Nonfinite or crossed predictive quantiles")
        forecasts.append((mu, np.maximum(((q[:, 1] - q[:, 0]) / 2)**2, 1e-6), q))
    return forecasts


def check_batch(pipeline):
    """Check independent batched calls against singleton calls."""
    t = np.arange(80)
    contexts = [np.sin(2*np.pi*t/52), np.cos(2*np.pi*t/26) + .01*t]
    rows = []
    for horizon in (1, 208):
        batched = predict_batch(pipeline, contexts, horizon)
        maximum = 0.
        for i in range(len(contexts)):
            singleton = predict_batch(pipeline, [contexts[i]], horizon)[0]
            for actual, expected in zip(batched[i], singleton):
                np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
                maximum = max(maximum, float(np.max(np.abs(actual - expected))))
        rows.append({"horizon": horizon, "max_abs_difference": maximum, "passed": True})
    return rows


def run_group(frames, horizon, pipeline=None):
    """Advance independent Canari models, batching only foundation calls."""
    slots, calls, models = {}, {name: 0 for name in frames}, {}

    def callback(name):
        def predict(context, requested):
            if requested != horizon:
                raise AssertionError("Block length changed")
            mu, variance, _, saved_context = slots[name]
            np.testing.assert_array_equal(context, saved_context)
            calls[name] += 1
            return mu, variance
        return predict

    for name, frame in frames.items():
        models[name] = build_model(frame.y.to_numpy()[:WARMUP], callback(name), max(horizon, 1), pipeline is not None)
        if pipeline is not None:
            models[name].aux_component.max_context_len = CONTEXT_MAX
    records = {name: [] for name in frames}
    started, last_progress = time.perf_counter(), time.perf_counter()
    for index in range(WARMUP, max(map(len, frames.values()))):
        active = [name for name, frame in frames.items() if index < len(frame)]
        offset = (index - WARMUP) % max(horizon, 1)
        if pipeline is not None and offset == 0:
            contexts = [np.asarray(models[name].aux_component.context[-CONTEXT_MAX:]) for name in active]
            forecasts = predict_batch(pipeline, contexts, horizon)
            for name, context, (mu, variance, q) in zip(active, contexts, forecasts):
                slots[name] = (mu, variance, q, context)
        for name in active:
            observed = frames[name].iloc[index]
            prediction = advance(models[name], observed.y)
            prediction.update(observed.to_dict())
            prediction.update(index=index, lead=offset + 1 if pipeline is not None else 1,
                              block_origin=index - offset if pipeline is not None else index)
            if pipeline is not None:
                prediction["native_qlo"], prediction["native_qhi"] = slots[name][2][offset]
            records[name].append(prediction)
        if time.perf_counter() - last_progress > 20:
            print(f"  H={horizon}: week {index-WARMUP+1}, {len(active)} active series", flush=True)
            last_progress = time.perf_counter()
    elapsed = time.perf_counter() - started
    for name, frame in frames.items():
        expected = int(np.ceil((len(frame) - WARMUP) / horizon)) if pipeline is not None else 0
        if calls[name] != expected:
            raise AssertionError(f"Incorrect refresh count for {name}: {calls[name]} != {expected}")
    return {name: pd.DataFrame(rows) for name, rows in records.items()}, calls, elapsed


def scores(frame):
    y, mu, variance = (frame[key].to_numpy() for key in ("y", "mu", "var"))
    if not np.isfinite(mu).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Invalid predictive moments")
    result = common.score(mu, np.sqrt(variance), y)
    result["n_scored"] = result.pop("num_scored")
    valid = np.isfinite(y)
    radius = norm.ppf(.95) * np.sqrt(variance[valid])
    result.update(
        coverage90=float(np.mean(np.abs(y[valid] - mu[valid]) <= radius)),
        width90=float(np.mean(2 * radius)),
        slope_mae=float(np.mean(np.abs(frame.slope_prior[valid] - frame.synthetic_slope[valid]))),
        acceleration_mae=float(np.mean(np.abs(frame.acceleration_prior[valid] - frame.synthetic_acceleration[valid]))),
    )
    return result


def write_group(run, model_id, horizon, frames, pipeline, metric_rows):
    predictions, calls, elapsed = run_group(frames, horizon, pipeline)
    directory = run / "raw" / slug(model_id)
    directory.mkdir(parents=True, exist_ok=True)
    for name, prediction in predictions.items():
        prediction.to_csv(directory / f"{name}_H{horizon}.csv", index=False)
        metric_rows.append({"model": model_id, "series": name, "horizon": horizon,
                            **scores(prediction), "foundation_calls": calls[name],
                            "elapsed_seconds": elapsed})
    pd.DataFrame(metric_rows).to_csv(run / "metrics.csv", index=False)
    print(f"{model_id} H={horizon}: median RMSE {np.median([row['rmse'] for row in metric_rows[-len(frames):]]):.4f}; {elapsed:.1f}s", flush=True)


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
    sources = [*Path(__file__).parent.glob("*.py"), EXP_DIR / "planning/README.md", EXP_DIR.parent / "llm_horizon_degradation/scripts/common.py", REPO_DIR / "src/canari/model.py", REPO_DIR / "src/canari/common.py", REPO_DIR / "src/canari/component/auxiliary_component.py", REPO_DIR / "src/canari/component/baseline_component.py"]
    config = {"smoke": args.smoke, "created_utc": datetime.now(timezone.utc).isoformat(), "seed": 0, "warmup": WARMUP, "horizons": horizons, "context_max": CONTEXT_MAX, "acceleration_per_year2": .05, "acceleration_per_week2": .05 / 52**2, "sigma_v": .1, "acceleration_process_std": 0., "models": [m for m, *_ in MODELS], "series": list(frames), "manifest": manifest, "quantile_levels": QUANTILES, "cross_learning": False, "torch_threads": 1, "batching": "independent series at matching relative forecast origin", "elapsed_seconds_definition": "shared wall time of a whole model/H group, repeated per series; do not sum across series", "checkpoints": checkpoints, "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "torch", "chronos-forecasting", "transformers", "matplotlib")}, "input_sha256": {str(p): digest(p) for p in inputs}, "source_sha256": {str(p.relative_to(REPO_DIR)): digest(p) for p in sources}}
    (run / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Run: {run}", flush=True)
    metric_rows, checks = [], []
    write_group(run, "acceleration_only", 0, frames, None, metric_rows)
    from chronos import BaseChronosPipeline
    for model_id, *_ in MODELS:
        pipeline = BaseChronosPipeline.from_pretrained(checkpoints[model_id]["path"], device_map="cpu", dtype=torch.float32, local_files_only=True)
        pipeline.model.eval()
        for row in check_batch(pipeline):
            checks.append({"model": model_id, **row})
        pd.DataFrame(checks).to_csv(run / "batch_checks.csv", index=False)
        for horizon in horizons:
            write_group(run, model_id, horizon, frames, pipeline, metric_rows)
        del pipeline
        gc.collect()
    (run / "COMPLETED").write_text(f"All {len(metric_rows)} model/series/H rows completed.\n")
    print(f"Completed: {run}", flush=True)


if __name__ == "__main__":
    main()
