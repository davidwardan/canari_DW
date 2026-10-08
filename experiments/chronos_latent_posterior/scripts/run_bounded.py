"""Exploratory bounded-mean Chronos control after the raw recursion failed."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_experiment import EXP, Transition, coherent, digest


class BoundedTransition(Transition):
    def __init__(self, pipeline, bound=3.):
        super().__init__(pipeline)
        self.bound = bound

    def __call__(self, histories, horizon=52):
        return self.bound*np.tanh(super().__call__(histories, horizon)/self.bound)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    if not (source / "COMPLETED").exists():
        raise ValueError("Original run must be complete")
    config = json.loads((source / "config.json").read_text())
    run = EXP / "results" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ_bounded")
    run.mkdir(parents=True, exist_ok=False)
    names = ("exact_control.csv", "example.npz", "hybrid_members.npz", "noise_fits.csv",
             "parameter_grid.csv", "parameter_scores.csv", "raw_baseline_quantiles.npz",
             "series.csv", "static_series.npz")
    for name in names:
        shutil.copy2(source / name, run / name)
    config.update(transition_kind="bounded_tanh", transition_bound=3., exploratory_followup=True,
                  original_unbounded_run=str(source),
                  reused_files_sha256={name: digest(source / name) for name in names},
                  bounded_script_sha256=digest(Path(__file__)),
                  bounded_protocol_sha256=digest(EXP / "planning/bounded_followup.md"))
    (run / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(config["model"], device_map="cpu", dtype=torch.float32,
                                               local_files_only=True)
    pipeline.model.eval()
    transition = BoundedTransition(pipeline)
    original = pd.read_csv(source / "scores.csv")
    scores = pd.concat([original[original.stage == "hybrid"],
                        pd.DataFrame(coherent(transition, run, config["smoke"]))], ignore_index=True)
    scores.to_csv(run / "scores.csv", index=False)
    primary = scores[(scores.stage == "coherent") & (scores.method == "particle64")]
    truth = np.load(run / "coherent_truth.npz")
    checks = {"transition_mean_bound": 3., "bound_verified": bool((primary.mu.abs() <= 3).all()),
              "finite_total_variances": bool(np.isfinite(scores.total).all()),
              "reconstruction_max": float(scores.reconstruction_error.max()),
              "maximum_abs_coherent_latent": float(max(np.abs(truth[key]).max() for key in truth.files if key.endswith("latent"))),
              "maximum_abs_coherent_prediction": float(primary.mu.abs().max()),
              "transition_calls": transition.calls, "transition_histories": transition.histories,
              "reused_hybrid_rows": int((scores.stage == "hybrid").sum()),
              "exploratory_model_change": True}
    if not checks["bound_verified"] or not checks["finite_total_variances"] or checks["reconstruction_max"] > 1e-10:
        raise ValueError(f"Bounded-run verification failed: {checks}")
    (run / "verification.json").write_text(json.dumps(checks, indent=2))
    (run / "COMPLETED").write_text("Exploratory bounded Chronos model completed; original failed run preserved.\n")
    print(f"Results: {run}", flush=True)


if __name__ == "__main__":
    main()
