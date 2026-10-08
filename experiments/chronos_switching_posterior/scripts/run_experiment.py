"""Synthetic switching-regime experiment around bounded frozen Chronos."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from switching import SwitchingParticleFilter, gaussian_crps, mixture_scores

EXP = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = EXP.parent / "chronos_latent_posterior/results/run_20261007T181022_448332Z_bounded"
DEFAULT_SMALL_MODEL = Path(
    "/Users/davidwardan/.cache/huggingface/hub/models--autogluon--chronos-2-small/"
    "snapshots/ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a")
REGIME_NAMES = ("normal", "abnormal")
STEPS = 96
CONTEXT = 32
SWITCH_TIMES = (24, 56)
Q = np.array([0.01, 0.01])
R = np.array([0.04, 0.04])
SHIFTS = np.array([0.0, 0.75])
TRANSITION_MATRIX = np.array([[0.995, 0.005], [0.08, 0.92]])
DATA_SEEDS = tuple(range(5))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class BoundedChronos:
    """Frozen Chronos median with the declared exploratory amplitude bound."""

    def __init__(self, pipeline, bound=3.0):
        self.pipeline = pipeline
        self.bound = float(bound)
        self.levels = np.asarray(pipeline.quantiles)
        self.median_index = int(np.flatnonzero(np.isclose(self.levels, .5))[0])
        self.calls = 0
        self.histories = 0

    def __call__(self, histories, horizon=1):
        histories = np.atleast_2d(np.asarray(histories, dtype=float))
        with torch.inference_mode():
            output, _ = self.pipeline.predict_quantiles(
                inputs=[torch.tensor(history, dtype=torch.float32) for history in histories],
                prediction_length=horizon, quantile_levels=self.levels.tolist(),
                batch_size=64, cross_learning=False, context_length=512)
        self.calls += 1
        self.histories += len(histories)
        raw = np.stack([value[0].cpu().numpy()[0, self.median_index] for value in output])
        return self.bound * np.tanh(raw / self.bound)


def gaussian_nll(observation, mean, variance):
    sigma = np.sqrt(max(variance, 1e-15))
    return float(-norm.logpdf(observation, loc=mean, scale=sigma))


def _auc(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return np.nan
    return float(np.mean((positives[:, None] > negatives[None, :]) +
                         .5 * (positives[:, None] == negatives[None, :])))


def _first_persistent(values, predicate, start=0, length=2):
    values = np.asarray(values)
    for index in range(start, len(values) - length + 1):
        if np.all(predicate(values[index:index + length])):
            return index
    return None


def simulate_episode(transition, seed, steps=STEPS, switch_times=SWITCH_TIMES):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4101]))
    initial = np.sin(2 * np.pi * np.arange(CONTEXT) / 16)
    regimes = np.zeros(steps, dtype=int)
    regimes[switch_times[0]:switch_times[1]] = 1
    context = initial.copy()
    latent, observed, means = [], [], []
    for regime in regimes:
        base_mean = float(transition(context)[0])
        mean = base_mean + SHIFTS[regime]
        x = mean + rng.normal(0., np.sqrt(Q[regime]))
        y = x + rng.normal(0., np.sqrt(R[regime]))
        means.append(mean)
        latent.append(x)
        observed.append(y)
        context = np.r_[context[1:], x]
    return {
        "initial": initial, "regime": regimes,
        "latent": np.asarray(latent), "observed": np.asarray(observed),
        "transition_mean": np.asarray(means),
    }


def run_filter(transition, episode, particles, seed, switching=True):
    if switching:
        matrix, shifts, q, r = TRANSITION_MATRIX, SHIFTS, Q, R
        initial_regimes = np.zeros(particles, dtype=int)
        method = f"switching{particles}"
    else:
        matrix, shifts, q, r = np.ones((1, 1)), np.zeros(1), Q[:1], R[:1]
        initial_regimes = np.zeros(particles, dtype=int)
        method = f"fixed{particles}"
    rng = np.random.default_rng(np.random.SeedSequence([seed, particles, 4102, int(switching)]))
    histories = np.tile(episode["initial"], (particles, 1))
    particle_filter = SwitchingParticleFilter(
        transition, histories, initial_regimes, matrix, shifts, q, r, rng)
    score_rows, state_rows = [], []
    for time, observation in enumerate(episode["observed"]):
        prediction = particle_filter.predict()
        state_variance = prediction["state"] + prediction["regime"]
        observed_nll = gaussian_nll(observation, prediction["mean"], prediction["total"])
        observed_crps = gaussian_crps(observation, prediction["mean"], prediction["total"])
        predictable = float(episode["transition_mean"][time])
        predictable_nll = gaussian_nll(predictable, prediction["mean"], state_variance)
        predictable_crps = gaussian_crps(predictable, prediction["mean"], state_variance)
        mix_nll, mix_crps = mixture_scores(
            observation, prediction["means"], prediction["variances"], prediction["weights"])
        posterior = particle_filter.update(observation, prediction)
        p_prior = prediction["regime_probability"]
        p_post = posterior["posterior_regime_probability"]
        p_abnormal_prior = float(p_prior[1]) if switching else 0.0
        p_abnormal_post = float(p_post[1]) if switching else 0.0
        score_rows.append({
            "method": method, "seed": seed, "time": time,
            "true_regime": int(episode["regime"][time]),
            "observation": float(observation), "latent": float(episode["latent"][time]),
            "predictable": predictable, "mu": prediction["mean"],
            "aleatoric": prediction["aleatoric"],
            "epistemic_state": prediction["state"],
            "epistemic_regime": prediction["regime"],
            "epistemic": state_variance, "total": prediction["total"],
            "observation_nll": observed_nll, "observation_crps": observed_crps,
            "predictable_nll": predictable_nll, "predictable_crps": predictable_crps,
            "mixture_nll": mix_nll, "mixture_crps": mix_crps,
            "observation_coverage90": float(abs(observation - prediction["mean"]) <=
                                               1.645 * np.sqrt(prediction["total"])),
            "predictable_coverage90": float(abs(predictable - prediction["mean"]) <=
                                               1.645 * np.sqrt(max(state_variance, 1e-15))),
            "p_abnormal_prior": p_abnormal_prior,
            "p_abnormal_posterior": p_abnormal_post,
            "switch_probability": posterior["switch_probability"] if switching else 0.0,
            "ess": posterior["ess"], "unique_ancestors": posterior["unique_ancestors"],
            "posterior_mean": posterior["posterior_mean"],
            "posterior_sd": posterior["posterior_sd"],
            "reconstruction_error": abs(prediction["total"] - prediction["aleatoric"] -
                                         prediction["state"] - prediction["regime"]),
        })
        state_rows.append({
            "method": method, "seed": seed, "time": time,
            "true_regime": int(episode["regime"][time]),
            "observation": float(observation), "latent": float(episode["latent"][time]),
            "transition_mean": predictable, "process_shock": float(episode["latent"][time] - predictable),
            "measurement_noise": float(observation - episode["latent"][time]),
            "forecast_mean": prediction["mean"], "forecast_sd": np.sqrt(prediction["total"]),
            "posterior_mean": posterior["posterior_mean"], "posterior_sd": posterior["posterior_sd"],
            "p_abnormal_prior": p_abnormal_prior, "p_abnormal_posterior": p_abnormal_post,
            "aleatoric": prediction["aleatoric"], "epistemic_state": prediction["state"],
            "epistemic_regime": prediction["regime"], "total": prediction["total"],
        })
    return score_rows, state_rows


def regime_summary(rows, switch_times=SWITCH_TIMES):
    result = []
    for (method, seed), frame in rows.groupby(["method", "seed"]):
        if method.startswith("fixed"):
            continue
        labels = frame.true_regime.to_numpy()
        probabilities = frame.p_abnormal_posterior.to_numpy()
        clipped = np.clip(probabilities, 1e-8, 1 - 1e-8)
        first_alarm = _first_persistent(probabilities, lambda x: x >= .5, length=2)
        after_recovery = _first_persistent(probabilities, lambda x: x < .5,
                                           start=switch_times[1], length=2)
        abnormal_start = switch_times[0]
        detection = first_alarm is not None and abnormal_start <= first_alarm < switch_times[1]
        result.append({
            "method": method, "seed": seed,
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
            "auroc": _auc(labels, probabilities),
            "detected": bool(detection),
            "detection_delay": float(first_alarm - abnormal_start) if detection else np.nan,
            "recovery_delay": float(after_recovery - switch_times[1]) if after_recovery is not None else np.nan,
            "false_alarm_before_switch": bool(np.any(probabilities[:abnormal_start] >= .5)),
            "mean_switch_probability": float(frame.switch_probability.mean()),
        })
    return pd.DataFrame(result)


def load_transition(source, model_override=None):
    config = json.loads((source / "config.json").read_text())
    model = Path(model_override) if model_override is not None else Path(config["model"])
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(
        str(model), device_map="cpu", dtype=torch.float32, local_files_only=True)
    pipeline.model.eval()
    return BoundedChronos(pipeline), config, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_SMALL_MODEL)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--micro", action="store_true", help="six-step actual-model mechanism check")
    parser.add_argument("--quick", action="store_true", help="one 16-step seed with reduced particle counts")
    args = parser.parse_args()
    source = args.source.resolve()
    if not (source / "COMPLETED").exists():
        raise ValueError(f"Completed bounded Chronos run required: {source}")
    transition, source_config, model = load_transition(source, args.model)
    if args.micro:
        seeds, steps, particles, sensitivity, switch_times = (0,), 6, 2, 2, (2, 4)
    elif args.smoke:
        seeds, steps, particles, sensitivity, switch_times = (0,), 8, 4, 8, (2, 6)
    elif args.quick:
        seeds, steps, particles, sensitivity, switch_times = (0,), 16, 4, 8, (4, 12)
    else:
        seeds, steps, particles, sensitivity, switch_times = DATA_SEEDS, STEPS, 64, 128, SWITCH_TIMES
    suffix = "_micro" if args.micro else "_smoke" if args.smoke else "_quick" if args.quick else ""
    run = EXP / "results" / (datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ") + suffix)
    run.mkdir(parents=True, exist_ok=False)
    script_paths = list(Path(__file__).parent.glob("*.py")) + [EXP / "planning/README.md"]
    config = {
        "smoke": args.smoke, "micro": args.micro, "quick": args.quick,
        "source_bounded_run": str(source), "model": str(model),
        "revision": model.name, "source_model_revision": source_config["revision"],
        "transition": "3*tanh(Chronos median/3)",
        "context": CONTEXT, "steps": steps, "switch_times": list(switch_times),
        "regime_names": list(REGIME_NAMES), "regime_shifts": SHIFTS.tolist(),
        "q": Q.tolist(), "r": R.tolist(), "transition_matrix": TRANSITION_MATRIX.tolist(),
        "data_seeds": list(seeds), "particles": particles,
        "particle_sensitivity": sensitivity,
        "source_hashes": {str(path): digest(path) for path in script_paths},
        "model_files_sha256": {name: digest(model / name) for name in ("config.json", "model.safetensors")},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "pandas", "torch", "chronos-forecasting", "transformers", "matplotlib")},
        "seeds": {"data": 4101, "filter": 4102},
    }
    (run / "config.json").write_text(json.dumps(config, indent=2))
    transition_fn = transition
    score_rows, state_rows, truth = [], [], {}
    for seed in seeds:
        episode = simulate_episode(transition_fn, seed, steps=steps, switch_times=switch_times)
        truth[f"seed{seed}_regime"] = episode["regime"]
        truth[f"seed{seed}_latent"] = episode["latent"]
        truth[f"seed{seed}_observed"] = episode["observed"]
        truth[f"seed{seed}_transition_mean"] = episode["transition_mean"]
        rows, states = run_filter(transition_fn, episode, config["particles"], seed, switching=True)
        score_rows.extend(rows); state_rows.extend(states)
        if not args.quick and not args.micro:
            baseline_rows, baseline_states = run_filter(
                transition_fn, episode, config["particles"], seed, switching=False)
            score_rows.extend(baseline_rows); state_rows.extend(baseline_states)
        if seed == 0 and not args.quick and not args.micro:
            sensitivity_rows, sensitivity_states = run_filter(
                transition_fn, episode, config["particle_sensitivity"], seed, switching=True)
            score_rows.extend(sensitivity_rows); state_rows.extend(sensitivity_states)
        print(f"Switching filter: seed{seed}, {config['particles']} particles", flush=True)
    np.savez_compressed(run / "truth.npz", initial=np.sin(2 * np.pi * np.arange(CONTEXT) / 16), **truth)
    scores = pd.DataFrame(score_rows)
    states = pd.DataFrame(state_rows)
    scores.to_csv(run / "scores.csv", index=False)
    states.to_csv(run / "state_plot_data.csv", index=False)
    regime = regime_summary(scores, switch_times=switch_times)
    regime.to_csv(run / "regime_scores.csv", index=False)
    primary = scores[scores.method == f"switching{config['particles']}"]
    particle_limits = scores.method.str.extract(r"(\d+)$")[0].astype(float).to_numpy()
    particle_limits *= np.where(scores.method.str.startswith("switching"), 2., 1.)
    summary_rows = []
    for method, frame in scores.groupby("method"):
        summary_rows.append({
            "method": method, "n_rows": len(frame),
            "observation_rmse": float(np.sqrt(np.mean((frame.observation-frame.mu)**2))),
            "observation_crps": float(frame.observation_crps.mean()),
            "observation_nll": float(frame.observation_nll.mean()),
            "observation_coverage90": float(frame.observation_coverage90.mean()),
            "predictable_coverage90": float(frame.predictable_coverage90.mean()),
            "aleatoric": float(frame.aleatoric.mean()),
            "epistemic_state": float(frame.epistemic_state.mean()),
            "epistemic_regime": float(frame.epistemic_regime.mean()),
            "total": float(frame.total.mean()),
            "reconstruction_max": float(frame.reconstruction_error.max()),
        })
    pd.DataFrame(summary_rows).to_csv(run / "summary.csv", index=False)
    checks = {
        "finite_scores": bool(np.isfinite(scores.select_dtypes(include=[np.number])).all().all()),
        "nonnegative_variances": bool((scores[["aleatoric", "epistemic_state", "epistemic_regime", "total"]].to_numpy() >= 0).all()),
        "reconstruction_max": float(scores.reconstruction_error.max()),
        "regime_probability_normalized": bool(np.allclose(
            primary.p_abnormal_posterior + (1 - primary.p_abnormal_posterior), 1.)),
        "transition_mean_abs_max": float(np.abs(primary.mu).max()),
        "transition_mean_bound_verified": bool(np.abs(primary.mu).max() <= 3. + np.abs(SHIFTS).max() + 1e-10),
        "max_particle_ess_violation": int((scores.ess.to_numpy() > particle_limits + 1e-10).sum()),
        "chronos_calls": transition.calls, "chronos_histories": transition.histories,
    }
    if (not checks["finite_scores"] or not checks["nonnegative_variances"] or
            checks["reconstruction_max"] > 1e-10 or not checks["transition_mean_bound_verified"]):
        raise ValueError(f"Switching verification failed: {checks}")
    (run / "verification.json").write_text(json.dumps(checks, indent=2))
    (run / "COMPLETED").write_text("Switching Chronos particle-filter experiment completed.\n")
    print(f"Results: {run}", flush=True)


if __name__ == "__main__":
    main()
