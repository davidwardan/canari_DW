"""Run Chronos-2 small switching detection on real weekly data with an injection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
SWITCHING_DIR = REPO / "experiments/chronos_switching_posterior/scripts"
sys.path.insert(0, str(SWITCHING_DIR))
from switching import SwitchingParticleFilter, gaussian_crps, mixture_scores

RAW_VALUES = REPO / "data/hq_benchmark_data/weekly/weekly_values_raw.csv"
RAW_DATES = REPO / "data/hq_benchmark_data/weekly/weekly_datetimes_raw.csv"
REAL_SOURCE = REPO / "experiments/chronos2_gaussianity/results/run_20260922T144822_196315Z"
NOISE_SOURCE = REPO / "experiments/chronos_uncertainty_decomposition/results/run_20260924T123843_788223Z/real_noise_estimates.csv"
MODEL = Path("/Users/davidwardan/.cache/huggingface/hub/models--autogluon--chronos-2-small/") / \
    "snapshots/ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a"
SERIES = ("ts67", "ts66")
SERIES_SEEDS = {"ts67": 8127, "ts66": 8128}
CONTEXT = 32
TEST_START = pd.Timestamp("2022-01-01")
TEST_OFFSET = 24
PRE_STEPS = 6
ANOMALY_STEPS = 12
POST_STEPS = 6
SHIFT = 2.5
TRANSITION_MATRIX = np.array([[.995, .005], [.08, .92]])
REGIME_SHIFTS = np.array([0., SHIFT])
DATA_SEED = 8127
FILTER_SEED = 9137


def digest(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


class ChronosMedian:
    def __init__(self, pipeline):
        self.pipeline = pipeline
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
        return np.asarray([value[0].cpu().numpy()[0, self.median_index] for value in output], dtype=float)


def gaussian_nll(observation, mean, variance):
    return float(-norm.logpdf(observation, loc=mean, scale=np.sqrt(max(variance, 1e-15))))


def _auc(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return np.nan
    return float(np.mean((positives[:, None] > negatives[None, :]) +
                         .5 * (positives[:, None] == negatives[None, :])))


def _first_persistent(values, threshold, start=0, above=True, length=2):
    values = np.asarray(values)
    for i in range(start, len(values) - length + 1):
        segment = values[i:i + length]
        if np.all(segment >= threshold) if above else np.all(segment < threshold):
            return i
    return None


def load_series():
    source_config = json.loads((REAL_SOURCE / "config.json").read_text())
    noise = pd.read_csv(NOISE_SOURCE).set_index("series")
    values = pd.read_csv(RAW_VALUES)
    dates = pd.read_csv(RAW_DATES)
    datasets = {}
    for series in SERIES:
        raw = values[series].to_numpy(float)
        valid = np.flatnonzero(np.isfinite(raw))
        raw = raw[valid]
        time = pd.DatetimeIndex(pd.to_datetime(dates[series].iloc[valid]))
        scaler = source_config["scalers"][series]
        if not np.isclose(raw[:104].mean(), scaler["mean"]) or not np.isclose(raw[:104].std(), scaler["std"]):
            raise ValueError(f"{series}: raw standardization differs from pinned source")
        standardized = (raw - scaler["mean"]) / scaler["std"]
        test = np.flatnonzero(time >= TEST_START)
        anomaly_start = int(test[TEST_OFFSET])
        window_start = anomaly_start - PRE_STEPS
        window_end = anomaly_start + ANOMALY_STEPS + POST_STEPS
        if window_start < CONTEXT or window_end > len(standardized):
            raise ValueError(f"{series}: selected window lacks context")
        anomaly = np.zeros(window_end - window_start, dtype=int)
        anomaly[PRE_STEPS:PRE_STEPS + ANOMALY_STEPS] = 1
        clean = standardized[window_start:window_end].copy()
        observed = clean + SHIFT * anomaly
        datasets[series] = {
            "series": series, "dates": time[window_start:window_end].strftime("%Y-%m-%d").to_numpy(),
            "initial": standardized[window_start - CONTEXT:window_start].copy(),
            "clean": clean, "observed": observed, "regime": anomaly,
            "q": float(noise.loc[series, "q"]), "r": float(noise.loc[series, "r"]),
            "anomaly_start": PRE_STEPS, "anomaly_end": PRE_STEPS + ANOMALY_STEPS,
            "raw_source_start": str(time[window_start]), "raw_source_end": str(time[window_end - 1]),
        }
    return datasets, source_config


def run_filter(transition, episode, particles, seed, injected, switching=True):
    k = len(REGIME_SHIFTS) if switching else 1
    matrix = TRANSITION_MATRIX if switching else np.ones((1, 1))
    shifts = REGIME_SHIFTS if switching else np.zeros(1)
    q = np.full(k, episode["q"])
    r = np.full(k, episode["r"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, particles, int(injected), int(switching)]))
    histories = np.tile(episode["initial"], (particles, 1))
    particle_filter = SwitchingParticleFilter(
        transition, histories, np.zeros(particles, dtype=int), matrix, shifts, q, r, rng)
    observations = episode["observed"] if injected else episode["clean"]
    method = f"{'switching' if switching else 'fixed'}{particles}_{'injected' if injected else 'clean'}"
    rows, states = [], []
    for t, observation in enumerate(observations):
        prediction = particle_filter.predict()
        state_variance = prediction["state"] + prediction["regime"]
        clean_target = episode["clean"][t]
        observed_nll = gaussian_nll(observation, prediction["mean"], prediction["total"])
        clean_nll = gaussian_nll(clean_target, prediction["mean"], state_variance)
        posterior = particle_filter.update(observation, prediction)
        if switching:
            # Chronos models the clean baseline history.  Keep the additive
            # regime residual out of the next baseline-context query so a
            # detected level shift is not immediately explained away by
            # feeding the anomaly back into Chronos.
            particle_filter.histories[:, -1] -= shifts[particle_filter.regimes]
        p_prior = prediction["regime_probability"]
        p_post = posterior["posterior_regime_probability"]
        p_abnormal_prior = float(p_prior[1]) if switching else 0.
        p_abnormal_post = float(p_post[1]) if switching else 0.
        mix_nll, mix_crps = mixture_scores(
            observation, prediction["means"], prediction["variances"], prediction["weights"])
        true_regime = int(episode["regime"][t]) if injected else 0
        rows.append({
            "series": episode["series"], "method": method, "time": t,
            "date": episode["dates"][t], "injected": bool(injected),
            "true_regime": true_regime, "clean_target": clean_target,
            "observation": float(observation), "injected_delta": float(observation - clean_target),
            "mu": prediction["mean"], "aleatoric": prediction["aleatoric"],
            "epistemic_state": prediction["state"], "epistemic_regime": prediction["regime"],
            "epistemic": state_variance, "total": prediction["total"],
            "observation_nll": observed_nll, "clean_target_nll": clean_nll,
            "observation_crps": gaussian_crps(observation, prediction["mean"], prediction["total"]),
            "mixture_nll": mix_nll, "mixture_crps": mix_crps,
            "observation_coverage90": float(abs(observation - prediction["mean"]) <=
                                               1.645 * np.sqrt(prediction["total"])),
            "clean_target_coverage90": float(abs(clean_target - prediction["mean"]) <=
                                               1.645 * np.sqrt(max(state_variance, 1e-15))),
            "p_abnormal_prior": p_abnormal_prior, "p_abnormal_posterior": p_abnormal_post,
            "switch_probability": posterior["switch_probability"] if switching else 0.,
            "ess": posterior["ess"], "unique_ancestors": posterior["unique_ancestors"],
            "posterior_mean": posterior["posterior_mean"], "posterior_sd": posterior["posterior_sd"],
            "reconstruction_error": abs(prediction["total"] - prediction["aleatoric"] -
                                         prediction["state"] - prediction["regime"]),
        })
        states.append({
            "series": episode["series"], "method": method, "time": t, "date": episode["dates"][t],
            "injected": bool(injected), "true_regime": true_regime,
            "clean_target": clean_target, "observation": float(observation),
            "injected_delta": float(observation - clean_target), "forecast_mean": prediction["mean"],
            "forecast_sd": np.sqrt(prediction["total"]), "posterior_mean": posterior["posterior_mean"],
            "posterior_sd": posterior["posterior_sd"], "p_abnormal_posterior": p_abnormal_post,
            "aleatoric": prediction["aleatoric"], "epistemic_state": prediction["state"],
            "epistemic_regime": prediction["regime"], "total": prediction["total"],
        })
    return rows, states


def regime_summary(rows, episodes):
    result = []
    for (series, method), frame in rows.groupby(["series", "method"]):
        if not method.startswith("switching"):
            continue
        frame = frame.sort_values("time")
        labels = frame.true_regime.to_numpy()
        probabilities = frame.p_abnormal_posterior.to_numpy()
        episode = episodes[series]
        alarm = _first_persistent(probabilities, .5, start=episode["anomaly_start"], above=True)
        recovery = _first_persistent(probabilities, .5, start=episode["anomaly_end"], above=False)
        detected = alarm is not None and alarm < episode["anomaly_end"]
        result.append({
            "series": series, "method": method, "injected": bool(frame.injected.iloc[0]),
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "log_loss": float(-np.mean(labels * np.log(np.clip(probabilities, 1e-8, 1)) +
                                       (1 - labels) * np.log(np.clip(1 - probabilities, 1e-8, 1)))),
            "auroc": _auc(labels, probabilities), "detected": bool(detected),
            "detection_delay": float(alarm - episode["anomaly_start"]) if detected else np.nan,
            "recovery_delay": float(recovery - episode["anomaly_end"]) if recovery is not None else np.nan,
            "false_alarm_before_switch": bool(np.any(probabilities[:episode["anomaly_start"]] >= .5)),
            "max_clean_probability_before_switch": float(np.max(probabilities[:episode["anomaly_start"]])),
        })
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not RAW_VALUES.exists() or not RAW_DATES.exists() or not MODEL.exists():
        raise FileNotFoundError("Pinned real inputs or Chronos-2 small checkpoint missing")
    episodes, source_config = load_series()
    run = EXP / "results" / (datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ") +
                              ("_smoke" if args.smoke else ""))
    run.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), SWITCHING_DIR / "switching.py", EXP / "planning/README.md"]
    config = {
        "smoke": args.smoke, "model": str(MODEL), "revision": MODEL.name,
        "real_source_run": str(REAL_SOURCE), "noise_source": str(NOISE_SOURCE),
        "series": list(SERIES), "context": CONTEXT, "test_start": str(TEST_START.date()),
        "test_offset": TEST_OFFSET, "pre_steps": PRE_STEPS, "anomaly_steps": ANOMALY_STEPS,
        "post_steps": POST_STEPS, "injected_shift": SHIFT, "regime_names": ["normal", "abnormal"],
        "transition_matrix": TRANSITION_MATRIX.tolist(), "data_seed": DATA_SEED,
        "filter_seed": FILTER_SEED, "particles": 8 if args.smoke else 32,
        "particle_sensitivity": 8 if args.smoke else 64,
        "source_hashes": {str(path): digest(path) for path in sources},
        "input_hashes": {str(path): digest(path) for path in (RAW_VALUES, RAW_DATES, REAL_SOURCE / "config.json", NOISE_SOURCE)},
        "model_files_sha256": {name: digest(MODEL / name) for name in ("config.json", "model.safetensors")},
        "series_windows": {series: {key: value for key, value in episode.items()
                                     if key in ("q", "r", "raw_source_start", "raw_source_end", "anomaly_start", "anomaly_end")}
                           for series, episode in episodes.items()},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "pandas", "torch", "chronos-forecasting", "transformers", "matplotlib")},
    }
    if args.smoke:
        selected_series = ("ts67",)
        selected_steps = 12
        for episode in episodes.values():
            for key in ("clean", "observed", "regime", "dates"):
                episode[key] = episode[key][:selected_steps]
            episode["anomaly_end"] = min(episode["anomaly_end"], selected_steps)
    else:
        selected_series = SERIES
    config["selected_series"] = list(selected_series)
    config["steps"] = len(episodes[selected_series[0]]["clean"])
    (run / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(str(MODEL), device_map="cpu", dtype=torch.float32,
                                                 local_files_only=True)
    pipeline.model.eval()
    transition = ChronosMedian(pipeline)
    rows, states = [], []
    for series in selected_series:
        episode = episodes[series]
        seed = SERIES_SEEDS[series]
        for injected in (False, True):
            method_rows, method_states = run_filter(transition, episode, config["particles"],
                                                     seed, injected, switching=True)
            rows.extend(method_rows); states.extend(method_states)
            if injected:
                fixed_rows, fixed_states = run_filter(transition, episode, config["particles"],
                                                       seed, injected, switching=False)
                rows.extend(fixed_rows); states.extend(fixed_states)
        if series == selected_series[0] and not args.smoke:
            sensitivity_rows, sensitivity_states = run_filter(
                transition, episode, config["particle_sensitivity"], seed, True, switching=True)
            rows.extend(sensitivity_rows); states.extend(sensitivity_states)
        print(f"Completed {series}", flush=True)
    scores, state_frame = pd.DataFrame(rows), pd.DataFrame(states)
    scores.to_csv(run / "scores.csv", index=False)
    state_frame.to_csv(run / "state_plot_data.csv", index=False)
    pd.DataFrame(regime_summary(scores, episodes)).to_csv(run / "regime_scores.csv", index=False)
    summary = []
    for (series, method), frame in scores.groupby(["series", "method"]):
        summary.append({
            "method": method, "series": series, "n_rows": len(frame),
            "observation_rmse": float(np.sqrt(np.mean((frame.observation - frame.mu) ** 2))),
            "observation_crps": float(frame.observation_crps.mean()),
            "observation_nll": float(frame.observation_nll.mean()),
            "clean_target_nll": float(frame.clean_target_nll.mean()),
            "observation_coverage90": float(frame.observation_coverage90.mean()),
            "clean_target_coverage90": float(frame.clean_target_coverage90.mean()),
            "aleatoric": float(frame.aleatoric.mean()), "epistemic_state": float(frame.epistemic_state.mean()),
            "epistemic_regime": float(frame.epistemic_regime.mean()), "total": float(frame.total.mean()),
            "reconstruction_max": float(frame.reconstruction_error.max()),
        })
    pd.DataFrame(summary).to_csv(run / "summary.csv", index=False)
    checks = {
        "finite_scores": bool(np.isfinite(scores.select_dtypes(include=[np.number])).all().all()),
        "nonnegative_variances": bool((scores[["aleatoric", "epistemic_state", "epistemic_regime", "total"]].to_numpy() >= 0).all()),
        "reconstruction_max": float(scores.reconstruction_error.max()),
        "regime_probability_normalized": bool(np.allclose(scores.p_abnormal_posterior +
                                                           (1 - scores.p_abnormal_posterior), 1.)),
        "max_particle_ess_violation": int((scores.ess.to_numpy() >
                                            np.where(scores.method.str.startswith("switching"),
                                                     2 * scores.method.str.extract(r"(\d+)$")[0].astype(float),
                                                     scores.method.str.extract(r"(\d+)$")[0].astype(float)) + 1e-10).sum()),
        "chronos_calls": transition.calls, "chronos_histories": transition.histories,
    }
    if not checks["finite_scores"] or not checks["nonnegative_variances"] or checks["reconstruction_max"] > 1e-10:
        raise ValueError(f"verification failed: {checks}")
    (run / "verification.json").write_text(json.dumps(checks, indent=2))
    (run / "COMPLETED").write_text("Real weekly data with injected anomaly completed.\n")
    print(f"Results: {run}", flush=True)


if __name__ == "__main__":
    main()
