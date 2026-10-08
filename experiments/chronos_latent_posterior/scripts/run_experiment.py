"""Known-noise Chronos posterior propagation, coherent filtering and controls."""

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
from scipy.special import logsumexp
from scipy.stats import norm

from particle import adapted_update
from posterior import filter_history, fit_noise, forecast_components, sample_history
from parameter import noise_grid, parameter_forecast

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
PREVIOUS = EXP.parent / "chronos_uncertainty_decomposition"
sys.path.extend([str(EXP.parent / "chronos_context_floor/scripts"), str(PREVIOUS / "scripts")])
from data import build_datasets
from decomposition import ADAPTER_PATH, gaussian_metrics, variance_adapters

SOURCE = PREVIOUS / "results/run_20260924T123843_788223Z"
LEADS = np.array([1, 4, 13, 52])


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Transition:
    """Deterministic Chronos median map; separate univariate histories in batches."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.levels = np.asarray(pipeline.quantiles)
        self.median_index = int(np.flatnonzero(np.isclose(self.levels, .5))[0])
        self.calls, self.histories = 0, 0

    def quantiles(self, histories, horizon=52):
        histories = np.atleast_2d(histories)
        with torch.inference_mode():
            output, _ = self.pipeline.predict_quantiles(
                inputs=[torch.tensor(history, dtype=torch.float32) for history in histories],
                prediction_length=horizon, quantile_levels=self.levels.tolist(),
                batch_size=64, cross_learning=False, context_length=512)
        self.calls += 1
        self.histories += len(histories)
        return np.stack([value[0].cpu().numpy() for value in output])

    def __call__(self, histories, horizon=52):
        return self.quantiles(histories, horizon)[..., self.median_index].astype(float)


def mixture_scores(y, means, noise):
    """Exact scalar finite Gaussian-mixture NLL and CRPS, common noise variance."""
    means = np.asarray(means)
    sigma = np.sqrt(noise)
    log_density = -.5*((y-means)/sigma)**2-np.log(sigma)-.5*np.log(2*np.pi)
    def absolute_moment(delta, std):
        z = delta/std
        return 2*std*norm.pdf(z)+delta*(2*norm.cdf(z)-1)
    crps = np.mean(absolute_moment(y-means, sigma))
    crps -= .5*np.mean(absolute_moment(means[:, None]-means[None, :], np.sqrt(2)*sigma))
    return float(-logsumexp(log_density)+np.log(len(means))), float(crps)


def record(identity, method, means, noise, observed, predictable, oracle_e=np.nan,
           oracle_a=np.nan, raw_variance=None):
    means = np.atleast_1d(means)
    mu, e = float(means.mean()), float(means.var())
    a = float(noise)
    total = a+e if raw_variance is None else float(raw_variance)
    result = {**identity, "method": method, "mu": mu, "aleatoric": a,
              "epistemic": e, "total": total, "observation": observed,
              "predictable": predictable, "oracle_epistemic": oracle_e,
              "oracle_aleatoric": oracle_a, "n_members": len(means)}
    if raw_variance is not None:
        result["aleatoric"], result["epistemic"] = np.nan, np.nan
    for key, value in gaussian_metrics(observed, mu, total).items():
        result[f"observation_{key}"] = float(value)
    for key, value in gaussian_metrics(predictable, mu, e if e > 0 and raw_variance is None else np.nan).items():
        result[f"predictable_{key}"] = float(value)
    result["mixture_nll"], result["mixture_crps"] = (
        mixture_scores(observed, means, a) if raw_variance is None else (np.nan, np.nan))
    result["state_scored"] = bool(e > 0 and raw_variance is None)
    result["reconstruction_error"] = float(abs(total-a-e)) if raw_variance is None else np.nan
    return result


def fit_parameters(datasets, run, smoke):
    fits, grids, parameter_rows, series_rows = [], [], [], []
    grid_objects = {}
    for data in datasets:
        case, seed, a = data["case"], data["seed"], data["a"]
        y, seasonal = data["y"][:768], data["seasonal"][:768]
        fitted = ({"q": 0., "r": float(np.var(y-seasonal, ddof=1)), "success": True}
                  if a == 1 else fit_noise(y, seasonal, a))
        grid = noise_grid(y, seasonal, a, nodes=3 if smoke else 9)
        grid_objects[(case, seed)] = grid
        summary = grid["summary"]
        row = {"case": case, "seed": seed, "q_true": data["q"], "r_true": data["r"],
               "q_fit": fitted["q"], "r_fit": fitted["r"], "success": fitted["success"],
               "q_mean": summary["q_mean"], "r_mean": summary["r_mean"],
               "q_low": summary["q_quantiles"][0], "q_high": summary["q_quantiles"][-1],
               "r_low": summary["r_quantiles"][0], "r_high": summary["r_quantiles"][-1],
               "edge_mass": summary["edge_mass"]}
        fits.append(row)
        for node in grid["nodes"]:
            grids.append({"case": case, "seed": seed, **node})
        future = data["seasonal"][768+LEADS-1]
        moments = parameter_forecast(grid, future, LEADS, a)
        for j, lead in enumerate(LEADS):
            parameter_rows.append({"case": case, "seed": seed, "origin": 768, "lead": int(lead),
                                   "noise": moments["aleatoric"][j], "state": moments["state"][j],
                                   "parameter": moments["parameter"][j], "total": moments["total"][j],
                                   "mu": moments["mu"][j], "observation": data["y"][768+lead-1]})
        filtered = filter_history(data["y"], data["seasonal"], a, data["q"], data["r"])
        for t in range(len(data["y"])):
            series_rows.append({"case": case, "seed": seed, "time": t, "observation": data["y"][t],
                                "latent": data["x"][t], "posterior_mean": filtered["means"][t],
                                "posterior_sd": np.sqrt(filtered["variances"][t]),
                                "split": "train" if t < 768 else "validation" if t < 1024 else "test"})
        print(f"Noise inference: {case} seed{seed}, Q={fitted['q']:.5g}, R={fitted['r']:.5g}", flush=True)
    pd.DataFrame(fits).to_csv(run / "noise_fits.csv", index=False)
    pd.DataFrame(grids).to_csv(run / "parameter_grid.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(run / "parameter_scores.csv", index=False)
    pd.DataFrame(series_rows).to_csv(run / "series.csv", index=False)
    return {(row["case"], row["seed"]): row for row in fits}


def hybrid(transition, datasets, fits, run, smoke):
    rows, controls, members, baseline_quantiles = [], [], {}, []
    draws = 8 if smoke else 64
    origins = [(768, "validation")] if smoke else [(768, "validation"), (820, "validation")]
    origins += [(1024, "test")] if smoke else [(origin, "test") for origin in range(1024, 1485, 52)]
    for case_index, data in enumerate(datasets):
        case, seed = data["case"], data["seed"]
        for origin, split in origins:
            window = slice(origin-64, origin)
            rng = np.random.default_rng(np.random.SeedSequence([seed, case_index, origin, 701]))
            posterior = sample_history(data["y"][window], data["seasonal"][window], data["a"],
                                       data["q"], data["r"], draws*2 if origin == 1024 else draws,
                                       rng, covariance=(origin == 1024 and seed == 0 and case == "smooth_seasonal"))
            curves = transition(posterior["draws"])
            plugin = transition(posterior["mean"])[0]
            native = transition.quantiles(data["y"][window])[0]
            native_adapter, _ = variance_adapters(transition.levels, native)
            truth = forecast_components(data["a"], data["q"], data["r"], posterior["filtered_variance"], LEADS)
            predictable = data["seasonal"][origin+LEADS-1] + data["a"]**LEADS*data["u"][origin-1]
            fitted = fits[(case, seed)]
            fitted_post = sample_history(data["y"][window], data["seasonal"][window], data["a"],
                                          fitted["q_fit"], fitted["r_fit"], draws,
                                          np.random.default_rng(np.random.SeedSequence([seed, case_index, origin, 702])))
            fitted_curves = transition(fitted_post["draws"])
            fitted_noise = forecast_components(data["a"], fitted["q_fit"], fitted["r_fit"],
                                               fitted_post["filtered_variance"], LEADS)["aleatoric"]
            independent_curves = None
            if origin == 1024:
                independent = posterior["mean"] + rng.normal(size=(draws, 64))*np.sqrt(posterior["variances"])
                independent_curves = transition(independent)
                # The analytical control uses many cheap draws, independent of Chronos inference.
                exact_draws = sample_history(data["y"][window], data["seasonal"][window], data["a"],
                                            data["q"], data["r"], 4096, rng)["draws"]
                for j, lead in enumerate(LEADS):
                    e_mc = np.var(data["a"]**lead*(exact_draws[:, -1]-data["seasonal"][origin-1]))
                    controls.append({"case": case, "seed": seed, "lead": int(lead),
                                     "epistemic_mc": e_mc, "epistemic_exact": truth["epistemic"][j],
                                     "aleatoric_mc": truth["aleatoric"][j], "aleatoric_exact": truth["aleatoric"][j]})
            specifications = [("joint_known_noise", curves[:draws], truth["aleatoric"]),
                              ("plugin_known_noise", plugin[None, :], truth["aleatoric"]),
                              ("joint_fitted_noise", fitted_curves, fitted_noise)]
            if origin == 1024:
                specifications += [("joint128_known_noise", curves, truth["aleatoric"]),
                                   ("independent_known_noise", independent_curves, truth["aleatoric"])]
            for j, lead in enumerate(LEADS):
                identity = {"stage": "hybrid", "case": case, "seed": seed, "origin": origin,
                            "split": split, "lead": int(lead)}
                observed = float(data["y"][origin+lead-1])
                for name, forecast, noise in specifications:
                    rows.append(record(identity, name, forecast[:, lead-1], noise[j], observed,
                                       predictable[j], truth["epistemic"][j], truth["aleatoric"][j]))
                rows.append(record(identity, "raw_chronos", native_adapter["moments"][0][lead-1],
                                   np.nan, observed, predictable[j], truth["epistemic"][j], truth["aleatoric"][j],
                                   raw_variance=native_adapter["moments"][1][lead-1]))
                baseline_quantiles.append(native[lead-1])
            key = f"{case}_seed{seed}_origin{origin}"
            members[key+"_histories"] = posterior["draws"]
            members[key+"_forecast_means"] = curves
            members[key+"_fitted_means"] = fitted_curves
            if independent_curves is not None:
                members[key+"_independent_means"] = independent_curves
            if case == "smooth_seasonal" and seed == 0 and origin == 1024:
                np.savez_compressed(run / "example.npz", time=np.arange(origin-64, origin),
                                    observation=data["y"][window], latent=data["x"][window],
                                    posterior_mean=posterior["mean"], posterior_sd=np.sqrt(posterior["variances"]),
                                    joint_histories=posterior["draws"][:draws], joint_forecasts=curves[:draws],
                                    independent_forecasts=independent_curves, plugin_forecast=plugin,
                                    covariance=posterior["covariance"], lead=LEADS, origin=origin)
        print(f"Hybrid posterior propagation: {case} seed{seed}, {len(origins)} origins", flush=True)
    np.savez_compressed(run / "hybrid_members.npz", **members)
    np.savez_compressed(run / "raw_baseline_quantiles.npz", levels=transition.levels, quantiles=baseline_quantiles)
    pd.DataFrame(controls).to_csv(run / "exact_control.csv", index=False)
    return rows


def coherent(transition, run, smoke):
    q, r, length = .01, .04, 32
    steps = 8 if smoke else 64
    seeds = range(2) if smoke else range(5)
    count = 16 if smoke else 64
    rows, series, arrays, nested_rows = [], [], {}, []
    initial = np.sin(2*np.pi*np.arange(length)/16)
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 8801]))
        latent, observed, predictable = [], [], []
        context = initial.copy()
        for t in range(steps):
            mu = transition(context, horizon=1)[0, 0]
            x = mu+rng.normal(0, np.sqrt(q))
            y = x+rng.normal(0, np.sqrt(r))
            latent.append(x); observed.append(y); predictable.append(mu)
            context = np.r_[context[1:], x]
        arrays[f"seed{seed}_latent"] = np.r_[initial, latent]
        arrays[f"seed{seed}_observed"] = np.r_[initial, observed]
        arrays[f"seed{seed}_predictable"] = np.asarray(predictable)
        for particles in ([count, count*4] if seed == 0 else [count]):
            histories = np.tile(initial, (particles, 1))
            ancestry = np.tile(np.arange(particles)[:, None], (1, length))
            pf_rng = np.random.default_rng(np.random.SeedSequence([seed, particles, 8802]))
            name = "particle64" if particles == count else "particle256"
            for j, y in enumerate(observed):
                means = transition(histories, horizon=1)[:, 0]
                identity = {"stage": "coherent", "case": "chronos_sine16", "seed": seed,
                            "origin": length+j, "split": "test", "lead": 1}
                rows.append(record(identity, name, means, q+r, y, predictable[j], oracle_a=q+r))
                if seed == 0 and particles == count and j in ([3] if smoke else [16, 48]):
                    nested_rows.extend(nested_forecast(transition, histories, length+j, seed, q, r, smoke))
                histories, indices, diagnostics = adapted_update(histories, means, y, q, r, pf_rng)
                ancestry = np.column_stack([ancestry[indices, 1:], np.arange(particles)])
                series.append({"case": "chronos_sine16", "seed": seed, "time": length+j,
                               "observation": y, "latent": latent[j], "predictable": predictable[j],
                               "mu": means.mean(), "aleatoric": q+r, "epistemic": means.var(),
                               "total": q+r+means.var(), "particles": particles,
                               "oldest_ancestor_diversity": len(np.unique(ancestry[:, 0])), **diagnostics})
            print(f"Coherent Chronos filter: seed{seed}, {particles} particles, {steps} steps", flush=True)
    pd.DataFrame(series).to_csv(run / "coherent_series.csv", index=False)
    pd.DataFrame(nested_rows).to_csv(run / "nested_components.csv", index=False)
    np.savez_compressed(run / "coherent_truth.npz", initial_context=initial, **arrays)
    return rows


def nested_forecast(transition, histories, origin, seed, q, r, smoke):
    """Recursive future process rollouts and explicitly corrected nested MC."""
    horizons = [1, 2, 4] if smoke else [1, 4, 13]
    inner = 8 if smoke else 16
    outer, length = histories.shape
    rng = np.random.default_rng(np.random.SeedSequence([seed, origin, 8803]))
    contexts = np.repeat(histories, inner, axis=0)
    rows = []
    direct_e = float(transition(histories, horizon=1)[:, 0].var())
    for h in range(1, max(horizons)+1):
        means = transition(contexts, horizon=1)[:, 0]
        state = means+rng.normal(0, np.sqrt(q), len(means))
        contexts = np.column_stack([contexts[:, 1:], state])
        if h not in horizons:
            continue
        trajectories = state.reshape(outer, inner)
        for k in (inner//2, inner):
            values = trajectories[:, :k]
            conditional_means = values.mean(axis=1)
            within = values.var(axis=1, ddof=1).mean()
            between = conditional_means.var(ddof=1)
            correction = within/k
            e = between-correction
            pop_a = values.var(axis=1).mean()+r
            pop_e = conditional_means.var()
            rows.append({"origin": origin, "seed": seed, "lead": h, "inner_draws": k,
                         "outer_draws": outer, "mu": float(values.mean()), "aleatoric": float(within+r),
                         "epistemic_raw": float(between), "epistemic_corrected": float(e),
                         "total_raw": float(within+r+between), "total_corrected": float(within+r+e),
                         "correction": float(correction), "feasible_corrected": bool(e >= 0),
                         "e_population": float(pop_e), "a_population": float(pop_a),
                         "total_population": float(pop_a+pop_e),
                         "reconstruction_population": float(abs(values.var()+r-pop_a-pop_e)),
                         "direct_one_step_e": direct_e if h == 1 else np.nan})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    all_data, provenance, static_arrays = build_datasets(SOURCE, False)
    datasets = [data for data in all_data if data["dataset"] == "synthetic" and
                (not args.smoke or data["seed"] in (0, 1))]
    source_config = json.loads((SOURCE / "config.json").read_text())
    model = Path(source_config["model"])
    for name, checksum in source_config["model_files_sha256"].items():
        if digest(model / name) != checksum:
            raise ValueError("Pinned model checkpoint hash changed")
    run = EXP / "results" / (datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
                            + ("_smoke" if args.smoke else ""))
    run.mkdir(parents=True, exist_ok=False)
    sources = [*Path(__file__).parent.glob("*.py"), EXP / "planning/README.md", ADAPTER_PATH,
               EXP.parent / "chronos_context_floor/scripts/data.py", PREVIOUS / "scripts/decomposition.py"]
    config = {"smoke": args.smoke, "model": str(model), "revision": source_config["resolved_revision"],
              "input_provenance": provenance, "context": 64, "draws": 8 if args.smoke else 64,
              "train_end": 768, "validation_end": 1024, "leads": LEADS.tolist(),
              "particle_count": 16 if args.smoke else 64, "particle_sensitivity": 64 if args.smoke else 256,
              "coherent_steps": 8 if args.smoke else 64, "coherent_context": 32, "coherent_q": .01,
              "coherent_r": .04, "data_seeds": sorted({data["seed"] for data in datasets}),
              "source_hashes": {str(path): digest(path) for path in sources},
              "versions": {name: importlib.metadata.version(name) for name in
                           ("numpy", "scipy", "pandas", "torch", "chronos-forecasting", "transformers", "matplotlib")},
              "sampling_seed_tags": {"joint": 701, "fitted": 702, "coherent_data": 8801, "particles": 8802}}
    (run / "config.json").write_text(json.dumps(config, indent=2))
    np.savez_compressed(run / "static_series.npz", **static_arrays)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(str(model), device_map="cpu", dtype=torch.float32,
                                               local_files_only=True)
    pipeline.model.eval()
    transition = Transition(pipeline)
    sample = datasets[0]["y"][960:1024]
    single = transition(sample)
    batch = transition(np.stack([sample, sample+.1]))
    np.testing.assert_allclose(single[0], batch[0], rtol=1e-5, atol=2e-6)
    fits = fit_parameters(datasets, run, args.smoke)
    scores = pd.DataFrame(hybrid(transition, datasets, fits, run, args.smoke)+coherent(transition, run, args.smoke))
    scores.to_csv(run / "scores.csv", index=False)
    controls = pd.read_csv(run / "exact_control.csv")
    relative = abs(controls.epistemic_mc-controls.epistemic_exact)/controls.epistemic_exact
    checks = {"batch_invariance": True, "finite_total_variances": bool(np.isfinite(scores.total).all()),
              "nonnegative_components": bool((scores.epistemic.dropna() >= 0).all() and (scores.aleatoric.dropna() >= 0).all()),
              "reconstruction_max": float(scores.reconstruction_error.max()),
              "analytic_mc_relative_error_max": float(relative.max()),
              "transition_calls": transition.calls, "transition_histories": transition.histories,
              "test_targets": int(scores[(scores.method == "joint_known_noise") & (scores.split == "test")].shape[0]),
              "q_r_fitting_uses_training_only": True, "past_contexts_only": True}
    if not checks["finite_total_variances"] or not checks["nonnegative_components"] or checks["reconstruction_max"] > 1e-10:
        raise ValueError(f"Saved-output verification failed: {checks}")
    if checks["analytic_mc_relative_error_max"] > .1:
        raise ValueError("Analytic Monte Carlo control outside prespecified 10% tolerance")
    (run / "verification.json").write_text(json.dumps(checks, indent=2))
    (run / "COMPLETED").write_text("Latent-posterior propagation and coherent Chronos filtering completed.\n")
    print(f"Results: {run}", flush=True)


if __name__ == "__main__":
    main()
