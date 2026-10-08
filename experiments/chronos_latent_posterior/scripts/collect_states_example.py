"""Replay seed 0's coherent filter and retain plot_states-style posterior states.

The prior prediction is computed before y_t. The displayed state posteriors
are computed after y_t: X_t, M_t=g(history), W_t=X_t-M_t, V_t=y_t-X_t.
Their posterior noises are correlated; their marginal variances cannot be
added to reconstruct a posterior observation variance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from particle import adapted_update
from run_experiment import SOURCE, Transition, digest


def collect(run, output_dir=None):
    run = Path(run).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else run
    output_names = ["state_plot_data.csv", "state_plot_posterior.npz", "states_checks.json"]
    if any((output_dir / name).exists() for name in output_names):
        raise FileExistsError("State collection refuses to overwrite previous outputs")
    inputs = [run / name for name in ("config.json", "coherent_truth.npz", "scores.csv")]
    config = json.loads(inputs[0].read_text())
    particles = config["particle_count"]
    q, r = config["coherent_q"], config["coherent_r"]
    steps, length = config["coherent_steps"], config["coherent_context"]
    with np.load(inputs[1]) as data:
        initial = data["initial_context"].copy()
        true_x = data["seed0_latent"][length:].copy()
        observations = data["seed0_observed"][length:].copy()
        true_m = data["seed0_predictable"].copy()
    if len(initial) != length or len(observations) != steps:
        raise ValueError("Stored coherent truth does not match run configuration")
    scores = pd.read_csv(inputs[2])
    baseline = scores[(scores.stage == "coherent") & (scores.seed == 0)
                      & (scores.method == "particle64")].sort_values("origin")
    if len(baseline) != steps:
        raise ValueError("Primary seed-0 coherent score rows are missing")
    np.testing.assert_array_equal(baseline.origin, np.arange(length, length + steps))

    model = Path(config["model"])
    pinned = json.loads((SOURCE / "config.json").read_text())
    for name, expected in pinned["model_files_sha256"].items():
        if digest(model / name) != expected:
            raise ValueError("Pinned model checkpoint hash changed")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(str(model), device_map="cpu",
                                               dtype=torch.float32, local_files_only=True)
    pipeline.model.eval()
    if config.get("transition_kind") == "bounded_tanh":
        from run_bounded import BoundedTransition
        transition = BoundedTransition(pipeline, config["transition_bound"])
    else:
        transition = Transition(pipeline)
    seed_tag = config["sampling_seed_tags"]["particles"]
    rng = np.random.default_rng(np.random.SeedSequence([0, particles, seed_tag]))
    histories = np.tile(initial, (particles, 1))
    rows, members, covariance, priors = [], [], [], []
    names = ["x", "m", "w", "v"]
    x_relation_error, y_relation_error, cancellation_error = 0., 0., 0.
    x_relative_error, y_relative_error, covariance_relative_error = 0., 0., 0.
    posterior_sd_exact = np.sqrt(q * r / (q + r))
    precision_limited_times = []
    for j, observation in enumerate(observations):
        means = transition(histories, horizon=1)[:, 0]
        prior_mean, prior_e = float(means.mean()), float(means.var())
        histories, indices, _ = adapted_update(histories, means, observation, q, r, rng)
        x = histories[:, -1]
        m = means[indices]
        w, v = x - m, observation - x
        states = np.stack([x, m, w, v], axis=1)
        # Center around one member first: averaging huge identical numbers can
        # otherwise create a false nonzero variance when the replay diverges.
        offsets = states - states[0]
        offset_mean = offsets.mean(axis=0)
        state_mean = states[0] + offset_mean
        centered = offsets - offset_mean
        cov = centered.T @ centered / particles
        x_error, y_error = np.abs(x - m - w), np.abs(observation - x - v)
        var_y_error = float(abs(cov[0, 0] + cov[3, 3] + 2 * cov[0, 3]))
        x_relation_error = max(x_relation_error, float(x_error.max()))
        y_relation_error = max(y_relation_error, float(y_error.max()))
        cancellation_error = max(cancellation_error, var_y_error)
        x_relative_error = max(x_relative_error, float(np.max(x_error / (1 + abs(x) + abs(m) + abs(w)))))
        y_relative_error = max(y_relative_error, float(np.max(y_error / (1 + abs(observation) + abs(x) + abs(v)))))
        covariance_relative_error = max(covariance_relative_error,
                                        var_y_error / (1 + cov[0, 0] + cov[3, 3]))
        resolution = float(np.spacing(np.max(np.abs(states))))
        if resolution >= posterior_sd_exact:
            precision_limited_times.append(length + j)
        truths = [true_x[j], true_m[j], true_x[j] - true_m[j], observation - true_x[j]]
        row = {"time": length + j, "observation": float(observation),
               "prediction_mean": prior_mean, "prediction_total_sd": np.sqrt(q + r + prior_e),
               "prediction_epistemic_sd": np.sqrt(prior_e),
               "prediction_aleatoric_sd": np.sqrt(q + r),
               "state_numeric_resolution": resolution,
               "precision_limited": resolution >= posterior_sd_exact}
        for k, name in enumerate(names):
            row.update({f"{name}_mean": float(state_mean[k]),
                        f"{name}_sd": float(np.sqrt(cov[k, k])),
                        f"{name}_truth": float(truths[k])})
        rows.append(row)
        members.append(states)
        covariance.append(cov)
        priors.append([prior_mean, prior_e, q + r, q + r + prior_e])
    priors = np.array(priors)
    references = baseline[["mu", "epistemic", "aleatoric", "total"]].to_numpy()
    replay_errors = np.max(np.abs(priors - references), axis=0)
    np.testing.assert_allclose(priors, references, rtol=1e-5, atol=2e-6)
    if max(x_relative_error, y_relative_error, covariance_relative_error) > 1e-12:
        raise ValueError("Posterior state relations or covariance cancellation failed")
    members, covariance = np.array(members), np.array(covariance)
    if members.shape != (steps, particles, 4) or not np.isfinite(members).all():
        raise ValueError("Invalid retained posterior states")
    input_hashes = {str(path): digest(path) for path in inputs}
    scripts = [Path(__file__), Path(__file__).with_name("run_experiment.py"),
               Path(__file__).with_name("particle.py")]
    checks = {
        "run": str(run), "seed": 0, "particles": particles, "steps": steps,
        "q": q, "r": r, "model": str(model), "revision": config["revision"],
        "particle_seed_sequence": [0, particles, seed_tag],
        "state_order": names, "posterior_conditioning": "observations through current time t",
        "prior_conditioning": "observations through previous time t-1",
        "x_equals_m_plus_w_max_error": x_relation_error,
        "y_equals_x_plus_v_max_error": y_relation_error,
        "posterior_var_y_covariance_cancellation_max_error": cancellation_error,
        "x_relation_relative_max_error": x_relative_error,
        "y_relation_relative_max_error": y_relative_error,
        "posterior_var_y_covariance_relative_max_error": covariance_relative_error,
        "relation_relative_tolerance": 1e-12,
        "centering": "subtract first particle before averaging offsets",
        "exact_conditional_posterior_sd": posterior_sd_exact,
        "precision_limited_times": precision_limited_times,
        "precision_limited_criterion": "floating spacing of largest state >= exact conditional posterior SD",
        "replay_absolute_max_errors": dict(zip(["mu", "epistemic", "aleatoric", "total"],
                                                replay_errors.tolist())),
        "replay_tolerance": {"rtol": 1e-5, "atol": 2e-6},
        "finite_states": True, "transition_calls": transition.calls,
        "input_sha256": input_hashes,
        "script_sha256": {str(path.resolve()): digest(path) for path in scripts},
        "model_files_sha256": pinned["model_files_sha256"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / output_names[0], index=False)
    np.savez_compressed(output_dir / output_names[1], time=np.arange(length, length + steps),
                        members=members, covariance=covariance, state_order=np.array(names),
                        prior_components=priors)
    checks["output_sha256"] = {name: digest(output_dir / name) for name in output_names[:2]}
    (output_dir / output_names[2]).write_text(json.dumps(checks, indent=2))
    print(f"States saved: {output_dir}; replay maximum error {replay_errors.max():.3g}", flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    collect(args.run, args.output_dir)


if __name__ == "__main__":
    main()
