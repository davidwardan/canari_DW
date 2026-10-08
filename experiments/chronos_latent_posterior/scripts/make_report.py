"""Render the latent-posterior experiment from its immutable saved run outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronos_latent_posterior_mpl")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (6.5, 3.5)
BLUE, GREEN, RED = "tab:blue", "tab:green", "tab:red"


def plot_defaults():
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex", "font.family": "serif", "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
        "lines.linewidth": 1, "figure.figsize": SINGLE_COL, "font.size": 9,
        "savefig.dpi": 300,
    })


def legend(ax, ncol=2):
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=ncol,
              columnspacing=.9, handlelength=1.5)


def save_figure(fig, folder, name):
    fig.tight_layout()
    for suffix in ("pdf", "pgf", "png"):
        fig.savefig(folder / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    return name


def markdown_table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(
            "—" if pd.isna(value) else f"{value:.4g}" if isinstance(value, (float, np.floating))
            else str(value) for value in row) + " |")
    return "\n".join(lines)


def figure_entry(name, caption):
    return {"name": name, "caption": caption}


LABELS = {
    "raw_chronos": "Raw Chronos", "joint_known_noise": "Joint posterior",
    "plugin_known_noise": "Posterior mean", "joint_fitted_noise": "Fitted noise",
    "independent_known_noise": "Independent history", "joint128_known_noise": "128 draws",
    "particle64": "64 particles", "particle256": "256 particles",
}
MAIN_HYBRID = ["raw_chronos", "joint_known_noise", "plugin_known_noise", "joint_fitted_noise"]
METRICS = ["mu", "aleatoric", "epistemic", "total", "oracle_epistemic", "oracle_aleatoric",
           "observation_squared_error", "observation_absolute_error", "observation_nll",
           "observation_crps", "observation_coverage90", "predictable_coverage90", "predictable_crps"]


def aggregate_scores(scores):
    """Origins average within seed/lead; seeds, leads and cases receive equal weight."""
    keys = ["stage", "case", "seed", "lead", "method"]
    group = scores.groupby(keys, dropna=False)
    by_lead = group[METRICS].mean()
    by_lead["n_targets"] = group.size()
    by_lead["n_predictable"] = group.predictable_coverage90.count()
    frames = {"summary_by_seed_lead": by_lead.reset_index()}
    for name, keys in [
        ("summary_by_seed", ["stage", "case", "seed", "method"]),
        ("summary_by_case", ["stage", "case", "method"]),
        ("summary_overall", ["stage", "method"]),
    ]:
        group = list(frames.values())[-1].groupby(keys, dropna=False)
        frame = group[METRICS].mean()
        frame["n_targets"] = group.n_targets.sum()
        frame["n_predictable"] = group.n_predictable.sum()
        frames[name] = frame.reset_index()
    for frame in frames.values():
        frame["observation_rmse"] = np.sqrt(frame.observation_squared_error)
    seeds = frames["summary_by_seed"].groupby(["stage", "case", "method"], dropna=False)
    dispersion = seeds[METRICS + ["observation_rmse"]].std(ddof=1)
    dispersion["n_data_seeds"] = seeds.size()
    frames["data_seed_dispersion"] = dispersion.reset_index()
    return frames


def paired_diagnostics(scores):
    keys = ["stage", "case", "seed", "origin", "lead"]
    pairs = []
    for reference, alternative in [("joint_known_noise", "independent_known_noise"),
                                    ("joint_known_noise", "joint128_known_noise"),
                                    ("particle64", "particle256")]:
        left = scores[scores.method == reference]
        right = scores[scores.method == alternative]
        pair = left.merge(right, on=keys, suffixes=("_reference", "_alternative"), validate="one_to_one")
        if pair.empty:
            continue
        pair["comparison"] = alternative + " versus " + reference
        pair["epistemic_ratio"] = pair.epistemic_alternative / pair.epistemic_reference.where(
            pair.epistemic_reference > 0)
        pair["mean_difference"] = pair.mu_alternative - pair.mu_reference
        pair["crps_difference"] = pair.observation_crps_alternative - pair.observation_crps_reference
        pairs.append(pair)
    return pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()


def calibration(scores):
    """Gaussian moment intervals; repeated/overlapping origins are not iid replicates."""
    rows = []
    levels = np.array([.5, .682689492, .8, .9, .95])
    for target, variance in [("observation", "total"), ("predictable", "epistemic")]:
        frame = scores[np.isfinite(scores[variance]) & (scores[variance] > 0)].copy()
        for level in levels:
            radius = NormalDist().inv_cdf((1 + level) / 2) * np.sqrt(frame[variance])
            covered = (np.abs(frame[target] - frame.mu) <= radius).astype(float)
            measured = frame[["stage", "case", "seed", "method"]].copy()
            measured["coverage"] = covered
            # Average leads/origins together within each seed (complete main cohorts have equal leads).
            seed_values = measured.groupby(["stage", "case", "seed", "method"]).coverage.mean()
            for (stage, case, seed, method), value in seed_values.items():
                rows.append({"stage": stage, "case": case, "seed": seed, "method": method,
                             "target": target, "nominal": level, "coverage": value})
    return pd.DataFrame(rows)


def stability_summary(coherent, config):
    values = coherent.copy()
    q_sd, r_sd = np.sqrt(config["coherent_q"]), np.sqrt(config["coherent_r"])
    values["process_resolution_fraction"] = np.spacing(values.predictable.abs()) >= q_sd
    values["measurement_resolution_fraction"] = np.spacing(values.latent.abs()) >= r_sd
    values["forecast_resolution_fraction"] = np.spacing(values.mu.abs()) >= q_sd
    values["chronos_input_resolution_fraction"] = np.spacing(values.latent.abs().astype(np.float32)) >= q_sd
    values["forecast_error"] = values.mu - values.predictable
    return values.groupby(["seed", "particles"]).agg(
        max_abs_latent=("latent", lambda frame: frame.abs().max()),
        max_abs_observation=("observation", lambda frame: frame.abs().max()),
        max_abs_forecast=("mu", lambda frame: frame.abs().max()),
        max_abs_transition_mean_error=("forecast_error", lambda frame: frame.abs().max()),
        process_resolution_fraction=("process_resolution_fraction", "mean"),
        measurement_resolution_fraction=("measurement_resolution_fraction", "mean"),
        forecast_resolution_fraction=("forecast_resolution_fraction", "mean"),
        chronos_input_resolution_fraction=("chronos_input_resolution_fraction", "mean")).reset_index()


def make_figures(run, scores, summary, folder, coherent, paired, calibration_, config, stability):
    plot_defaults()
    entries = []
    bounded = config.get("transition_kind") == "bounded_tanh"
    unstable = (stability[[column for column in stability if "resolution_fraction" in column]] > 0).any().any()
    model_description = (f"the exploratory bounded map g={config['transition_bound']:g} "
                         f"tanh(Chronos median/{config['transition_bound']:g})" if bounded
                         else "the raw frozen Chronos median map")

    def save(fig, name, caption):
        entries.append(figure_entry(save_figure(fig, folder, name), caption))

    with np.load(run / "example.npz", allow_pickle=False) as example:
        time, observation, latent = example["time"], example["observation"], example["latent"]
        mean, sd = example["posterior_mean"], example["posterior_sd"]
        histories, forecasts = example["joint_histories"], example["joint_forecasts"]
        plugin, covariance = (example[key] for key in ("plugin_forecast", "covariance"))
        fig, ax = plt.subplots()
        ax.plot(time, observation, color=RED, label="Observed")
        ax.plot(time, latent, color="black", linestyle="--", label="Latent truth")
        ax.set_xlabel("Time (weeks)")
        ax.set_ylabel("Signal (a.u.)")
        legend(ax)
        save(fig, "01_observations_and_latent", "The observed series contains measurement noise. "
             "The dashed latent truth is available only because this is a synthetic control; inference never sees it. "
             "This fixed smooth-case, seed-0 context is also used in Figures 2–4.")

        fig, ax = plt.subplots()
        for draw in histories[:6]:
            ax.plot(time, draw, color=BLUE, alpha=.3, linewidth=.7)
        ax.plot(time, mean, color=BLUE, label="Posterior mean")
        ax.fill_between(time, mean-sd, mean+sd, color=BLUE, alpha=.3, label=r"State $\pm1\sigma$")
        ax.plot(time, latent, color="black", linestyle="--", label="Latent truth")
        ax.set_xlabel("Time (weeks)")
        ax.set_ylabel("Latent signal (a.u.)")
        legend(ax)
        save(fig, "02_joint_posterior_histories", "Six correlated latent histories and marginal "
             "plus/minus one-standard-deviation bands from the exact 64-observation-window AR posterior. "
             "The prior is reinitialized at the window start; earlier observations are not conditioned on. Past values are smoothed "
             "using observations only through the forecast origin; no later observations enter this posterior. "
             "Each entire history, rather than its mean alone, becomes a Chronos input.")

        correlation = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        fig, ax = plt.subplots()
        image = ax.imshow(correlation, origin="lower", extent=(time[0], time[-1], time[0], time[-1]),
                          vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
        ax.set_xlabel("History time (weeks)")
        ax.set_ylabel("History time (weeks)")
        fig.colorbar(image, ax=ax, label="Posterior correlation", fraction=.05, pad=.04)
        save(fig, "03_history_correlation", "Joint posterior covariance preserves temporal dependence. "
             "Independently sampling each history point replaces this matrix by its diagonal, changing the "
             "forecast-mean distribution even though every marginal uncertainty is retained.")

        fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
        selected_leads = (1, 13)
        for ax, lead in zip(axes, selected_leads):
            values = forecasts[:, lead-1]
            ax.hist(values, bins=min(12, max(5, len(values)//4)), density=True,
                    color=BLUE, alpha=.45, label="Joint histories")
            ax.axvline(plugin[lead-1], color="black", linestyle="--", label="Mean-history input")
            ax.axvline(values.mean(), color=BLUE, label="Mean of forecasts")
            ax.set_xlabel(f"Lead-{lead} forecast mean (a.u.)")
            ax.set_ylabel(r"Density (a.u.$^{-1}$)")
        legend(axes[0], ncol=1)
        save(fig, "04_histories_to_forecasts", "Each frozen Chronos median forecast is a deterministic "
             "function of one latent-history draw. The spread across those outputs estimates state uncertainty. "
             "The dashed line forecasts the posterior-mean history; it need not equal the average forecast "
             "because Chronos is nonlinear. These are uncertainty about forecast means, not draws of future observation noise.")

    control = pd.read_csv(run / "exact_control.csv")
    chosen_case = next((case for case in control.case.unique() if "smooth" in case), control.case.iloc[0])
    control_case = control[control.case == chosen_case].groupby("lead").mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    for ax, component, color in zip(axes, ("epistemic", "aleatoric"), (BLUE, GREEN)):
        if component == "epistemic":
            ax.plot(control_case.index, control_case.epistemic_mc, color=color, label="Monte Carlo")
            ax.plot(control_case.index, control_case.epistemic_exact, color="black", linestyle="--", label="Exact AR")
        else:
            ax.plot(control_case.index, control_case.aleatoric_exact, color=color, label="Noise formula")
        ax.set_xlabel("Forecast lead (weeks)")
        ax.set_ylabel(component.capitalize() + r" variance (a.u.$^2$)")
    legend(axes[0])
    legend(axes[1], ncol=1)
    save(fig, "05_exact_linear_control", "The analytical AR control checks the machinery before "
         "interpreting Chronos results. Current-state uncertainty is $a^{2h}P_t$; future noise is "
         r"$R+Q\sum_{j=0}^{h-1}a^{2j}$. Only the left panel uses Monte Carlo; the right panel "
         "shows the known analytical noise formula. Values average within data seed and then across seeds. "
         f"Displayed case: {chosen_case}. All cases remain in exact_control.csv.")

    hybrid = summary["summary_by_seed_lead"].query("stage == 'hybrid' and method == 'joint_known_noise'")
    fig, axes = plt.subplots(2, 2, figsize=DOUBLE_COL)
    panel_cases = list(hybrid.case.unique())
    for index, (ax, case) in enumerate(zip(axes.flat, panel_cases)):
        values = hybrid[hybrid.case == case].groupby("lead").mean(numeric_only=True)
        ax.plot(values.index, values.epistemic, color=BLUE, label="Chronos map")
        ax.plot(values.index, values.oracle_epistemic, color="black", linestyle="--", label="Exact AR")
        ax.set_xlabel("Forecast lead (weeks)")
        ax.set_ylabel(r"State variance (a.u.$^2$)")
        ax.text(.04, .22 if "static" in case else .94, f"({chr(97+index)})",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.set_ylim(0, 1.12 * max(values.epistemic.max(), values.oracle_epistemic.max()))
    for ax in list(axes.flat)[len(hybrid.case.unique()):]:
        ax.set_axis_off()
    legend(axes.flat[0])
    save(fig, "06_hybrid_state_variance", "The AR-derived posterior is propagated through Chronos "
         "at each horizon. Blue and dashed curves condition on the same past observations but use different "
         "forecast functions. Their disagreement measures sensitivity/dynamics mismatch; it is not by itself "
         "Monte Carlo failure or recovery of a Chronos weight posterior. Panel labels identify the synthetic "
         "cases: " + "; ".join(f"({chr(97+i)}) {case.replace('_', ' ')}" for i, case in enumerate(panel_cases)) +
         ". Vertical scales differ to keep small components readable.")

    selected = coherent[(coherent.seed == coherent.seed.min()) & (coherent.particles == coherent.particles.min())]
    selected = selected.sort_values("time")
    x = selected.time.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    ax = axes[0]
    mean, sd = selected.posterior_mean.to_numpy(), selected.posterior_sd.to_numpy()
    ax.plot(x, selected.observation, color=RED, label="Observed")
    ax.plot(x, selected.latent, color="black", linestyle="--", label="Latent truth")
    ax.plot(x, mean, color=BLUE, label="Filtered mean")
    ax.fill_between(x, mean-sd, mean+sd, color=BLUE, alpha=.3, label=r"State $\pm1\sigma$")
    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel("Latent signal (a.u.)")
    legend(ax, ncol=2)
    ax = axes[1]
    mean = selected.mu.to_numpy()
    epi, total = np.sqrt(selected.epistemic.to_numpy()), np.sqrt(selected.total.to_numpy())
    ax.fill_between(x, mean-total, mean+total, color="0.75", alpha=.4, label=r"Total $\pm1\sigma$")
    ax.fill_between(x, mean-epi, mean+epi, color=BLUE, alpha=.3, label=r"State $\pm1\sigma$")
    ax.plot(x, mean, color=BLUE, label="Forecast mean")
    ax.plot(x, selected.predictable, color="black", linestyle="--", label="True transition")
    ax.plot(x, selected.observation, color=RED)
    ax.set_xlabel("Predicted time (weeks)")
    ax.set_ylabel("Signal (a.u.)")
    legend(ax, ncol=2)
    save(fig, "07_coherent_filter_and_forecast", "The self-consistent experiment generates "
         "truth with " + model_description + " also used by the particle filter. Left: after "
         "assimilating each observation, particles estimate the latent state. Right: before assimilation, "
         "the state component covers uncertainty about the true transition mean and the total component "
         "adds future process and observation noise. All forecast bands are plus/minus one standard "
         "deviation; they are not 90% intervals. This is a fixed seed-0 example, not an average trajectory. " +
         ("The raw run fails stability: large scale growth and grossly incorrect filter forecasts make "
          "the narrow bands unsuitable as evidence of calibration." if unstable else
          "The bounded run is an exploratory new dynamic model, not a decomposition of original Chronos quantiles." if bounded else ""))

    fig, ax = plt.subplots()
    e = selected.epistemic.to_numpy()
    q = np.full_like(e, .01)
    r = np.full_like(e, .04)
    ax.fill_between(x, 0, e, color=BLUE, alpha=.3, label="State")
    ax.fill_between(x, e, e+q, color=GREEN, alpha=.3, label="Process noise")
    ax.fill_between(x, e+q, e+q+r, color="0.7", alpha=.4, label="Measurement noise")
    ax.plot(x, selected.total, color="black", label="Total")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Predicted time (weeks)")
    ax.set_ylabel(r"Variance (a.u.$^2$)")
    legend(ax, ncol=2)
    save(fig, "08_coherent_variance_accounting", "Variance components add: state uncertainty plus "
         "known process variance $Q=0.01$ plus known measurement variance $R=0.04$. State uncertainty "
         "changes with the observations and the nonlinear Chronos map. The aleatoric part is exactly "
         "$Q+R=0.05$ in this one-step model. Standard deviations and band widths do not add. " +
         ("In the unstable raw run, a small finite-particle state variance can accompany a catastrophically "
          "wrong forecast after ancestry collapse; it must not be read as reliable certainty." if unstable else ""))

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    convergence = paired[paired.comparison == "particle256 versus particle64"].sort_values("origin")
    primary_particles = int(convergence.n_members_reference.iloc[0])
    sensitivity_particles = int(convergence.n_members_alternative.iloc[0])
    for ax, key, label in zip(axes, ("epistemic", "mu"),
                              (r"State variance (a.u.$^2$)", "Forecast mean (a.u.)")):
        ax.plot(convergence.origin, convergence[key+"_reference"], color=BLUE, label=f"{primary_particles} particles")
        ax.plot(convergence.origin, convergence[key+"_alternative"], color="black", linestyle="--", label=f"{sensitivity_particles} particles")
        ax.set_xlabel("Predicted time (weeks)")
        ax.set_ylabel(label)
    legend(axes[0])
    save(fig, "09_particle_count_sensitivity", "Two particle counts use the same seed-0 observations "
         "and origins, so changes are paired numerical sensitivity. Agreement supports feasibility; "
         f"{primary_particles} versus {sensitivity_particles} particles on one data seed does not prove convergence to the exact nonlinear "
         "posterior. Frozen Chronos weights are identical in both runs.")

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    for ax, stage, methods in [(axes[0], "hybrid", ["joint_known_noise", "joint_fitted_noise"]),
                                (axes[1], "coherent", ["particle64"])]:
        for method, style, color in zip(methods, ("-", "--", ":"), (BLUE, "0.45", GREEN)):
            values = calibration_[(calibration_.stage == stage) & (calibration_.method == method)
                                  & (calibration_.target == "predictable")]
            values = values.groupby(["case", "nominal"]).coverage.mean().groupby("nominal").mean()
            ax.plot(100*values.index, 100*values, color=color, linestyle=style, label=LABELS[method])
        ax.plot([0, 100], [0, 100], color="black", linestyle=":", label="Calibrated")
        ax.set_xlim(45, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Nominal coverage (percent)")
        ax.set_ylabel("Predictable coverage (percent)")
        legend(ax, ncol=1)
    save(fig, "10_state_interval_calibration", "Left: hybrid state intervals against the AR "
         "predictable target; dynamics mismatch can reduce coverage. Right: coherent Chronos state "
         "intervals against the actual Chronos transition mean. These Gaussian moment intervals are "
         "diagnostics of the particle approximation and moment summary. Rates average within data seed "
         "then case. Zero-variance rows are excluded; repeated origins and horizons are not independent trials.")

    fits = pd.read_csv(run / "noise_fits.csv")
    cases = list(fits.case.unique())
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    for ax, component, color in zip(axes, ("q", "r"), (GREEN, "0.4")):
        for index, case in enumerate(cases):
            frame = fits[fits.case == case]
            offsets = np.linspace(-.15, .15, len(frame))
            ax.plot(index+offsets, frame[component+"_fit"], linestyle="none", marker="o", markersize=3,
                    color=color, label="Training fit" if index == 0 else None)
            ax.plot([index-.2, index+.2], [frame[component+"_true"].iloc[0]]*2,
                    color="black", linestyle="--", label="Truth" if index == 0 else None)
        ax.set_xticks(range(len(cases)), [case.replace("_", "\n") for case in cases])
        ax.set_ylabel(("Process" if component == "q" else "Measurement") + r" variance (a.u.$^2$)")
        ax.set_ylim(bottom=0)
    legend(axes[0])
    save(fig, "11_training_noise_fits", "Each dot is one data seed's training-prefix innovation-likelihood "
         "fit with known AR coefficient; dashed segments show generating noise variances. Test observations "
         "never select or fit noise. These point estimates have no parameter posterior; variation across "
         "data seeds is not an epistemic variance for an individual prediction.")

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    main_draws = int(scores.loc[scores.method == "joint_known_noise", "n_members"].iloc[0])
    more_draws = int(scores.loc[scores.method == "joint128_known_noise", "n_members"].iloc[0])
    for ax, alternative, label in [(axes[0], "independent_known_noise", "Independent / joint"),
                                    (axes[1], "joint128_known_noise", f"{more_draws} draws / {main_draws} draws")]:
        values = paired[paired.comparison.str.startswith(alternative)]
        for case in values.case.unique():
            curve = values[values.case == case].groupby("lead").epistemic_ratio.mean()
            ax.plot(curve.index, curve, label=case.replace("_", " "))
        ax.axhline(1, color="black", linestyle="--")
        ax.set_xlabel("Forecast lead (weeks)")
        ax.set_ylabel(label + " state-variance ratio")
        legend(ax, ncol=1)
    save(fig, "12_history_and_draw_sensitivity", "Paired first-origin diagnostics. Left: replacing "
         "joint histories with independently drawn marginals changes forecast-mean variance. Right: "
         "using more joint-history draws shows Monte Carlo sensitivity. Ratios average across data seeds "
         "on the same case/lead/origin cohort. This figure is not an unpaired comparison over all test origins.")

    if (run / "parameter_scores.csv").exists():
        parameters = pd.read_csv(run / "parameter_scores.csv")
        fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
        parameter_cases = list(parameters.case.unique()[:2])
        for index, (ax, case) in enumerate(zip(axes, parameter_cases)):
            frame = parameters[parameters.case == case].groupby("lead").mean(numeric_only=True)
            ax.stackplot(frame.index, frame.noise, frame.state, frame.parameter,
                         colors=[GREEN, BLUE, "0.6"], alpha=.3,
                         labels=["Future noise", "Current state", "Parameters"])
            ax.plot(frame.index, frame.total, color="black", label="Total")
            ax.set_xlabel("Forecast lead (weeks)")
            ax.set_ylabel(r"Variance (a.u.$^2$)")
            ax.text(.04, .88, f"({chr(97+index)})", transform=ax.transAxes, va="top")
            ax.set_ylim(bottom=0)
        legend(axes[0], ncol=2)
        save(fig, "13_parameter_posterior_components", "An additional exact-AR diagnostic integrates "
             "over the saved training-only Q/R posterior grid at the first validation forecast. "
             "The same training prefix supplies both parameter weights and conditional state posteriors. "
             "The law of total variance retains "
             "future noise, uncertainty about current state conditional on parameters, and variance "
             "of conditional forecast means across parameters. This is a separate AR parameter "
             "experiment; the coherent Chronos primary run keeps Q/R known. " +
             "; ".join(f"({chr(97+i)}) {case.replace('_', ' ')}" for i, case in enumerate(parameter_cases)) +
             ". Panel scales may differ.")
    if (run / "parameter_grid.csv").exists():
        grid = pd.read_csv(run / "parameter_grid.csv")
        case = next(case for case in grid.case.unique() if grid[grid.case == case].q.nunique() > 1)
        selected = grid[(grid.case == case) & (grid.seed == grid.seed.min())]
        weights = selected.pivot(index="r", columns="q", values="weight")
        log_q, log_r = np.log(weights.columns.to_numpy()), np.log(weights.index.to_numpy())
        q_edges = np.exp(np.r_[log_q[0]-(log_q[1]-log_q[0])/2,
                               (log_q[1:]+log_q[:-1])/2, log_q[-1]+(log_q[-1]-log_q[-2])/2])
        r_edges = np.exp(np.r_[log_r[0]-(log_r[1]-log_r[0])/2,
                               (log_r[1:]+log_r[:-1])/2, log_r[-1]+(log_r[-1]-log_r[-2])/2])
        fig, ax = plt.subplots()
        image = ax.pcolormesh(q_edges, r_edges, weights.to_numpy(), shading="flat", cmap="Blues")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(weights.columns.min(), weights.columns.max())
        ax.set_ylim(weights.index.min(), weights.index.max())
        ax.set_xlabel(r"Process variance $Q$ (a.u.$^2$)")
        ax.set_ylabel(r"Measurement variance $R$ (a.u.$^2$)")
        fig.colorbar(image, ax=ax, label="Posterior node mass", fraction=.05, pad=.04)
        save(fig, "14_noise_parameter_posterior", f"Training-only Q/R posterior node mass for {case}, seed 0. "
             "The fixed log-spaced grid has equal prior node mass: Q from 0.0001 to 1 and R from 0.005 to 1 a.u.². "
             "Concentration or ridges expose "
             "what the training observations identify under the known AR dynamics. A coarse grid and "
             "its bounds can affect results; posterior mass on grid edges is a truncation diagnostic. "
             "This is probability mass at grid nodes, not a continuous posterior density.")
    if (run / "nested_components.csv").exists():
        nested = pd.read_csv(run / "nested_components.csv")
        origin = int(nested.origin.min())
        selected = nested[nested.origin == origin]
        fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
        for inner_count, style in zip(sorted(selected.inner_draws.unique()), ("-", "--")):
            frame = selected[selected.inner_draws == inner_count].sort_values("lead")
            axes[0].plot(frame.lead, frame.epistemic_corrected, color=BLUE, linestyle=style,
                         label=f"{int(inner_count)} future draws")
            axes[1].plot(frame.lead, frame.aleatoric, color=GREEN, linestyle=style,
                         label=f"{int(inner_count)} future draws")
        reference = selected[selected.lead == 1].iloc[0]
        axes[0].plot(1, reference.direct_one_step_e, linestyle="none", marker="o", markersize=4,
                     markerfacecolor="none", color="black", label="Direct one-step")
        axes[1].plot(1, .05, linestyle="none", marker="o", markersize=4,
                     markerfacecolor="none", color="black", label="Known one-step")
        axes[0].axhline(0, color="black", linestyle=":")
        for ax, label in zip(axes, ("Corrected state", "Future noise")):
            ax.set_xlabel("Forecast lead (weeks)")
            ax.set_ylabel(label + r" variance (a.u.$^2$)")
        legend(axes[0], ncol=1)
        legend(axes[1], ncol=1)
        save(fig, "15_recursive_nonlinear_components", f"Recursive Chronos diagnostic at seed-0 origin {origin}. "
             "Each current-history particle starts multiple future process-noise trajectories; measurement "
             "variance is added only at the scored future observation. Finite inner draws inflate variance "
             "across conditional sample means. The state estimator subtracts mean within-history process "
             "variance divided by the inner draw count. Negative corrected values are retained as Monte "
             "Carlo warnings. Open markers show the direct one-step finite-particle state variance and "
             "known Q+R, which require no nested simulation. The nested state estimator uses outer sample "
             "variance (ddof=1), while the direct marker uses empirical population variance (ddof=0); "
             "their normalization differs by M/(M-1) for M outer histories. These few paired origins do not extend the primary one-step calibration claim "
             "to multiple steps. Nonlinear state/noise interactions are assigned to future noise by conditioning order.")
    if (run / "state_plot_data.csv").exists():
        states = pd.read_csv(run / "state_plot_data.csv").sort_values("time")
        if unstable and not bounded:
            states = states.iloc[:24]
        time = states.time.to_numpy()
        fig, axes = plt.subplots(5, 1, figsize=(6.5, 8.5), sharex=True)
        mean, sd = states.prediction_mean.to_numpy(), states.prediction_total_sd.to_numpy()
        axes[0].plot(time, states.observation, color=RED, label="Observed")
        axes[0].plot(time, mean, color=BLUE, label="Forecast mean")
        axes[0].fill_between(time, mean-sd, mean+sd, color="0.7", alpha=.3, label=r"Total $\pm1\sigma$")
        axes[0].set_ylabel("Observation (a.u.)")
        legend(axes[0], ncol=3)
        for ax, key, label in zip(axes[1:], ("x", "m", "w", "v"),
                                  (r"$X$ (signal)", r"$M$ (transition)", r"$W$ (process)", r"$V$ (measurement)")):
            mean, sd = states[key+"_mean"].to_numpy(), states[key+"_sd"].to_numpy()
            ax.plot(time, mean, color=BLUE, label="Posterior mean")
            ax.fill_between(time, mean-sd, mean+sd, color=BLUE, alpha=.3, label=r"Posterior $\pm1\sigma$")
            ax.plot(time, states[key+"_truth"], color="black", linestyle="--", label="Truth")
            ax.set_ylabel(label + "\n(a.u.)")
        legend(axes[1], ncol=3)
        axes[-1].set_xlabel("Time (weeks)")
        save(fig, "16_states_posterior", "A plot_states-style view of the same seed-0 coherent Chronos "
             "episode using " + model_description + ". " +
             ("Only the first 24 steps are displayed as an early illustration, because later raw "
              "dynamics/filter forecasts become unstable. All 64 steps remain in the CSV and statistical "
              "tables; this visual prefix is not a calibration cohort. " if unstable and not bounded else "") +
             "Top: the observation forecast before seeing that observation, with its total "
             "plus/minus one-standard-deviation band. Four state panels: posterior means and plus/minus "
             "one-standard-deviation bands AFTER assimilation, for signal X, transition mean M, realized "
             "process shock W=X-M, and realized measurement noise V=Y-X. These are joint posterior "
             "variables, not four independent uncertainty components. Their posterior covariances preserve "
             "X=M+W and the fixed observed value Y=X+V; standard deviations do not add. Uncertainty "
             "about a realized past shock differs from the known future aleatoric variances Q and R. "
             "All blue bands indicate posterior uncertainty about the displayed variable; future noise "
             "is separately shown in Figure8. State panels share the posterior legend above X.")
    return entries


def verify_saved_components(run, scores):
    declared = scores[np.isfinite(scores.epistemic) & np.isfinite(scores.aleatoric)]
    if (declared[["epistemic", "aleatoric"]] < 0).any().any():
        raise ValueError("Declared variance components must be nonnegative")
    np.testing.assert_allclose(declared.total, declared.epistemic + declared.aleatoric,
                               rtol=1e-10, atol=1e-12)
    coherent = declared[declared.stage == "coherent"]
    np.testing.assert_allclose(coherent.aleatoric, .05, rtol=1e-12)
    checks = {"saved_components_sum_to_total": True,
              "saved_components_nonnegative": True,
              "coherent_known_aleatoric_is_0.05": True,
              "report_uses_test_split_only": bool((scores.split == "test").all()),
              "forecast_rows": len(scores),
              "data_seeds": sorted(int(seed) for seed in scores.seed.unique()),
              "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "raw_input_sha256": {name: hashlib.sha256((run/name).read_bytes()).hexdigest()
                  for name in ("config.json", "scores.csv", "example.npz", "exact_control.csv",
                               "coherent_series.csv", "noise_fits.csv", "parameter_grid.csv",
                               "parameter_scores.csv", "nested_components.csv", "state_plot_data.csv") if (run/name).exists()}}
    particle_series = pd.read_csv(run / "coherent_series.csv")
    checks["particle_ess_within_normalized_weight_bounds"] = bool(
        ((particle_series.ess >= 1-1e-8) & (particle_series.ess <= particle_series.particles+1e-8)).all())
    if (run / "parameter_scores.csv").exists():
        parameters = pd.read_csv(run / "parameter_scores.csv")
        np.testing.assert_allclose(parameters.total, parameters.noise + parameters.state + parameters.parameter,
                                   rtol=1e-10, atol=1e-12)
        checks["parameter_three_term_sum_verified"] = True
    if (run / "parameter_grid.csv").exists():
        grid = pd.read_csv(run / "parameter_grid.csv")
        np.testing.assert_allclose(grid.groupby(["case", "seed"]).weight.sum(), 1, rtol=1e-10)
        if (grid.weight < 0).any():
            raise ValueError("Parameter posterior weights must be nonnegative")
        checks["parameter_posterior_weights_verified"] = True
    if (run / "nested_components.csv").exists():
        nested = pd.read_csv(run / "nested_components.csv")
        np.testing.assert_allclose(nested.total_corrected,
                                   nested.epistemic_corrected + nested.aleatoric,
                                   rtol=1e-10, atol=1e-12)
        checks["recursive_corrected_sum_verified"] = True
        checks["recursive_negative_state_estimates"] = int((nested.epistemic_corrected < 0).sum())
        np.testing.assert_allclose(nested.total_population, nested.e_population + nested.a_population,
                                   rtol=1e-10, atol=1e-12)
        checks["recursive_population_identity_verified"] = True
    if (run / "state_plot_data.csv").exists():
        states = pd.read_csv(run / "state_plot_data.csv")
        np.testing.assert_allclose(states.x_mean, states.m_mean + states.w_mean, rtol=1e-10, atol=1e-12)
        residual = np.abs(states.observation - (states.x_mean + states.v_mean))
        roundoff = 16*np.finfo(float).eps*(states.observation.abs()+states.x_mean.abs()+states.v_mean.abs()) + 1e-12
        if (residual > roundoff).any():
            raise ValueError("Joint state mean identity exceeds component-scale roundoff budget")
        np.testing.assert_allclose(states.x_truth, states.m_truth + states.w_truth, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(states.observation, states.x_truth + states.v_truth, rtol=1e-10, atol=1e-12)
        if (states[["x_sd", "m_sd", "w_sd", "v_sd"]] < 0).any().any():
            raise ValueError("Posterior state standard deviations must be nonnegative")
        checks["joint_state_mean_identities_with_roundoff_budget_verified"] = True
        checks["joint_state_observation_identity_max_abs_error"] = float(residual.max())
        checks["joint_state_strict_observation_identity_verified"] = bool(np.allclose(
            states.observation, states.x_mean + states.v_mean, rtol=1e-10, atol=1e-12))
    (run / "report_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    return checks


def write_report(run, config, scores, summary, entries, paired, fits, checks, stability):
    overall = summary["summary_overall"]
    hybrid = overall[(overall.stage == "hybrid") & overall.method.isin(MAIN_HYBRID)]
    coherent = overall[(overall.stage == "coherent") & (overall.method == "particle64")]
    primary = coherent.iloc[0]
    control = pd.read_csv(run / "exact_control.csv")
    exact_error = float(np.abs(control.epistemic_mc - control.epistemic_exact).max())
    matched_particles = paired[paired.comparison == "particle256 versus particle64"]
    particle_mu_error = float(matched_particles.mean_difference.abs().mean())
    positive_ratio = matched_particles.epistemic_ratio.dropna()
    particle_ratio = float(positive_ratio.mean()) if len(positive_ratio) else float("nan")
    main_count = int(scores.loc[scores.method == "particle64", "n_members"].iloc[0])
    sensitivity_count = int(scores.loc[scores.method == "particle256", "n_members"].iloc[0])
    history_count = int(scores.loc[scores.method == "joint_known_noise", "n_members"].iloc[0])
    bounded = config.get("transition_kind") == "bounded_tanh"
    precision_columns = [column for column in stability if "resolution_fraction" in column]
    unstable = (stability[precision_columns] > 0).any().any()
    primary_stability = stability[stability.particles == main_count]
    model_definition = (f"$g(z)={config['transition_bound']:g} "
                        f"\\tanh(g_{{\\mathrm{{median}}}}(z)/{config['transition_bound']:g})$" if bounded
                        else "$g$ is the frozen Chronos median forecast")
    n_seed = scores[scores.stage == "coherent"].seed.nunique()
    n_coherent = int(primary.n_targets)
    methods = scores.groupby(["stage", "method"]).agg(
        data_seeds=("seed", "nunique"), forecast_rows=("mu", "size"),
        history_or_particle_members=("n_members", "first")).reset_index()
    all_scores = pd.read_csv(run / "scores.csv")
    cohort = all_scores[all_scores.method.isin(["joint_known_noise", "particle64"])].copy()
    cohort["predicted_time"] = cohort.origin + cohort.lead - 1
    split_counts = cohort.groupby(["stage", "split"]).agg(
        forecast_rows=("mu", "size"), cases=("case", "nunique"), data_seeds=("seed", "nunique"),
        first_predicted_week=("predicted_time", "min"), last_predicted_week=("predicted_time", "max")).reset_index()
    origins = cohort.drop_duplicates(["stage", "split", "case", "seed", "origin"]).groupby(
        ["stage", "split"]).size().rename("origin_series_pairs").reset_index()
    split_counts = split_counts.merge(origins, on=["stage", "split"])
    split_counts.to_csv(run / "split_counts.csv", index=False)
    particle_series = pd.read_csv(run / "coherent_series.csv")
    filter_diagnostics = particle_series.groupby(["seed", "particles"]).agg(
        mean_ess=("ess", "mean"), minimum_ess=("ess", "min"),
        mean_update_ancestors=("unique_ancestors", "mean"),
        minimum_oldest_history_ancestors=("oldest_ancestor_diversity", "min")).reset_index()
    filter_diagnostics.to_csv(run / "particle_diagnostic_summary.csv", index=False)
    metrics = ["method", "n_targets", "n_predictable", "observation_rmse", "observation_crps", "observation_nll",
               "observation_coverage90", "predictable_coverage90", "aleatoric", "epistemic", "total"]
    metrics += [column for column in ("mixture_nll", "mixture_crps")
                if column in coherent.columns]
    by_case = summary["summary_by_case"]
    hybrid_case = by_case[(by_case.stage == "hybrid") & (by_case.method == "joint_known_noise")]
    fits_summary = fits.groupby("case")[["q_true", "r_true", "q_fit", "r_fit"]].mean().reset_index()
    fit_sd = fits.groupby("case")[["q_fit", "r_fit"]].std(ddof=1).rename(
        columns={"q_fit": "q_fit_seed_sd", "r_fit": "r_fit_seed_sd"}).reset_index()
    fits_summary = fits_summary.merge(fit_sd, on="case")
    pair_summary = paired.groupby(["comparison", "case", "seed"]).agg(
        mean_abs_difference=("mean_difference", lambda values: values.abs().mean()),
        state_variance_ratio=("epistemic_ratio", "mean"),
        crps_difference=("crps_difference", "mean"), n_pairs=("lead", "size")).reset_index()
    pair_summary = pair_summary.groupby(["comparison", "case"]).agg(
        mean_abs_difference=("mean_abs_difference", "mean"),
        state_variance_ratio=("state_variance_ratio", "mean"),
        crps_difference=("crps_difference", "mean"), n_pairs=("n_pairs", "sum")).reset_index()
    pair_summary.to_csv(run / "paired_diagnostic_summary.csv", index=False)
    fits_summary.to_csv(run / "noise_fit_summary.csv", index=False)
    lines = [
        "# Chronos latent-posterior uncertainty experiment", "", f"Saved run: `{run.name}`.", "",
        "## What this experiment establishes", "",
        "The law of total variance gives an identified decomposition after specifying a latent "
        "dynamic model, observation noise, and a posterior over its latent history. This experiment "
        "tests the computation and numerical behavior. It does not extract a unique aleatoric/epistemic "
        "split from raw Chronos quantiles, and frozen Chronos weights have no inferred posterior.", "",
        *( ["**The raw recursive Chronos control failed stability.** "
            f"The largest true latent magnitude is {primary_stability.max_abs_latent.max():.4g} a.u. "
            f"from an amplitude-one initial history, while the largest primary forecast magnitude is "
            f"{primary_stability.max_abs_forecast.max():.4g} a.u. and the largest transition-mean error is "
            f"{primary_stability.max_abs_transition_mean_error.max():.4g} a.u. The per-seed table below "
            "shows which episode failed. Collapsed finite-particle clouds and precision-limited "
            "forecasts prevent interpreting aggregate coverage as valid calibration. The following "
            "full-horizon scores report this failure, rather than validate a practical decomposition.", ""] if unstable else []),
        *( ["**The raw filter also has an invalid-weight normalization failure.** Saved ESS "
            "falls below one, outside the mathematically valid range for normalized weights. "
            "Extreme absolute log likelihoods lost the normalization offset in floating-point "
            "arithmetic. This is a numerical implementation defect in addition to the raw "
            "generator's signal growth; the affected filter outputs are not posterior calibration evidence.", ""]
           if not checks["particle_ess_within_normalized_weight_bounds"] else []),
        *( ["**This is an exploratory bounded-model follow-up.** " + model_definition +
            " replaces the raw median transition after observing its instability. The fixed bound "
            "is an explicit model choice, not a recovered component of original Chronos uncertainty. "
            "Truth and filtering both use this new transition. Raw and bounded episodes have different "
            "generating dynamics, so their forecast accuracy scores are not a paired method ranking. "
            "Hybrid AR-to-Chronos outputs are reused unchanged and still use the raw direct median map.", "",
            f"[Original unbounded failure report]({config['original_unbounded_run']}/report.md).", ""] if bounded else []),
        f"The self-consistent Chronos control uses {n_seed} independent data seeds and "
        f"{n_coherent} one-step test forecasts, with {main_count} particles per seed. Its 90% Gaussian "
        f"moment intervals cover observations {100*primary.observation_coverage90:.1f}% of the time "
        f"and predictable transition means {100*primary.predictable_coverage90:.1f}% of the time "
        "on positive-state-variance rows. " +
        ("In this unstable run these descriptive rates pool stable and failed episodes and cannot "
         "establish calibration." if unstable else "These are conditional empirical diagnostics, not a guarantee of calibration."), "",
        f"On the paired seed-0 particle-count sensitivity, moving from {main_count} to "
        f"{sensitivity_count} particles changes the forecast mean by {particle_mu_error:.4g} a.u. "
        f"on average (absolute difference); mean state-variance ratio is {particle_ratio:.4g}. "
        "This comparison at two particle counts leaves exact nonlinear posterior convergence unresolved.", "",
        "## The two experiments answer different questions", "",
        "**Coherent control.** Truth and inference share the deterministic transition: " + model_definition +
        ". The equations are $X_{t+1}=g(Z_t)+W_t$, "
        "$Y_{t+1}=X_{t+1}+V_{t+1}$. Known independent Gaussian variances are "
        "$Q=0.01$ and $R=0.04$. A known deterministic 32-step, period-16 sinusoidal "
        "initial context is followed by stochastic observations. A fully adapted particle filter "
        "retains joint ancestry for each 32-value latent context. Sequential filtering carries "
        "information from all earlier episode observations; it does not reset the posterior "
        "at each prediction. Before each update, the forecast components are", "",
        r"$$E_{t+1}=\operatorname{Var}[g(Z_t)\mid D_t],\qquad "
        r"A_{t+1}=Q+R,\qquad T_{t+1}=E_{t+1}+A_{t+1}.$$", "",
        "The identity is exact under this model; the saved state variance is a finite-particle "
        "estimate. Here 'epistemic' means uncertainty about the present latent history, not pretrained "
        r"weights. The predictable target is the actual transition mean $g(Z_t^{\mathrm{true}})$, "
        "before process and measurement noise. The first forecast has a deterministic input "
        "history, so its zero state variance is correct and is excluded from Gaussian state scores.", "",
        "**Hybrid sensitivity control.** An exact AR filter/smoother supplies the posterior "
        "over the most recent 64 observed weeks, and joint draws are mapped through frozen Chronos. "
        "The stationary AR prior (or proper static N(0,1) prior) is reinitialized at each window "
        "start. This window-conditioned posterior excludes observations before those 64 weeks; "
        "it is not Canari's posterior conditioned on the full available history. The AR oracle "
        "uses exactly the same window and prior. "
        f"Each main origin uses {history_count} joint histories. At lead $h$, declared future noise "
        r"is the AR quantity $A_h=R+Q\sum_{j=0}^{h-1}a^{2j}$, while state variance is the "
        "variance across direct Chronos lead-$h$ forecasts. This defines an origin-specific "
        "conditional forecast mixture. It is not a self-consistent multi-step nonlinear Chronos "
        "transition model. The analytical AR target is a control for sensitivity and dynamics "
        "mismatch; Chronos and AR need not have the same propagated state variance.", "",
        "A posterior-mean input removes all input-state variance. Joint histories preserve "
        "temporal covariance; independent marginal histories deliberately destroy it. Raw Chronos "
        "is retained as a total-forecast baseline with no declared component split.", "",
        "## Data, leakage prevention, and aggregation", "",
        markdown_table(methods), "", markdown_table(split_counts), "",
        f"Each synthetic series contributes {config['train_end']} training observations. "
        "The split-count table uses one primary method per stage, so particle/history "
        "replicates do not multiply dataset counts. Hybrid horizons share an origin and "
        "are not independent data seeds. Primary one-step SMC episodes are evaluated separately "
        "from the historical AR train/validation/test series.", "",
        "Synthetic latent truth is used only for held-out scoring and illustrations. "
        "Training-prefix noise fits use observed data only with the AR coefficient and seasonal "
        "mean fixed. Validation rows are saved separately; the tables here use `split=test`. "
        "Every posterior uses observations at or before its forecast origin. Past-prefix "
        "smoothing is valid conditioning on already available observations. The run config "
        "below records exact boundaries and seed choices.", "",
        "Forecast errors average over origins within seed/lead, then equally over leads, "
        "data seeds, and cases. RMSE is the square root of the resulting mean squared error. "
        "History draws and particles are numerical members, not additional independent datasets. "
        "Sparse independent-history, extra-draw and extra-particle runs are compared only with "
        "their matched cohorts. Gaussian NLL/CRPS score moment summaries; the underlying "
        "mixtures can be non-Gaussian. `mixture_nll` and `mixture_crps`, when present, score the "
        "explicit finite Gaussian mixture with the same declared conditional noise for each "
        "history/particle member; they do not score a native Chronos quantile density. "
        "`n_predictable` gives the number of eligible predictable-target rows. "
        "Per-data-seed standard deviations are saved in "
        "`data_seed_dispersion.csv`; they describe repeated-series variation, not an iid standard "
        "error over forecast rows. Coverage columns are fractions.", "",
        "The known deterministic initial context has exactly zero state variance. For reporting "
        "only, those initial coherent rows are set to E=0 and excluded from state NLL/CRPS/coverage, "
        "including when floating-point variance of identical values produced a tiny positive raw "
        "number. This follows the known initial-condition control; no data-dependent variance "
        "floor is applied. `scores.csv` is preserved, and `scores_for_report.csv` records the "
        "derived test cohort. Every later state variance is unchanged.", "",
        "## Primary coherent Chronos results", "", markdown_table(coherent[metrics]), "",
        "Known future noise fixes the aleatoric component at 0.05 a.u.². Good total prediction "
        "scores alone do not validate the state component; predictable-target coverage and "
        "particle-count sensitivity examine that component separately. With a small number "
        "of independent data seeds, " +
        ("the raw control exposes model/filter failure rather than establish calibration." if unstable else
         "these controls test numerical feasibility rather than certify universal calibration."),
        "", markdown_table(filter_diagnostics), "",
        *( ["**Late filter weights are numerically invalid in this raw run.** With normalized "
            "weights, ESS must lie between 1 and the particle count. The saved ESS falls below "
            "one. The original implementation subtracted log-sum-exp from extreme absolute "
            "Gaussian log weights; floating-point arithmetic lost the normalization offset. "
            "This defect is separate from the raw generator's growing signal. Late filter "
            "states, uncertainty estimates, and their scores therefore cannot be interpreted "
            "as a valid particle-posterior approximation. The early state illustration remains "
            "an explicitly limited prefix, and the original full outputs remain preserved.", ""]
           if not checks["particle_ess_within_normalized_weight_bounds"] else []),
        "ESS describes the predictive ancestor weights at one update; the ancestor columns "
        "count retained history ancestry after resampling. Repeated updates can lose older "
        "history diversity even when current-update ESS is high. This can affect a "
        "history-dependent Chronos map and understate state uncertainty. Initial context "
        "values are known, so initial ancestry IDs do not represent distinct uncertain "
        "latent values. Diagnostics and the particle-count comparison should be assessed together.", "",
        "## Numerical stability and precision", "", markdown_table(stability), "",
        "Resolution fractions compare floating-point spacing with the nominal noise standard "
        "deviation. `process_resolution_fraction` uses float64 spacing at the true transition mean "
        "before adding process noise; `measurement_resolution_fraction` uses float64 spacing at "
        "the latent signal before adding measurement noise. `forecast_resolution_fraction` uses "
        "float64 spacing at the finite-particle forecast mean, and "
        "`chronos_input_resolution_fraction` uses float32 spacing at the true latent input, since "
        "the pinned Chronos pipeline consumes float32 contexts. These columns distinguish "
        "precision of simulated truth from precision of filter forecasts and model inputs. "
        "Fractions are over steps within a seed/count; they are not calibration scores.", "",
        "## Hybrid results on AR-generated truth", "", markdown_table(hybrid[metrics]), "",
        markdown_table(hybrid_case[["case", "observation_rmse", "observation_crps", "observation_coverage90",
                                   "predictable_coverage90", "epistemic", "oracle_epistemic",
                                   "aleatoric", "oracle_aleatoric"]]), "",
        "The known-noise and fitted-noise rows have the same interpretation only conditional "
        "on their declared noise parameters. Differences from the AR oracle combine the "
        "Chronos map's sensitivity with forecast-function mismatch. They should not be described "
        "as identification of the original Chronos quantile variance.", "",
        "## Analytical sanity check and numerical sensitivities", "",
        f"Maximum absolute Monte Carlo state-variance discrepancy on the saved exact linear "
        f"control is {exact_error:.4g} a.u.². Exact future-noise components and all raw control "
        "rows are saved in `exact_control.csv`.", "", markdown_table(pair_summary), "",
        "A state-variance ratio of one means agreement with the paired reference. A positive "
        "CRPS difference means a worse score for the alternative on precisely the same cohort. "
        "A change caused by independent histories is a covariance ablation, not extra model "
        "uncertainty. More history draws assess Monte Carlo sensitivity; more particles assess "
        "both filtering approximation and forecast integration.", "",
        "## Fitting noise from training observations", "", markdown_table(fits_summary), "",
        "Noise fits use the AR innovation likelihood with a fixed known coefficient. The "
        "reported standard deviations describe repeated data-seed variation, not a posterior "
        "over parameters. A fixed pair of fitted Q/R values omits parameter uncertainty. "
        "Separating Q from R depends on temporal structure: at a=0, only Q+R is identified "
        "from these observations. A successful optimizer alone does not establish identification.", "",
    ]
    if (run / "parameter_scores.csv").exists():
        parameters = pd.read_csv(run / "parameter_scores.csv")
        values = parameters.groupby(["case", "lead"])[["noise", "state", "parameter", "total"]].mean().reset_index()
        grid = pd.read_csv(run / "parameter_grid.csv")
        nodes = int(grid.groupby("case").q.nunique().max())
        lines += ["## Additional AR parameter-posterior diagnostic", "", markdown_table(values), "",
                  "The saved training-only Q/R grid supplies a posterior under the recorded prior "
                  f"and grid bounds. The finite prior assigns equal mass to {nodes} fixed log-spaced "
                  f"Q nodes from 0.0001 to 1 and {nodes} R nodes from 0.005 to 1 a.u.²; static Q is fixed at zero. "
                  "Bounds and node weights do not depend on observed data. The forecast origin is the first validation origin, immediately "
                  "after training. The same training prefix supplies the parameter posterior and "
                  "the conditional state posterior at every grid node, so the decomposition uses "
                  "one consistent joint posterior. Validation observations remain unseen. "
                  "The decomposition retains three terms: expected future noise, "
                  "expected conditional state variance, and variation of conditional forecast "
                  "means across parameters. They sum to the saved total. This is an AR experiment "
                  "with known transition coefficient, not parameter inference for the nonlinear "
                  "Chronos particle model. Grid/prior sensitivity and identifiability remain material.", ""]
        posterior_summary = fits.groupby("case").agg(
            q_posterior_mean_across_seeds=("q_mean", "mean"),
            r_posterior_mean_across_seeds=("r_mean", "mean"),
            largest_edge_mass=("edge_mass", "max")).reset_index()
        lines += [markdown_table(posterior_summary), "",
                  "Posterior means above average equally over independent data seeds; they do not "
                  "form a pooled posterior. The largest edge mass flags sensitivity to grid truncation. "
                  "Per-seed posterior quantiles and all node weights are retained in `noise_fits.csv` "
                  "and `parameter_grid.csv`; quantile bounds are not averaged into a purported interval.", ""]
    if (run / "nested_components.csv").exists():
        nested = pd.read_csv(run / "nested_components.csv")
        columns = ["origin", "lead", "inner_draws", "aleatoric", "epistemic_raw",
                   "correction", "epistemic_corrected", "total_corrected", "feasible_corrected"]
        lines += ["## Recursive nonlinear multi-step diagnostic", "", markdown_table(nested[columns]), "",
                  r"For the actual recursive Chronos transition, $E_h=\operatorname{Var}_Z "
                  r"\mathbb{E}_W[X_{t+h}\mid Z]$ and $A_h=\mathbb{E}_Z "
                  r"\operatorname{Var}_W[X_{t+h}\mid Z]+R$. The conditional expectations "
                  "and variances require nested simulation. With K future trajectories per current "
                  "history, sample-mean variation includes process-noise Monte Carlo variance; the "
                  "saved correction subtracts expected within-history process variance divided by K. "
                  "Corrected negative state variances remain visible rather than being silently "
                  "clipped. The uncorrected empirical finite-mixture identity and its reconstruction "
                  "error are retained separately in the raw CSV. The corrected nested estimator uses "
                  "outer sample variance (ddof=1); the direct one-step and empirical finite-particle "
                  "identity use population variance (ddof=0). Their normalizations differ by "
                  "M/(M-1) for M outer histories, in addition to process-simulation error. The diagnostic uses only saved "
                  "seed-0 origins and paired inner-draw counts, so it is a numerical feasibility "
                  "check rather than a held-out calibration assessment. This nested simulation "
                  "is distinct from the direct multi-horizon hybrid map in Stage B.", ""]
    if (run / "state_plot_data.csv").exists():
        lines += ["## A plot_states-style posterior view", "",
                  "Figure16 displays signal X, transition mean M, realized process shock W, and "
                  "realized measurement noise V from the joint filtered posterior in the seed-0 "
                  "coherent episode. The top forecast is computed before observing Y; the state "
                  "panels are computed after assimilating Y. The joint relations are X=M+W and "
                  "Y=X+V. In particular, conditioning on an observed Y makes X and V negatively "
                  "correlated, so the displayed posterior standard deviations cannot be added. "
                  "Posterior uncertainty about realized W/V differs from the future-noise variance "
                  "Q/R used in the forecast decomposition. Raw posterior means, standard deviations, "
                  "truth, and prior forecasts are retained in `state_plot_data.csv`.", ""]
    lines += ["## Figure guide", "",
              "Read Figures 1–4 as the mechanism, Figure 5 as the analytical check, and Figure 6 "
              "as the hybrid diagnostic. Figures 7–10 show the primary coherent model; Figures "
              "11–12 expose fitted-noise and Monte Carlo assumptions. Every figure is saved as "
              "PNG for viewing and PDF/PGF for publication. All time-series uncertainty bands "
              "show plus/minus one standard deviation.", ""]
    figure_guide = ["# Figure guide", "", f"Run `{run.name}`. Read the report for definitions and scoring cohorts.", ""]
    for number, entry in enumerate(entries, start=1):
        prefix = f"../../figures/{run.name}/{entry['name']}"
        block = [f"### Figure {number}", "", f"![Figure {number}]({prefix}.png)", "", entry["caption"], "",
                 f"[PDF]({prefix}.pdf) · [PGF]({prefix}.pgf)", ""]
        lines += block
        figure_guide += block
    lines += ["## What remains unresolved", "",
              "This synthetic control uses a correct generator, known noise, and known initial "
              "history. Real Canari signals may have unknown noise, dynamics mismatch, "
              "nonstationarity, and uncertain initialization. The coherent experiment tests one "
              "step calibration only; the recursive multi-step diagnostic has too few origins "
              "and inner trajectories for a calibration claim. Higher particle counts and "
              "independent seed replications are required before a deployment claim. An inferred "
              "noise posterior would add a third parameter term. Frozen weights still provide no "
              "weight-posterior uncertainty. Nothing here changes Canari `src/`.", "",
              "## Saved verification and reproducibility", "", "```json", json.dumps(checks, indent=2), "```", "",
              "The numerical unit tests and runner checks are recorded separately in "
              "`verification.json`. Report checks verify saved component sums and known coherent "
              "noise; they do not assert that arbitrary Chronos forecasts match the AR oracle.", "",
              "```json", json.dumps(config, indent=2), "```", ""]
    (run / "report.md").write_text("\n".join(lines))
    (run / "figure_guide.md").write_text("\n".join(figure_guide))
    (run / "figure_manifest.json").write_text(json.dumps(entries, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run / "COMPLETED").exists():
        raise ValueError("Inference must finish before report generation")
    config = json.loads((run / "config.json").read_text())
    all_scores = pd.read_csv(run / "scores.csv")
    scores = all_scores[all_scores.split == "test"].copy()
    initial = (scores.stage == "coherent") & (scores.origin == config["coherent_context"])
    np.testing.assert_allclose(scores.loc[initial, "epistemic"], 0, rtol=0, atol=1e-20)
    scores.loc[initial, "epistemic"] = 0.
    scores.loc[initial, ["predictable_coverage90", "predictable_crps", "predictable_nll",
                         "predictable_width90"]] = np.nan
    scores.loc[initial, "state_scored"] = False
    scores.to_csv(run / "scores_for_report.csv", index=False)
    METRICS.extend(column for column in ("mixture_nll", "mixture_crps") if column in scores.columns)
    for method in ("particle64", "particle256"):
        count = int(scores.loc[scores.method == method, "n_members"].iloc[0])
        LABELS[method] = f"{count} particles"
    checks = verify_saved_components(run, scores)
    checks["zero_initial_variance_control_verified"] = True
    checks["initial_coherent_state_rows_excluded"] = int(initial.sum())
    (run / "report_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    summary = aggregate_scores(scores)
    for name, frame in summary.items():
        frame.to_csv(run / f"{name}.csv", index=False)
    paired = paired_diagnostics(scores)
    paired.to_csv(run / "paired_diagnostics.csv", index=False)
    calibration_ = calibration(scores)
    calibration_.to_csv(run / "interval_calibration_by_seed.csv", index=False)
    folder = EXP_DIR / "figures" / run.name
    folder.mkdir(parents=True, exist_ok=True)
    coherent = pd.read_csv(run / "coherent_series.csv")
    fits = pd.read_csv(run / "noise_fits.csv")
    stability = stability_summary(coherent, config)
    stability.to_csv(run / "run_stability.csv", index=False)
    entries = make_figures(run, scores, summary, folder, coherent, paired, calibration_, config, stability)
    write_report(run, config, scores, summary, entries, paired, fits, checks, stability)
    print(json.dumps({"report": str(run / "report.md"), "figures": len(entries), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
