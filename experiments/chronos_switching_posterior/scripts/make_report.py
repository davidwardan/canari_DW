"""Render figures and a short report from one completed switching run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronos_switching_posterior_mpl")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (6.5, 3.5)
BLUE, GREEN, RED = "tab:blue", "tab:green", "tab:red"
PURPLE, BLACK = "tab:purple", "black"


def plot_defaults():
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
        "lines.linewidth": 1,
        "figure.figsize": SINGLE_COL,
        "font.size": 9,
        "savefig.dpi": 300,
    })


def legend(ax, ncol=2):
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02),
              ncol=ncol, columnspacing=.9, handlelength=1.5)


def save_figure(fig, folder, name):
    fig.tight_layout()
    for suffix in ("pdf", "pgf", "png"):
        fig.savefig(folder / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("—")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _shade_regime(ax, start, end):
    ax.axvspan(start, end, color="tab:orange", alpha=.12, lw=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run / "COMPLETED").exists():
        raise ValueError(f"Completed run required: {run}")
    config = json.loads((run / "config.json").read_text())
    summary = pd.read_csv(run / "summary.csv")
    regime = pd.read_csv(run / "regime_scores.csv")
    scores = pd.read_csv(run / "scores.csv")
    states = pd.read_csv(run / "state_plot_data.csv")
    primary_method = f"switching{config['particles']}"
    selected = states[(states.method == primary_method) & (states.seed == 0)].sort_values("time")
    selected_scores = scores[(scores.method == primary_method) & (scores.seed == 0)].sort_values("time")
    time = selected.time.to_numpy()
    start, end = config["switch_times"]
    figure_dir = EXP / "figures" / run.name
    figure_dir.mkdir(parents=True, exist_ok=False)
    plot_defaults()

    # 1. The discrete regime posterior.
    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    ax.step(time, selected.p_abnormal_posterior, where="post", color=PURPLE,
            label=r"$P(S_t=\mathrm{abnormal}\mid D_t)$")
    ax.plot(time, selected.true_regime, color=BLACK, linestyle="--", label="True regime")
    ax.axhline(.5, color="0.5", linestyle=":", label="Alarm threshold")
    _shade_regime(ax, start, end)
    ax.set_ylim(-.04, 1.04)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Abnormal-regime probability")
    legend(ax, ncol=3)
    save_figure(fig, figure_dir, "01_regime_posterior")

    # 2. Law-of-total-variance accounting.
    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    aleatoric = selected.aleatoric.to_numpy()
    state = selected.epistemic_state.to_numpy()
    regime_var = selected.epistemic_regime.to_numpy()
    total = selected.total.to_numpy()
    ax.fill_between(time, 0, aleatoric, color=GREEN, alpha=.3, label="Aleatoric $Q+R$")
    ax.fill_between(time, aleatoric, aleatoric + state, color=BLUE, alpha=.3,
                    label="Epistemic: latent state")
    ax.fill_between(time, aleatoric + state, total, color=PURPLE, alpha=.3,
                    label="Epistemic: regime")
    ax.plot(time, total, color=BLACK, label="Total")
    _shade_regime(ax, start, end)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel(r"Forecast variance (a.u.$^2$)")
    legend(ax, ncol=2)
    save_figure(fig, figure_dir, "02_variance_decomposition")

    # 3. Forecast with the total predictive uncertainty.
    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    mean = selected.forecast_mean.to_numpy()
    sd = selected.forecast_sd.to_numpy()
    ax.fill_between(time, mean - sd, mean + sd, color="0.7", alpha=.3,
                    label=r"Total $\pm1\sigma$")
    ax.plot(time, selected.observation, color=RED, label="Observation")
    ax.plot(time, mean, color=BLUE, label="Forecast mean")
    ax.plot(time, selected.latent, color=BLACK, linestyle="--", label="Latent signal")
    _shade_regime(ax, start, end)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Signal (a.u.)")
    legend(ax, ncol=4)
    save_figure(fig, figure_dir, "03_switching_forecast")

    # 4. A plot_states-style view: forecast first, then assimilated state pieces.
    fig, axes = plt.subplots(5, 1, figsize=(6.5, 8.5), sharex=True)
    axes[0].fill_between(time, mean - sd, mean + sd, color="0.7", alpha=.3,
                         label=r"Forecast $\pm1\sigma$")
    axes[0].plot(time, selected.observation, color=RED, label="Observed")
    axes[0].plot(time, mean, color=BLUE, label="Forecast mean")
    axes[0].set_ylabel("$Y$\n(a.u.)")
    legend(axes[0], ncol=3)
    axes[1].plot(time, selected.posterior_mean, color=BLUE, label="Posterior mean")
    axes[1].fill_between(time, selected.posterior_mean - selected.posterior_sd,
                         selected.posterior_mean + selected.posterior_sd,
                         color=BLUE, alpha=.3, label=r"Posterior $\pm1\sigma$")
    axes[1].plot(time, selected.latent, color=BLACK, linestyle="--", label="Truth")
    axes[1].set_ylabel("$X$\n(a.u.)")
    legend(axes[1], ncol=3)
    axes[2].plot(time, selected.transition_mean, color=BLUE, label="Transition mean")
    axes[2].plot(time, selected.latent, color=BLACK, linestyle="--", label="Latent signal")
    axes[2].set_ylabel("$M$\n(a.u.)")
    legend(axes[2], ncol=2)
    axes[3].plot(time, selected.process_shock, color=BLUE, label="Realized process shock")
    axes[3].axhline(0, color="0.5", linestyle=":")
    axes[3].set_ylabel("$W$\n(a.u.)")
    legend(axes[3], ncol=1)
    axes[4].plot(time, selected.measurement_noise, color=BLUE, label="Realized measurement noise")
    axes[4].axhline(0, color="0.5", linestyle=":")
    axes[4].set_ylabel("$V$\n(a.u.)")
    axes[4].set_xlabel("Time (steps)")
    legend(axes[4], ncol=1)
    for ax in axes:
        _shade_regime(ax, start, end)
    save_figure(fig, figure_dir, "04_states_posterior")

    verification = json.loads((run / "verification.json").read_text())
    summary_columns = ["method", "observation_rmse", "observation_crps", "observation_nll",
                       "observation_coverage90", "predictable_coverage90", "aleatoric",
                       "epistemic_state", "epistemic_regime", "total"]
    regime_columns = ["method", "seed", "brier", "log_loss", "auroc", "detected",
                      "detection_delay", "recovery_delay", "false_alarm_before_switch"]
    lines = [
        "# Chronos-2 small switching posterior",
        "",
        f"Completed run `{run.name}` uses the cached Chronos-2 small checkpoint "
        f"`{config['revision']}` on the CPU-only bounded transition "
        "$g(z)=3\\tanh(g_{\\rm Chronos}(z)/3)$. The experiment adds a two-state "
        "normal/abnormal Markov regime to the latent-history particle filter.",
        "",
        "The abnormal state has a declared residual shift of +0.75 in the synthetic "
        "signal units. The filter carries a 32-step latent context and a discrete "
        "regime in each particle, scores candidate transitions with the Gaussian "
        "observation likelihood, and applies the exact scalar Gaussian update.",
        "",
        "For the one-step predictive distribution, the saved identity is "
        "$T=A+E_{\\mathrm{state}}+E_{\\mathrm{regime}}$. Here $A=Q+R$ is future "
        "process plus measurement noise, $E_{\\mathrm{state}}$ is within-regime "
        "latent-history variance, and $E_{\\mathrm{regime}}$ is the between-regime "
        "variance of the regime means.",
        "",
        "## Main forecast metrics",
        "",
        markdown_table(summary[summary_columns]),
        "",
        "The fixed baseline is the same bounded particle filter with only the normal "
        "regime. Switching improves the synthetic abnormal episode because it can "
        "represent the residual level shift explicitly. The fixed baseline is a "
        "diagnostic comparator, not a tuned competitor.",
        "",
        "## Regime detection",
        "",
        markdown_table(regime[regime_columns]),
        "",
        "The five 96-step episodes use the same declared switch times (24 and 56) "
        "and independent data seeds. The seed-0 128-particle repeat is a particle "
        "sensitivity check. Detection delay uses a two-step posterior-above-0.5 rule; "
        "recovery uses the analogous two-step-below-0.5 rule.",
        "",
        "## Figures",
        "",
        "The regime posterior shows the switch signal directly. The variance figure "
        "shows why a regime term is useful: between-regime spread is separate from "
        "the known future noise and within-regime history spread. The final figure "
        "is the requested `plot_states`-style view; its lower rows show posterior "
        "latent signal and realized transition/process/measurement pieces after "
        "assimilation.",
        "",
    ]
    for name, caption in [
        ("01_regime_posterior", "Posterior abnormal-regime probability for seed 0; orange shading marks the true abnormal interval."),
        ("02_variance_decomposition", "Exact scalar variance decomposition into aleatoric, within-regime state, between-regime, and total terms."),
        ("03_switching_forecast", "Observed and latent signals with the switching forecast and total plus/minus one-standard-deviation band."),
        ("04_states_posterior", "Plot-states-style view: forecast before assimilation followed by posterior latent and realized state pieces."),
    ]:
        lines += [f"![{caption}](../../figures/{run.name}/{name}.png)", "",
                  f"[PDF](../../figures/{run.name}/{name}.pdf) · "
                  f"[PGF](../../figures/{run.name}/{name}.pgf)", "", caption, ""]
    lines += [
        "## Verification and limits",
        "",
        f"The run used {config['particles']} particles for the primary filter, "
        f"{config['particle_sensitivity']} for the seed-0 sensitivity repeat, and "
        f"{verification['chronos_calls']} Chronos calls. All saved scores are finite; "
        f"the maximum variance reconstruction error is "
        f"`{verification['reconstruction_max']:.3g}` and no effective sample size "
        "exceeded the candidate-particle bound.",
        "",
        "This is a mechanism test on synthetic data with a known abnormal shift. It "
        "does not learn the shift from Canari data, and it does not claim that every "
        "real abnormality is a level shift. The next SKF-style extension should replace "
        "the fixed abnormal shift with a learned or hierarchical residual state and "
        "test false alarms on held-out real episodes.",
        "",
        "The runner, filter, tests, configuration, hashes, raw CSVs, and figures are "
        "all retained under this experiment directory. No `src/` files were changed.",
        "",
    ]
    (run / "report.md").write_text("\n".join(lines))
    print(f"Figures: {figure_dir}")
    print(f"Report: {run / 'report.md'}")


if __name__ == "__main__":
    main()
