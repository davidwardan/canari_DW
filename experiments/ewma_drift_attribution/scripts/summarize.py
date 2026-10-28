"""Tables, pre-registered criteria and figures from one saved run of the toy.

    PYTHONPATH=/Users/davidwardan/canari_DW/src python \
        experiments/ewma_drift_attribution/scripts/summarize.py --tag RUN_TAG
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import run_experiment as rx  # noqa: E402
from metrics import PERIOD, adaptation_kept  # noqa: E402

SINGLE_COL = (3.5, 2.5)
PREAMBLE = r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}"
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex", "font.family": "serif", "text.usetex": True,
    "pgf.rcfonts": False, "pgf.preamble": PREAMBLE, "text.latex.preamble": PREAMBLE,
    "lines.linewidth": 1, "figure.figsize": SINGLE_COL, "font.size": 9, "savefig.dpi": 300,
})

ARM_ORDER = list(rx.ARMS)
SWEEP = [f"rsr_k{rx.kappa_label(kappa)}" for kappa in rx.KAPPAS]
INF_X = 16.0  # where kappa = infinity sits on the log axis
ARM_COLORS = {"none": "tab:gray", "frozen": "tab:purple", "rsr_kinf": "tab:orange",
              "rsr_k1": "tab:blue", "rsr_k0.25": "tab:green"}
SWEEP_COLOR = "tab:blue"
COLUMNS = {
    "A": ["lstm_drift", "pattern_rmse", "rmse", "ll", "crps", "coverage_95",
          "reading_ratio_mean", "reading_ratio_slope"],
    "B": ["share_lstm", "share_level", "share_noise", "share_sum", "share_trend",
          "pattern_rmse", "rmse", "ll", "crps", "coverage_95"],
    "C": ["share_lstm", "share_level", "share_noise", "share_sum", "share_trend",
          "pattern_rmse", "rmse", "ll", "crps", "coverage_95"],
    "D": ["share_lstm", "share_level", "share_noise", "share_trend", "pattern_rmse",
          "crps_after_change", "eta", "rmse", "crps", "coverage_95"],
}
SCENARIO_TITLES = {
    "A": "A, null (no slope)", "B": "B, slope", "C": "C, slope, unknown trend start",
    "D": "D, slope + phase shift at week 416",
}


def kappa_x(arm: str) -> float:
    label = arm.removeprefix("rsr_k")
    return INF_X if label == "inf" else float(label)


def add_eta(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-seed adaptation kept, from the paired D - B excess CRPS after the change."""

    crps = runs.pivot_table(index=["arm", "seed"], columns="scenario",
                            values="crps_after_change")
    if not {"B", "D"} <= set(crps.columns):
        return runs
    excess = (crps["D"] - crps["B"]).unstack("arm")
    eta = pd.concat([
        pd.DataFrame({"scenario": "D", "arm": arm, "seed": excess.index,
                      "eta": adaptation_kept(excess[arm], excess["frozen"],
                                             excess["rsr_kinf"]).to_numpy()})
        for arm in excess.columns
    ])
    return runs.merge(eta, on=["scenario", "seed", "arm"], how="left")


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    numeric = runs.select_dtypes("number").columns.drop("seed")
    grouped = runs.groupby(["scenario", "arm"])[list(numeric)]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std"))
    summary["seeds"] = runs.groupby(["scenario", "arm"])["seed"].nunique()
    return summary


def fmt(metric: str, mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "–"
    if metric == "lstm_drift":
        return f"{mean:.2e} ± {std:.1e}"
    digits = 3 if metric.startswith(("share", "reading", "eta", "coverage")) else 4
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def scenario_table(summary: pd.DataFrame, scenario: str) -> str:
    columns = COLUMNS[scenario]
    rows = ["| arm | " + " | ".join(f"`{c}`" for c in columns) + " |",
            "| :--- |" + " ---: |" * len(columns)]
    for arm in ARM_ORDER:
        if (scenario, arm) not in summary.index:
            continue
        record = summary.loc[(scenario, arm)]
        cells = [fmt(c, record.get(f"{c}_mean", np.nan), record.get(f"{c}_std", np.nan))
                 for c in columns]
        rows.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def mean_of(summary, scenario, arm, metric) -> float:
    try:
        return float(summary.loc[(scenario, arm), f"{metric}_mean"])
    except KeyError:
        return np.nan


def criteria(summary: pd.DataFrame) -> tuple:
    """Pre-registered criteria 1-3 per arm, and criterion 4 per scenario."""

    def drift_kept_out(scenario, arm):
        reference = mean_of(summary, scenario, "none", "share_lstm")
        if not reference >= 0.05:
            return "n/a"
        lstm = mean_of(summary, scenario, arm, "share_lstm")
        trend, trend_none = (mean_of(summary, scenario, name, "share_trend")
                             for name in (arm, "none"))
        return "yes" if lstm <= reference / 4 and abs(1 - trend) < abs(1 - trend_none) else "no"

    def low_cost(arm):
        crps, crps_free = (mean_of(summary, "A", name, "crps") for name in (arm, "rsr_kinf"))
        cov, cov_free = (mean_of(summary, "A", name, "coverage_95") for name in (arm, "rsr_kinf"))
        return "yes" if crps <= 1.05 * crps_free and abs(cov - cov_free) <= 0.02 else "no"

    def pattern_learned(arm):
        excess = {name: mean_of(summary, "D", name, "crps_after_change")
                  - mean_of(summary, "B", name, "crps_after_change")
                  for name in ("frozen", "rsr_kinf")}
        if not excess["frozen"] > excess["rsr_kinf"]:
            return "n/a"
        return "yes" if mean_of(summary, "D", arm, "eta") >= 0.8 else "no"

    rows = []
    for arm in SWEEP + ["rsr_mean_k1", "rsr_slope_k1"]:
        checks = {"1 (B)": drift_kept_out("B", arm), "1 (C)": drift_kept_out("C", arm),
                  "2 (A)": low_cost(arm), "3 (D)": pattern_learned(arm)}
        checks["all"] = "yes" if all(value == "yes" for value in checks.values()) else "no"
        rows.append({"arm": arm, **checks})
    monotone = {}
    for scenario in ("B", "C"):
        values = [mean_of(summary, scenario, arm, "share_lstm") for arm in SWEEP]
        monotone[scenario] = float(spearmanr([kappa_x(arm) for arm in SWEEP], values)[0])
    return pd.DataFrame(rows), monotone


def save(fig, figures_dir: Path, name: str) -> None:
    for suffix in ("pgf", "pdf"):
        fig.savefig(figures_dir / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def sweep_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ticks = [kappa_x(arm) for arm in SWEEP]
    ax.set_xticks(ticks)
    ax.set_xticklabels([r"$\infty$" if x == INF_X else f"${x:g}$" for x in ticks])
    ax.minorticks_off()


def sweep_line(ax, summary, scenario, metric, **style) -> None:
    arms = [arm for arm in SWEEP[::-1] if (scenario, arm) in summary.index]
    x = [kappa_x(arm) for arm in arms]
    mean = [mean_of(summary, scenario, arm, metric) for arm in arms]
    std = [float(summary.loc[(scenario, arm), f"{metric}_std"]) for arm in arms]
    ax.errorbar(x, mean, yerr=std, color=SWEEP_COLOR, capsize=2, **style)


def plot_shares(summary, figures_dir, tag) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.5, 3.4))
    for ax, metric, label in zip(axes, ("share_lstm", "share_trend"),
                                 (r"LSTM share $a_{\mathtt{LSTM}}$", r"trend share $a_{\mathtt{T}}$")):
        for scenario, linestyle in (("B", "-"), ("C", "--")):
            sweep_line(ax, summary, scenario, metric, linestyle=linestyle,
                       label=f"{scenario}, constrained RSR")
            ax.axhline(mean_of(summary, scenario, "none", metric), color=ARM_COLORS["none"],
                       linestyle=linestyle, label=f"{scenario}, no constraint")
        ax.set_ylabel(label)
        sweep_axis(ax)
    axes[-1].set_xlabel(r"online tolerance ratio $\kappa = \tau/\lambda$")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.align_ylabels(axes)
    save(fig, figures_dir, f"{tag}_share_vs_kappa")


def plot_costs(summary, figures_dir, tag) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.5, 3.2))
    ratio = summary.xs("A")["crps_mean"] / mean_of(summary, "A", "rsr_kinf", "crps")
    arms = [arm for arm in SWEEP[::-1] if arm in ratio.index]
    axes[0].plot([kappa_x(arm) for arm in arms], ratio[arms], color=SWEEP_COLOR, marker="o",
                 markersize=3)
    axes[0].set_ylabel(r"CRPS ratio, A")
    if ("D", "rsr_kinf") in summary.index and "eta_mean" in summary:
        sweep_line(axes[1], summary, "D", "eta", marker="o", markersize=3)
    axes[1].set_ylabel(r"$\eta$, D")
    axes[1].set_xlabel(r"online tolerance ratio $\kappa = \tau/\lambda$")
    for ax in axes:
        sweep_axis(ax)
    fig.align_ylabels(axes)
    save(fig, figures_dir, f"{tag}_cost_vs_kappa")


def plot_lstm_cycles(runs, results_dir, figures_dir, tag) -> None:
    """Cycle means of the LSTM output in B: zero for a pure seasonal output."""

    fig, ax = plt.subplots()
    for arm in ("none", "rsr_kinf", "rsr_k1", "rsr_k0.25"):
        names = runs[(runs.scenario == "B") & (runs.arm == arm)]["run"]
        if names.empty:
            continue
        cycles = []
        for name in names:
            lstm = pd.read_csv(results_dir / name / "states.csv")["lstm_mean"].to_numpy()
            count = len(lstm) // PERIOD
            cycles.append(lstm[: count * PERIOD].reshape(count, PERIOD).mean(axis=1))
        cycles = np.asarray(cycles)
        years = (rx.TEST_START + PERIOD * np.arange(cycles.shape[1]) + PERIOD / 2) / PERIOD
        mean, std = cycles.mean(axis=0), cycles.std(axis=0, ddof=1) if len(cycles) > 1 else 0
        label = {"none": "no constraint", "rsr_kinf": r"$\kappa=\infty$ online",
                 "rsr_k1": r"$\kappa=1$", "rsr_k0.25": r"$\kappa=1/4$"}[arm]
        ax.fill_between(years, mean - std, mean + std, color=ARM_COLORS[arm], alpha=0.3,
                        linewidth=0)
        ax.plot(years, mean, color=ARM_COLORS[arm], label=label)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("LSTM cycle mean (data units)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    save(fig, figures_dir, f"{tag}_lstm_cycle_means_B")


def training_table(batch_dir: Path) -> str:
    training = pd.read_csv(batch_dir / "training.csv")
    rows = ["| snapshot | family | selected $\\sigma_v$ | epoch | initial trend / $\\delta$ |",
            "| :--- | :--- | ---: | ---: | ---: |"]
    for (kind, family), group in training.groupby(["snapshot", "family"]):
        share = []
        for _, record in group.iterrows():
            y_std = rx.make_processor(rx.toy_series(family, int(record.seed))).scale_const_std[0]
            share.append(record.trend_init_normalized * y_std / rx.SLOPE)
        share = np.asarray(share)
        rows.append(f"| `{kind}` | {family} | {group.selected_sigma_v.mean():.3f} ± "
                    f"{group.selected_sigma_v.std():.3f} | {group.optimal_epoch.mean():.1f} | "
                    + ("–" if family == "A" else f"{share.mean():.3f} ± {share.std(ddof=1):.3f}")
                    + " |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    batch_dir = rx.EXP_DIR / "results" / args.tag
    results_dir = rx.EXP_DIR / "results"
    figures_dir = rx.EXP_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    runs = pd.read_csv(batch_dir / "metrics.csv")
    failures = runs[runs.status != "complete"]
    runs = add_eta(runs[runs.status == "complete"])
    summary = summarize(runs)
    summary.to_csv(batch_dir / "summary_mean_std.csv")
    checks, monotone = criteria(summary)
    checks.to_csv(batch_dir / "criteria.csv", index=False)

    lines = [f"# Results, run `{args.tag}`", "",
             "Mean ± std over seeds "
             f"{sorted(runs.seed.unique().tolist())}; test split, data units. "
             "Definitions: `planning/metrics.md`.", "",
             f"Numerical failures: {len(failures)}"
             + ("" if failures.empty else " — " + ", ".join(failures.run)), "",
             f"Largest |level + LSTM + noise − y| over all runs: "
             f"{runs.sum_identity_error.max():.1e}", "",
             "## Offline snapshots", "", training_table(batch_dir), ""]
    for scenario in COLUMNS:
        if scenario in summary.index.get_level_values(0):
            lines += [f"## {SCENARIO_TITLES[scenario]}", "", scenario_table(summary, scenario), ""]
    lines += ["## Pre-registered criteria (seed means)", "",
              "| " + " | ".join(checks.columns) + " |",
              "|" + " :--- |" * len(checks.columns),
              *("| " + " | ".join(map(str, row)) + " |" for row in checks.itertuples(index=False)),
              "",
              "Criterion 4, Spearman correlation between κ and seed-mean LSTM share "
              "(≥ 0.8 required): "
              + ", ".join(f"{s}: {value:.2f}" for s, value in monotone.items()), ""]
    (batch_dir / "summary.md").write_text("\n".join(lines))

    plot_shares(summary, figures_dir, args.tag)
    plot_costs(summary, figures_dir, args.tag)
    plot_lstm_cycles(runs, results_dir, figures_dir, args.tag)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
