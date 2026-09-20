"""Build the figures and tables of the H sweep from saved per-run results.

    python scripts/make_figures.py [run_directory]

Defaults to the most recent directory under `results/`.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

OBS_COLOR = "tab:red"
PRED_COLOR = "tab:blue"
REF_COLOR = "0.4"
EXAMPLE_SERIES = "ts67"
EXAMPLE_HORIZONS = (1, 52)
LEAD_HORIZON = 208  # the run whose blocks span every lead time of the sweep


def latest_run():
    runs = sorted((common.EXP_DIR / "results").glob("run_*"))
    if not runs:
        raise SystemExit("no results/run_* directory; run scripts/run_experiment.py")
    return runs[-1]


def load_raw(run_dir, series, horizon):
    return pd.read_csv(run_dir / "raw" / f"{series}_H{horizon}.csv", parse_dates=["date"])


def band(frame, value_col):
    """Median and interquartile range of `value_col` across series, per horizon."""

    grouped = frame.groupby("horizon")[value_col]
    return grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)


def horizon_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(common.HORIZONS)
    ax.set_xticklabels([str(h) for h in common.HORIZONS])
    ax.minorticks_off()
    ax.set_xlabel(r"Re-forecast interval $H$ (weeks)")


def legend_above(ax, ncol=2):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False)


def figure_skill(metrics):
    """RMSE as a fraction of the climatological RMSE, median and IQR over series."""

    median, low, high = band(metrics, "rmse_skill")
    fig, ax = plt.subplots()
    ax.fill_between(median.index, low, high, color=PRED_COLOR, alpha=0.3, lw=0)
    ax.plot(median.index, median, color=PRED_COLOR, label="Chronos-2 (median, IQR)")
    ax.axhline(1.0, color=REF_COLOR, ls="--", label="Climatology")
    horizon_axis(ax)
    ax.set_ylabel(r"RMSE / climatological RMSE")
    ax.set_ylim(bottom=0)
    legend_above(ax)
    common.save_figure(fig, "skill_vs_horizon")
    plt.close(fig)


def figure_degradation(metrics):
    """Loss relative to H=1: the direct cost of re-forecasting less often."""

    fig, ax = plt.subplots()
    median, low, high = band(metrics, "rmse_ratio")
    ax.fill_between(median.index, low, high, color=PRED_COLOR, alpha=0.3, lw=0)
    ax.plot(median.index, median, color=PRED_COLOR, label="RMSE (median, IQR)")
    crps, _, _ = band(metrics, "crps_ratio")
    ax.plot(crps.index, crps, color=PRED_COLOR, ls="--", label="CRPS (median)")
    ax.axhline(1.0, color=REF_COLOR, ls=":", lw=0.8)
    horizon_axis(ax)
    ax.set_ylabel(r"Score relative to $H=1$")
    legend_above(ax)
    common.save_figure(fig, "degradation_vs_horizon")
    plt.close(fig)


def lead_profile(run_dir, series_names):
    """Per-lead-time RMSE/climatology from the H=208 runs, median over series."""

    bins = np.unique(np.round(np.geomspace(1, LEAD_HORIZON, 24)).astype(int))
    profiles = []
    for series in series_names:
        raw = load_raw(run_dir, series, LEAD_HORIZON).dropna(subset=["y"])
        climatology = np.mean(raw["y"].to_numpy() ** 2)
        squared = (raw["mu"] - raw["y"]) ** 2
        binned = squared.groupby(pd.cut(raw["lead"], bins, include_lowest=True), observed=False).mean()
        profiles.append(np.sqrt(binned / climatology))
    profile = pd.concat(profiles, axis=1)
    lead = np.array([interval.mid for interval in profile.index])
    return lead, profile.median(axis=1).to_numpy()


def figure_lead(run_dir, metrics, series_names):
    """Where the aggregate penalty comes from: error growth inside one block."""

    lead, rmse = lead_profile(run_dir, series_names)
    aggregate, _, _ = band(metrics, "rmse_skill")

    fig, ax = plt.subplots()
    ax.plot(lead, rmse, color=PRED_COLOR, label=r"Per-lead, $H=208$")
    ax.plot(
        aggregate.index,
        aggregate,
        color="tab:green",
        ls="--",
        label=r"Block average at $H$",
    )
    ax.axhline(1.0, color=REF_COLOR, ls=":", lw=0.8)
    horizon_axis(ax)
    ax.set_xlabel(r"Steps ahead of the last Chronos-2 call (weeks)")
    ax.set_ylabel(r"RMSE / climatological RMSE")
    ax.set_ylim(bottom=0)
    legend_above(ax)
    common.save_figure(fig, "error_by_lead")
    plt.close(fig)


def figure_example(run_dir):
    """One series, short and long re-forecast interval, in standardized units."""

    fig, axes = plt.subplots(
        2, 1, figsize=common.DOUBLE_COL, sharex=True, sharey=True
    )
    for ax, horizon in zip(axes, EXAMPLE_HORIZONS):
        raw = load_raw(run_dir, EXAMPLE_SERIES, horizon)
        ax.plot(raw["date"], raw["y"], color=OBS_COLOR, lw=0.7, label="Observation")
        ax.fill_between(
            raw["date"],
            raw["mu"] - raw["std"],
            raw["mu"] + raw["std"],
            color=PRED_COLOR,
            alpha=0.3,
            lw=0,
            label=r"$\mu \pm \sigma$",
        )
        ax.plot(raw["date"], raw["mu"], color=PRED_COLOR, lw=0.7, label=r"$\mu$")
        ax.set_ylabel(rf"$H={horizon}$")
    axes[-1].set_xlabel("Date")
    legend_above(axes[0], ncol=3)
    common.save_figure(fig, f"example_{EXAMPLE_SERIES}")
    plt.close(fig)


def write_tables(run_dir, metrics):
    summary = (
        metrics.groupby("horizon")[
            ["rmse", "crps", "log_lik", "rmse_skill", "rmse_ratio", "crps_ratio"]
        ]
        .median()
        .round(3)
    )
    summary.to_csv(run_dir / "summary.csv")

    by_series = metrics.pivot(index="series", columns="horizon", values="rmse_skill")
    by_series.round(3).to_csv(run_dir / "rmse_skill_by_series.csv")

    columns = "l" + "r" * len(common.HORIZONS)
    header = " & ".join([r"$H$"] + [str(h) for h in common.HORIZONS])
    lines = [
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for label, key in (
        ("RMSE", "rmse"),
        ("CRPS", "crps"),
        ("Log-likelihood", "log_lik"),
        (r"RMSE / climatology", "rmse_skill"),
        (r"RMSE / RMSE($H{=}1$)", "rmse_ratio"),
    ):
        values = " & ".join(f"{summary.loc[h, key]:.3f}" for h in common.HORIZONS)
        lines.append(f"{label} & {values} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (run_dir / "summary_table.tex").write_text("\n".join(lines) + "\n")
    return summary


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    metrics = pd.read_csv(run_dir / "metrics.csv")
    metrics["rmse_skill"] = metrics["rmse"] / metrics["rmse_clim"]
    metrics["crps_skill"] = metrics["crps"] / metrics["crps_clim"]
    reference = metrics[metrics["horizon"] == 1].set_index("series")
    metrics["rmse_ratio"] = metrics["rmse"] / metrics["series"].map(reference["rmse"])
    metrics["crps_ratio"] = metrics["crps"] / metrics["series"].map(reference["crps"])

    series_names = sorted(metrics["series"].unique())
    common.use_paper_style()
    figure_skill(metrics)
    figure_degradation(metrics)
    figure_lead(run_dir, metrics, series_names)
    figure_example(run_dir)
    summary = write_tables(run_dir, metrics)

    print(f"run: {run_dir.name}\n")
    print(summary.to_string())
    print(f"\nfigures -> {common.EXP_DIR / 'figures'}")


if __name__ == "__main__":
    main()
