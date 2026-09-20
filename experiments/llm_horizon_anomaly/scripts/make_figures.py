"""Build the figures and tables of the anomaly-detection H sweep.

    python scripts/make_figures.py [run_directory]

Defaults to the most recent `results/run_*` directory.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

POD_COLOR = "tab:blue"
TTD_COLOR = "tab:green"
FA_COLOR = "tab:red"
REF_COLOR = "0.4"


def latest_run():
    runs = sorted((common.EXP_DIR / "results").glob("run_*"))
    if not runs:
        raise SystemExit("no results/run_* directory; run scripts/run_experiment.py")
    return runs[-1]


def horizon_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(common.HORIZONS)
    ax.set_xticklabels([str(h) for h in common.HORIZONS])
    ax.minorticks_off()
    ax.set_xlabel(r"Re-forecast interval $H$ (weeks)")


def legend_above(ax, ncol=2):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False)


def wilson_interval(successes, trials, z=1.96):
    """Binomial confidence interval that stays inside [0, 1] near the edges."""

    if trials == 0:
        return np.nan, np.nan
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return max(centre - half, 0.0), min(centre + half, 1.0)


def figure_pod(detections, false_alarms, figures_dir):
    """Detection probability against H, against what false alarms alone would give."""

    grouped = detections.groupby("horizon")["detected"]
    pod = grouped.mean()
    bounds = [wilson_interval(int(s), int(n)) for s, n in zip(grouped.sum(), grouped.size())]
    low, high = zip(*bounds)

    window_years = common.MAX_WEEKS_TO_DETECT / common.WEEKS_PER_YEAR
    chance = (
        false_alarms.assign(
            chance=1 - np.exp(-false_alarms["false_alarms_per_year"] * window_years)
        )
        .groupby("horizon")["chance"]
        .mean()
    )

    fig, ax = plt.subplots()
    ax.fill_between(pod.index, low, high, color=POD_COLOR, alpha=0.3, lw=0)
    ax.plot(pod.index, pod, color=POD_COLOR, label=r"Detected (95\% CI)")
    ax.plot(chance.index, chance, color=REF_COLOR, ls="--",
            label="Expected from false alarms alone")
    horizon_axis(ax)
    ax.set_ylabel("Probability of detection")
    ax.set_ylim(0, 1.05)
    legend_above(ax, ncol=1)
    common.base.save_figure(fig, "pod_vs_horizon", figures_dir)
    plt.close(fig)


def figure_excess_pod(detections, false_alarms, figures_dir):
    """The detector's own contribution: detections above its false-alarm process."""

    window_years = common.MAX_WEEKS_TO_DETECT / common.WEEKS_PER_YEAR
    chance = (
        false_alarms.assign(
            chance=1 - np.exp(-false_alarms["false_alarms_per_year"] * window_years)
        )
        .groupby(["series", "horizon"])["chance"]
        .mean()
    )
    pod = detections.groupby(["series", "horizon"])["detected"].mean()
    excess = (pod - chance).unstack()

    median = excess.median()
    low, high = excess.quantile(0.25), excess.quantile(0.75)

    fig, ax = plt.subplots()
    ax.fill_between(median.index, low, high, color=POD_COLOR, alpha=0.3, lw=0)
    ax.plot(median.index, median, color=POD_COLOR, label="Median over series (IQR)")
    ax.axhline(0.0, color=REF_COLOR, ls="--", label="No better than false alarms")
    horizon_axis(ax)
    ax.set_ylabel("Detection probability above chance")
    legend_above(ax, ncol=1)
    common.base.save_figure(fig, "excess_pod_vs_horizon", figures_dir)
    plt.close(fig)


def figure_time_to_detection(detections, figures_dir):
    """Time to detection among detected anomalies: median and interquartile range."""

    detected = detections[detections["detected"]]
    grouped = detected.groupby("horizon")["weeks_to_detect"]
    median, low, high = grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)

    fig, ax = plt.subplots()
    ax.fill_between(median.index, low, high, color=TTD_COLOR, alpha=0.3, lw=0)
    ax.plot(median.index, median, color=TTD_COLOR, label="Median (IQR), detected only")
    ax.axhline(
        common.MAX_WEEKS_TO_DETECT,
        color=REF_COLOR,
        ls="--",
        label="Detection deadline (3 years)",
    )
    horizon_axis(ax)
    ax.set_ylabel("Time to detection (weeks)")
    ax.set_ylim(bottom=0)
    legend_above(ax)
    common.base.save_figure(fig, "time_to_detection_vs_horizon", figures_dir)
    plt.close(fig)


def figure_false_alarms(false_alarms, figures_dir):
    """False alarms measured on the same series with no anomaly injected."""

    grouped = false_alarms.groupby("horizon")["false_alarms_per_year"]
    mean = grouped.mean()
    low, high = grouped.quantile(0.25), grouped.quantile(0.75)

    fig, ax = plt.subplots()
    ax.fill_between(mean.index, low, high, color=FA_COLOR, alpha=0.3, lw=0)
    ax.plot(mean.index, mean, color=FA_COLOR, label="Clean series (mean, IQR)")
    ax.axhline(
        common.MAX_FALSE_ALARMS_PER_YEAR,
        color=REF_COLOR,
        ls="--",
        label="Budget set during tuning",
    )
    horizon_axis(ax)
    ax.set_ylabel(r"False alarms (1/year)")
    ax.set_ylim(bottom=0)
    legend_above(ax)
    common.base.save_figure(fig, "false_alarms_vs_horizon", figures_dir)
    plt.close(fig)


def write_tables(run_dir, detections, false_alarms):
    summary = common.summarize(detections, false_alarms, by="horizon")
    summary.round(3).to_csv(run_dir / "summary.csv")

    by_series = detections.pivot_table(
        index="series", columns="horizon", values="detected", aggfunc="mean"
    )
    by_series.round(2).to_csv(run_dir / "pod_by_series.csv")

    columns = "l" + "r" * len(common.HORIZONS)
    header = " & ".join([r"$H$"] + [str(h) for h in common.HORIZONS])
    lines = [
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for label, key, fmt in (
        ("Probability of detection", "pod", "{:.2f}"),
        ("Expected from false alarms", "chance_pod", "{:.2f}"),
        ("Excess over chance", "excess_pod", "{:.2f}"),
        ("Median time to detection [weeks]", "median_weeks_to_detect", "{:.0f}"),
        ("False alarms [1/year]", "false_alarms_per_year", "{:.3f}"),
    ):
        values = " & ".join(
            "--" if np.isnan(summary.loc[h, key]) else fmt.format(summary.loc[h, key])
            for h in common.HORIZONS
        )
        lines.append(f"{label} & {values} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (run_dir / "summary_table.tex").write_text("\n".join(lines) + "\n")
    return summary


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    detections = pd.read_csv(run_dir / "detections.csv")
    false_alarms = pd.read_csv(run_dir / "false_alarms.csv")
    figures_dir = common.EXP_DIR / "figures"

    common.base.use_paper_style()
    figure_pod(detections, false_alarms, figures_dir)
    figure_excess_pod(detections, false_alarms, figures_dir)
    figure_time_to_detection(detections, figures_dir)
    figure_false_alarms(false_alarms, figures_dir)
    summary = write_tables(run_dir, detections, false_alarms)

    tuned = json.loads((run_dir / "config.json").read_text())["tuned"]
    print(f"run: {run_dir.name}")
    print(f"anomaly magnitude: {tuned['slope_per_year']:.2f} sigma/year")
    print(f"SKF parameters (tuned at H={tuned['tuned_at_horizon']}): {tuned['params']}\n")
    print(summary.round(3).to_string())
    print(f"\nfigures -> {figures_dir}")


if __name__ == "__main__":
    main()
