"""Figures and tables for the LLM anomaly-detection benchmark.

    python scripts/make_figures.py [run_directory]

Defaults to the most recent `results/run_*`.
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
        raise SystemExit("no results/run_*; run scripts/benchmark.py first")
    return runs[-1]


def load(run_dir):
    summary = json.loads((run_dir / "benchmark_summary.json").read_text())
    config_path = run_dir / "config_used.json"
    run_config = json.loads(config_path.read_text()) if config_path.exists() else common.load_config()
    rows = []
    for series_summary in summary["per_series"]:
        threshold = series_summary["model_parameters_used"]["threshold"]
        for row in series_summary["multi_realization_evaluation"]:
            rows.append(
                {
                    "series": series_summary["series"],
                    "llm_horizon": series_summary["llm_horizon"],
                    "threshold": threshold,
                    **row,
                }
            )
    return summary, pd.DataFrame(rows), run_config


def horizon_axis(ax, horizons):
    ax.set_xscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.minorticks_off()
    ax.set_xlabel(r"Re-forecast interval $H$ (weeks)")


def magnitude_axis(ax, magnitudes):
    ax.set_xscale("log")
    ax.set_xticks(magnitudes)
    ax.set_xticklabels([f"{m:g}" for m in magnitudes])
    ax.minorticks_off()
    ax.set_xlabel(r"Anomaly magnitude ($\sigma$/year)")


def legend_above(ax, ncol=1):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False)


def _by_horizon(frame, column):
    """Mean over series at each (H, magnitude)."""

    return frame.pivot_table(
        index="llm_horizon", columns="anomaly_magnitude", values=column, aggfunc="mean"
    )


def figure_pod_vs_horizon(frame, figures_dir):
    """The headline comparison: detection against H, one curve per magnitude."""

    table = _by_horizon(frame, "probability_of_detection")
    shades = plt.cm.viridis(np.linspace(0.15, 0.9, len(table.columns)))
    fig, ax = plt.subplots()
    for color, magnitude in zip(shades, table.columns):
        ax.plot(table.index, table[magnitude], color=color,
                label=rf"{magnitude:g}$\sigma$/yr")
    horizon_axis(ax, list(table.index))
    ax.set_ylabel("Probability of detection")
    ax.set_ylim(0, 1.05)
    legend_above(ax, ncol=4)
    common.base.save_figure(fig, "pod_vs_horizon", figures_dir)
    plt.close(fig)


def figure_false_alarms_vs_horizon(frame, figures_dir):
    """False alarms come from the clean series, so they depend on H but not magnitude."""

    per_horizon = frame.groupby(["llm_horizon", "series"])["false_alarm_rate_per_y"].first()
    grouped = per_horizon.groupby("llm_horizon")
    mean, low, high = grouped.mean(), grouped.quantile(0.25), grouped.quantile(0.75)

    fig, ax = plt.subplots()
    ax.fill_between(mean.index, low, high, color=FA_COLOR, alpha=0.3, lw=0)
    ax.plot(mean.index, mean, color=FA_COLOR, label="Clean series (mean, IQR)")
    horizon_axis(ax, list(mean.index))
    ax.set_ylabel(r"False alarms (1/year)")
    ax.set_ylim(bottom=0)
    legend_above(ax)
    common.base.save_figure(fig, "false_alarms_vs_horizon", figures_dir)
    plt.close(fig)


def figure_ttd_vs_horizon(frame, run_config, figures_dir):
    table = _by_horizon(frame, "time_to_detection_years_mean")
    shades = plt.cm.viridis(np.linspace(0.15, 0.9, len(table.columns)))
    fig, ax = plt.subplots()
    for color, magnitude in zip(shades, table.columns):
        ax.plot(table.index, table[magnitude], color=color,
                label=rf"{magnitude:g}$\sigma$/yr")
    ax.axhline(
        run_config["max_timestep_to_detect"] / common.WEEKS_PER_YEAR,
        color=REF_COLOR, ls="--", label="Deadline",
    )
    horizon_axis(ax, list(table.index))
    ax.set_ylabel("Time to detection (years)")
    ax.set_ylim(bottom=0)
    legend_above(ax, ncol=4)
    common.base.save_figure(fig, "ttd_vs_horizon", figures_dir)
    plt.close(fig)


def figure_pod_vs_magnitude(frame, figures_dir):
    """The same surface read the other way: detection against magnitude, per H."""

    table = _by_horizon(frame, "probability_of_detection").T
    shades = plt.cm.plasma(np.linspace(0.1, 0.85, len(table.columns)))
    fig, ax = plt.subplots()
    for color, horizon in zip(shades, table.columns):
        ax.plot(table.index, table[horizon], color=color, label=rf"$H$={horizon}")
    magnitude_axis(ax, list(table.index))
    ax.set_ylabel("Probability of detection")
    ax.set_ylim(0, 1.05)
    legend_above(ax, ncol=4)
    common.base.save_figure(fig, "pod_vs_magnitude", figures_dir)
    plt.close(fig)


def write_tables(run_dir, summary, frame):
    aggregates = pd.DataFrame(summary["aggregate_by_magnitude"])
    aggregates.round(4).to_csv(run_dir / "aggregate_by_horizon_magnitude.csv", index=False)
    frame.round(4).to_csv(run_dir / "per_series.csv", index=False)

    pod = _by_horizon(frame, "probability_of_detection")
    pod.round(3).to_csv(run_dir / "pod_horizon_by_magnitude.csv")

    magnitudes = list(pod.columns)
    columns = "l" + "r" * len(magnitudes)
    lines = [
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        " & ".join([r"$H$ \textbackslash{} mag"] + [f"{m:g}" for m in magnitudes]) + r" \\",
        r"\midrule",
    ]
    for horizon in pod.index:
        values = [
            "--" if not np.isfinite(pod.loc[horizon, m]) else f"{pod.loc[horizon, m]:.2f}"
            for m in magnitudes
        ]
        lines.append(f"{horizon} & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (run_dir / "pod_table.tex").write_text("\n".join(lines) + "\n")
    return aggregates, pod


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    summary, frame, run_config = load(run_dir)
    figures_dir = common.EXP_DIR / "figures"

    common.base.use_paper_style()
    figure_pod_vs_horizon(frame, figures_dir)
    figure_false_alarms_vs_horizon(frame, figures_dir)
    figure_ttd_vs_horizon(frame, run_config, figures_dir)
    figure_pod_vs_magnitude(frame, figures_dir)
    _, pod = write_tables(run_dir, summary, frame)

    print(f"run: {run_dir.name}  |  model={run_config.get('model_id')}")
    print(f"horizons: {sorted(frame['llm_horizon'].unique())}")
    print(f"series:   {sorted(frame['series'].unique())}\n")
    print("Probability of detection, H (rows) x magnitude (columns):")
    print(pod.round(2).to_string())
    print(f"\nfigures -> {figures_dir}")


if __name__ == "__main__":
    main()
