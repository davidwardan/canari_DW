"""Figures and tables of the model comparison, from saved per-run results.

    python scripts/make_figures.py [run_directory]

Defaults to the most recent directory under `results/`.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import EXP_DIR, MODELS, common

METRICS = [("rmse", "RMSE"), ("log_lik", "Log-likelihood")]


def latest_run():
    runs = sorted((EXP_DIR / "results").glob("run_*"))
    if not runs:
        raise SystemExit("no results/run_* directory; run scripts/run_experiment.py")
    return runs[-1]


def horizon_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(common.HORIZONS)
    ax.set_xticklabels([str(h) for h in common.HORIZONS])
    ax.minorticks_off()
    ax.set_xlabel(r"Re-forecast interval $H$ (weeks)")


def figure_metric(median, key, ylabel, name):
    """One metric vs H, one curve per model, median over the 10 series."""

    fig, ax = plt.subplots()
    for model_id, label, color, linestyle in MODELS:
        if model_id in median.index:
            values = median.loc[model_id, key]
            ax.plot(values.index, values, color=color, ls=linestyle, label=label)
    horizon_axis(ax)
    ax.set_ylabel(ylabel)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    common.save_figure(fig, name, EXP_DIR / "figures")
    plt.close(fig)


def write_tables(run_dir, median):
    median.round(3).to_csv(run_dir / "summary.csv")

    labels = {model_id: label for model_id, label, *_ in MODELS}
    models = [model_id for model_id, *_ in MODELS if model_id in median.index]
    header = " & ".join(["Model"] + [str(h) for h in common.HORIZONS])
    lines = [r"\begin{tabular}{l" + "r" * len(common.HORIZONS) + "}", r"\toprule"]
    for key, title in METRICS:
        lines += [
            rf"\multicolumn{{{len(common.HORIZONS) + 1}}}{{l}}{{{title}}} \\",
            r"\midrule",
            header + r" \\",
            r"\midrule",
        ]
        for model_id in models:
            values = " & ".join(
                f"{median.loc[(model_id, h), key]:.3f}" for h in common.HORIZONS
            )
            lines.append(f"{labels[model_id]} & {values} " + r"\\")
        lines.append(r"\midrule" if key != METRICS[-1][0] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (run_dir / "summary_table.tex").write_text("\n".join(lines) + "\n")


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    metrics = pd.read_csv(run_dir / "metrics.csv")
    median = metrics.groupby(["model", "horizon"])[[k for k, _ in METRICS]].median()

    common.use_paper_style()
    figure_metric(median, "rmse", "RMSE (standardized units)", "rmse_vs_horizon")
    figure_metric(median, "log_lik", "Mean log-likelihood", "loglik_vs_horizon")
    write_tables(run_dir, median)

    print(f"run: {run_dir.name}\n")
    for key, title in METRICS:
        table = median[key].unstack("horizon")
        table = table.loc[[m for m, *_ in MODELS if m in table.index]]
        print(f"{title} (median over series)\n{table.round(3).to_string()}\n")
    print(f"figures -> {EXP_DIR / 'figures'}")


if __name__ == "__main__":
    main()
