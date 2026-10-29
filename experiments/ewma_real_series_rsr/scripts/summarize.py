"""Tables and figures from one saved run of the real-series EWMA experiment.

    PYTHONPATH=/Users/davidwardan/canari_DW/src python \
        experiments/ewma_real_series_rsr/scripts/summarize.py --tag RUN_TAG
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_experiment as rx  # noqa: E402

SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (6.5, 3.5)
PREAMBLE = r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}"
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex", "font.family": "serif", "text.usetex": True,
    "pgf.rcfonts": False, "pgf.preamble": PREAMBLE, "text.latex.preamble": PREAMBLE,
    "lines.linewidth": 1, "figure.figsize": SINGLE_COL, "font.size": 9, "savefig.dpi": 300,
})

KAPPA_LABELS = [f"{kappa:g}" for kappa in rx.KAPPAS]
ARM_ORDER = ["frozen", *(f"frozen_ewma_k{k}" for k in KAPPA_LABELS),
             "rsr", *(f"rsr_ewma_k{k}" for k in KAPPA_LABELS)]
COLORS = {"frozen": "tab:purple", "frozen_ewma": "tab:green", "rsr": "tab:orange",
          "rsr_ewma": "tab:blue"}
NAMES = {"frozen": "frozen", "frozen_ewma": "frozen + EWMA", "rsr": "RSR",
         "rsr_ewma": "RSR + EWMA"}
MAIN = ["level_rmse", "trend_rmse"]
SECONDARY = ["rmse", "ll", "crps", "coverage_95"]
YLABELS = {"level_rmse": "level RMSE", "trend_rmse": "trend RMSE", "crps": "CRPS"}


def fmt(metric: str, mean: float, std: float) -> str:
    if metric == "trend_rmse":
        return f"{mean:.2e} ± {std:.1e}"
    return f"{mean:.3f} ± {std:.3f}"


def table(frame: pd.DataFrame, columns, spread: str) -> str:
    """Rows: arms; cells: mean ± std of `frame` grouped by arm."""

    mean = frame.groupby("arm")[columns].mean()
    std = frame.groupby("arm")[columns].std()
    rows = ["| arm | " + " | ".join(f"`{c}`" for c in columns) + " |",
            "| :--- |" + " ---: |" * len(columns)]
    for arm in ARM_ORDER:
        if arm in mean.index:
            rows.append(f"| `{arm}` | " + " | ".join(
                fmt(c, mean.loc[arm, c], std.loc[arm, c]) for c in columns) + " |")
    return f"Mean ± std over {spread}.\n\n" + "\n".join(rows)


def save(fig, figures_dir: Path, name: str) -> None:
    for suffix in ("pgf", "pdf"):
        fig.savefig(figures_dir / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def kappa_axis(ax, sparse: bool = False) -> None:
    """Label every kappa, or every other one in narrow panels."""

    ax.set_xscale("log", base=2)
    ax.set_xticks(rx.KAPPAS)
    labels = [{0.5: "$1/2$", 0.25: "$1/4$", 0.125: "$1/8$"}.get(kappa, f"${kappa:g}$")
              for kappa in rx.KAPPAS]
    if sparse:
        labels = [label if index % 2 else "" for index, label in enumerate(labels)]
    ax.set_xticklabels(labels)
    ax.minorticks_off()


def draw(ax, mean: pd.Series, std: pd.Series, sparse: bool = False) -> None:
    """EWMA sweeps as lines against kappa, unconstrained arms as dashed levels."""

    for family in ("frozen", "rsr"):
        if family in mean.index:
            ax.axhline(mean[family], color=COLORS[family], linestyle="--")
        arms = [f"{family}_ewma_k{k}" for k in KAPPA_LABELS]
        arms = [arm for arm in arms if arm in mean.index]
        ax.errorbar([float(arm.rsplit("_k", 1)[1]) for arm in arms], mean[arms],
                    yerr=std[arms], color=COLORS[f"{family}_ewma"], marker="o",
                    markersize=2.5, capsize=2)
    kappa_axis(ax, sparse)


def legend(fig, ncol: int) -> None:
    handles = [plt.Line2D([], [], color=COLORS[key], linestyle="--" if "_" not in key else "-",
                          marker=None if "_" not in key else "o", markersize=2.5)
               for key in ("frozen", "frozen_ewma", "rsr", "rsr_ewma")]
    fig.legend(handles, [NAMES[key] for key in ("frozen", "frozen_ewma", "rsr", "rsr_ewma")],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False)


def plot_overall(seed_means: pd.DataFrame, figures_dir: Path, tag: str) -> None:
    metrics = ["level_rmse", "trend_rmse", "crps"]
    fig, axes = plt.subplots(len(metrics), 1, sharex=True, figsize=(3.5, 3.8))
    grouped = seed_means.groupby("arm")
    for ax, metric in zip(axes, metrics):
        draw(ax, grouped[metric].mean(), grouped[metric].std())
        ax.set_ylabel(YLABELS[metric])
    axes[-1].set_xlabel(r"tolerance ratio $\kappa = \tau/\lambda$")
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.4)
    legend(fig, ncol=2)
    save(fig, figures_dir, f"{tag}_rmse_vs_kappa_all")


def plot_per_series(runs: pd.DataFrame, figures_dir: Path, tag: str) -> None:
    series = list(dict.fromkeys(runs["series"]))
    fig, axes = plt.subplots(2, len(series), sharex=True, squeeze=False,
                             figsize=(6.5, 3.4))
    for column, name in enumerate(series):
        grouped = runs[runs["series"] == name].groupby("arm")
        for row, metric in enumerate(MAIN):
            draw(axes[row, column], grouped[metric].mean(), grouped[metric].std(), sparse=True)
        axes[0, column].set_title(name, fontsize=9)
        axes[-1, column].set_xlabel(r"$\kappa$")
    for row, metric in enumerate(MAIN):
        axes[row, 0].set_ylabel(YLABELS[metric])
    for ax in axes.flat:
        ax.tick_params(labelsize=7)
        ax.yaxis.get_offset_text().set_fontsize(7)
    fig.tight_layout(w_pad=0.4)
    legend(fig, ncol=4)
    save(fig, figures_dir, f"{tag}_rmse_vs_kappa_per_series")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    batch_dir = rx.EXP_DIR / "results" / args.tag
    figures_dir = rx.EXP_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    runs = pd.read_csv(batch_dir / "metrics.csv")
    failures = runs[runs["status"] != "complete"]
    runs = runs[runs["status"] == "complete"]
    seed_means = runs.groupby(["series", "arm"], as_index=False)[MAIN + SECONDARY].mean()
    seed_means.to_csv(batch_dir / "seed_means.csv", index=False)
    training = pd.read_csv(batch_dir / "training.csv")

    lines = [f"# Results, run `{args.tag}`", "",
             f"Series {', '.join(dict.fromkeys(runs['series']))}; seeds "
             f"{sorted(runs['seed'].unique().tolist())}; test split, data units. "
             "Definitions: `planning/metrics.md`.", "",
             f"Numerical failures: {len(failures)}"
             + ("" if failures.empty else " — " + ", ".join(failures["run"])), "",
             "Training: selected epoch "
             f"{training['optimal_epoch'].mean():.1f} on average (range "
             f"{training['optimal_epoch'].min()}–{training['optimal_epoch'].max()}).", "",
             "## All series", "",
             table(seed_means, MAIN + SECONDARY, "series (each series averaged over seeds)"),
             ""]
    for name in dict.fromkeys(runs["series"]):
        subset = runs[runs["series"] == name]
        lines += [f"## {name}", "", table(subset, MAIN, "seeds"), "",
                  table(subset, SECONDARY, "seeds"), ""]
    (batch_dir / "summary.md").write_text("\n".join(lines))

    plot_overall(seed_means, figures_dir, args.tag)
    plot_per_series(runs, figures_dir, args.tag)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
