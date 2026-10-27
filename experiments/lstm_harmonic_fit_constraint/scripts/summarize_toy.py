"""Tables and figures for one `run_toy.py` batch, from its saved results.

Tables: mean ± sample std over seeds per arm and method. Figures: seed 1 (declared in
advance), test split, against the known true components.

Run from the repository root:

    python experiments/lstm_harmonic_fit_constraint/scripts/summarize_toy.py [--tag RUN_TAG]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
# LaTeX plotting defaults, figure saving and arm colors of the SSM-constraint experiment
sys.path.insert(0, str(EXP_DIR.parent / "lstm_local_trend_rsr_ssm" / "scripts"))

from summarize import ARM_COLORS, DOUBLE_COL, METHOD_LABELS, save  # noqa: E402

RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
EXAMPLE_SEED = 1
ARM_LABELS = {"unconstrained": "Unconstrained", "ewma": "EWMA", "harmonic": "Harmonic fit"}
COLORS = {**ARM_COLORS, "harmonic": "tab:purple"}
METRICS = {
    "rmse": ("RMSE", "{:.4f}"),
    "ll_mean": ("LL", "{:.3f}"),
    "crps": ("CRPS", "{:.4f}"),
    "coverage_95": ("95% cov.", "{:.3f}"),
    "trend_mean": ("Trend", "{:.5f}"),
    "trend_rmse_last": ("Trend dev. last 3 y", "{:.5f}"),
    "lstm_drift_slope": ("LSTM drift", "{:.2e}"),
    "level_rmse_last": ("Level err. last 3 y", "{:.3f}"),
    "seasonal_rmse_last": ("LSTM vs seasonal err. last 3 y", "{:.3f}"),
    "reading_level_ripple_last": ("Level reading ripple", "{:.2e}"),
    "reading_slope_ripple_last": ("Slope reading ripple", "{:.2e}"),
}


def table(runs: pd.DataFrame, seeds) -> str:
    stats = runs.groupby(["method", "arm"])[list(METRICS)].agg(["mean", "std"])
    lines = [
        f"Mean ± sample std over seeds {seeds}; test split, data units.",
        "",
        "| Method | Arm | " + " | ".join(label for label, _ in METRICS.values()) + " |",
        "| :--- | :--- |" + " ---: |" * len(METRICS),
    ]
    for method in METHOD_LABELS:
        for arm in ARM_LABELS:
            if (method, arm) not in stats.index:
                continue
            cells = []
            for metric, (_, spec) in METRICS.items():
                mean, std = stats.loc[(method, arm), (metric, "mean")], \
                    stats.loc[(method, arm), (metric, "std")]
                cells.append("—" if pd.isna(mean) else
                             f"{spec.format(mean)} ± {spec.format(std)}")
            lines.append(f"| {METHOD_LABELS[method]} | {ARM_LABELS[arm]} | "
                         + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def load(run_dir: Path, arm: str, method: str) -> pd.DataFrame:
    return pd.read_csv(run_dir / f"{arm}_{method}.csv", index_col=0, parse_dates=True)


def band(ax, time, mean, std, color, label=None, linewidth=0.8):
    ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.3, linewidth=0)
    ax.plot(time, mean, color=color, linewidth=linewidth, label=label)


def truth(ax, time, values, label="truth"):
    ax.plot(time, values, color="k", linestyle="--", linewidth=0.7, label=label)


def plot_tracking(run_dir: Path, arms, methods, path: Path) -> None:
    """Arms side by side: absorbed drift, trend, and the two constrained readings."""

    rows = ["drift", "trend", "reading_level", "reading_slope"]
    labels = ["52-wk mean of\n$\\mu_R - s$", "Trend\n[1/week]", "Level\nreading",
              "Slope reading\n[1/week]"]
    fig, axes = plt.subplots(len(rows), len(methods), figsize=(DOUBLE_COL[0], 6.0),
                             sharex=True, sharey="row", squeeze=False)
    for column, method in zip(axes.T, methods):
        column[0].set_title(METHOD_LABELS[method])
        for arm in arms:
            frame = load(run_dir, arm, method)
            time, color = frame.index, COLORS[arm]
            absorbed = (frame["lstm_mean"] - frame["true_seasonal"]).rolling(52, center=True).mean()
            column[0].plot(time, absorbed, color=color, label=ARM_LABELS[arm])
            band(column[1], time, frame["trend_mean"], frame["trend_std"], color)
            if arm != "unconstrained":
                for ax, name in zip(column[2:], rows[2:]):
                    band(ax, time, frame[f"{name}_mean"], frame[f"{name}_std"], color)
        for ax, values in zip(column[:2], (0.0, frame["true_trend"])):
            truth(ax, time, np.broadcast_to(values, len(time)))
        for ax in column[2:]:
            ax.axhline(0, color="k", linestyle="--", linewidth=0.7)
    for ax, label in zip(axes[:, 0], labels):
        ax.set_ylabel(label)
    fig.supxlabel("Date", fontsize=plt.rcParams["font.size"])
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), frameon=False)
    fig.tight_layout()
    save(fig, path)


def plot_states(run_dir: Path, arm: str, methods, path: Path) -> None:
    """One arm: prediction and every hidden state against its true component."""

    rows = [("y", "$y$"), ("level", "Level"), ("trend", "Trend\n[1/week]"),
            ("lstm", "LSTM")]
    if arm != "unconstrained":
        rows += [("reading_level", "Level\nreading"), ("reading_slope", "Slope reading\n[1/week]")]
    if arm == "harmonic":
        rows += [("amplitude", "Harmonic\namplitudes")]
    fig, axes = plt.subplots(len(rows), len(methods),
                             figsize=(DOUBLE_COL[0], 1.15 * len(rows) + 0.6),
                             sharex=True, sharey="row", squeeze=False)
    for column, method in zip(axes.T, methods):
        frame = load(run_dir, arm, method)
        time = frame.index
        column[0].set_title(METHOD_LABELS[method])
        for ax, (name, _) in zip(column, rows):
            if name == "y":
                ax.plot(time, frame["obs"], color="tab:red", linewidth=0.5, label="observation")
                band(ax, time, frame["mean"], frame["std"], "tab:blue", "prediction", 0.5)
            elif name == "amplitude":
                for k, true_amplitude in ((1, 1.0), (2, 0.5)):
                    ax.plot(time, frame[f"amplitude{k}"], color="tab:blue")
                    truth(ax, time, np.full(len(time), true_amplitude))
            else:
                band(ax, time, frame[f"{name}_mean"], frame[f"{name}_std"], "tab:blue",
                     linewidth=0.5)
                reference = {"level": frame.get("true_level"), "trend": frame.get("true_trend"),
                             "lstm": frame.get("true_seasonal")}.get(name)
                truth(ax, time, np.zeros(len(time)) if reference is None else reference)
    for ax, (_, label) in zip(axes[:, 0], rows):
        ax.set_ylabel(label)
    fig.supxlabel("Date", fontsize=plt.rcParams["font.size"])
    handles, names = axes[0, 0].get_legend_handles_labels()
    truth_handles, truth_names = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles + truth_handles[-1:], names + ["true component / target"],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="batch run tag; defaults to the latest batch")
    args = parser.parse_args()
    tag = args.tag or sorted(p.name for p in RESULTS_DIR.iterdir()
                             if (p / "jobs.json").exists())[-1]
    batch_dir = RESULTS_DIR / tag
    config = json.loads((batch_dir / "config.json").read_text())
    runs = pd.read_csv(batch_dir / "metrics.csv")
    numeric = runs.select_dtypes("number").columns.drop("seed")
    runs.groupby(["method", "arm"])[list(numeric)].agg(["mean", "std"]).to_csv(
        batch_dir / "mean_std_over_seeds.csv")
    text = table(runs, config["seeds"])
    (batch_dir / "mean_std_over_seeds.md").write_text(text)
    print(text)

    FIGURES_DIR.mkdir(exist_ok=True)
    run_dir = Path(runs.loc[runs["seed"] == EXAMPLE_SEED, "run_dir"].iloc[0])
    run_dir = RESULTS_DIR / run_dir.name
    plot_tracking(run_dir, config["arms"], config["methods"],
                  FIGURES_DIR / f"{tag}_tracking_seed{EXAMPLE_SEED}")
    for arm in config["arms"]:
        plot_states(run_dir, arm, config["methods"],
                    FIGURES_DIR / f"{tag}_states_{arm}_seed{EXAMPLE_SEED}")
    links = [f"# Figures for {tag} (seed {EXAMPLE_SEED}, test split)", "",
             f"- [Arm comparison](../../figures/{tag}_tracking_seed{EXAMPLE_SEED}.pdf)"]
    links += [f"- [States, {ARM_LABELS[arm]}](../../figures/{tag}_states_{arm}_seed"
              f"{EXAMPLE_SEED}.pdf)" for arm in config["arms"]]
    links += ["", "Dashed black lines are the true components or the zero targets. "
              "Bands show mean ± one marginal standard deviation. Matching PGF files "
              "sit beside each PDF."]
    (batch_dir / "figures.md").write_text("\n".join(links) + "\n")
    print(f"Saved tables in {batch_dir} and figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
