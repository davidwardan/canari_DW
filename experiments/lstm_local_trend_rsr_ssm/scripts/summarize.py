"""Tables and figures for one batch of `run_experiment.py`, from its saved results.

A series' metric is its mean over seeds; tables report the median and the IQR over the
series, per method and arm. Figures use a predeclared seed and method (see README).

Run from the repository root:

    python experiments/lstm_local_trend_rsr_ssm/scripts/summarize.py [--tag RUN_TAG]
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"

SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (6.5, 3.5)
PREAMBLE = r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}"
mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": PREAMBLE,
        "text.latex.preamble": PREAMBLE,
        "lines.linewidth": 1,
        "figure.figsize": SINGLE_COL,
        "font.size": 9,
        "savefig.dpi": 300,
    }
)

METHOD_LABELS = {
    "offline": "Offline (frozen)",
    "rsr_replay_filtered": "RSR replay filtered",
    "rsr_replay_smoothed": "RSR replay smoothed",
}
COLUMN_TITLES = {
    "offline": "Offline",
    "rsr_replay_filtered": "Replay filt.",
    "rsr_replay_smoothed": "Replay smooth.",
}
ARM_LABELS = {"unconstrained": "Unconstrained", "ewma": "EWMA", "ssm": "Aux SSM"}
# Same colors as lstm_ewma_constraints (native CANARI / moderate constraint)
ARM_COLORS = {"unconstrained": "tab:orange", "ewma": "tab:blue", "ssm": "tab:green"}
METRICS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "ll_mean": "LL",
    "crps": "CRPS",
    "coverage_95": "95% cov.",
    "trend_mean": "Trend",
    "trend_rmse_first": "Trend dev. first 3 y",
    "trend_rmse_last": "Trend dev. last 3 y",
    "lstm_drift_slope": "LSTM drift",
    "lstm_drift_abs": "Abs. LSTM drift",
}
FIGURE_METRICS = {
    "rmse": "RMSE [std]",
    "crps": "CRPS [std]",
    "ll_mean": "LL [nats/step]",
    "trend_rmse_last": "Trend dev. last 3 y\n[$10^{-3}$ std/week]",
    "lstm_drift_abs": "$|$LSTM drift$|$\n[$10^{-4}$ std/week]",
}
FIGURE_SCALE = {"trend_rmse_last": 1e3, "lstm_drift_abs": 1e4}
# Predeclared before the benchmark results were inspected
EXAMPLE_SEED = 1
EXAMPLE_METHOD = "rsr_replay_filtered"
STATE_LABELS = {
    "level": "Level [std]",
    "trend": "Trend\n[std/week]",
    "lstm": "LSTM [std]",
    "white noise": "White noise\n[std]",
}
# Auxiliary states of the constrained arms
MEMORY_LABELS = {
    "ewma": {"lstm ewma fast": "Fast EWMA\n$M_f$ [std]",
             "lstm ewma slow": "Slow EWMA\n$M_s$ [std]"},
    "ssm": {"lstm aux level": "Aux level\n$M_L$ [std]",
            "lstm aux slope": "Aux slope\n$M_S$\n[std/week]"},
}


def save(fig, path: Path) -> None:
    for suffix in (".pdf", ".pgf"):
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def fmt(metric: str, value: float) -> str:
    if metric.startswith("lstm_drift"):
        return f"{value:.2e}"
    return f"{value:.{5 if metric.startswith('trend') else 3}f}"


def summarize(per_series: pd.DataFrame) -> pd.DataFrame:
    """Median and quartiles over series, per method, arm and metric."""

    grouped = per_series.groupby(["method", "arm"])[list(METRICS)]
    summary = pd.concat(
        {
            "median": grouped.median(),
            "q25": grouped.quantile(0.25),
            "q75": grouped.quantile(0.75),
        },
        axis=1,
    )
    order = [(method, arm) for method in METHOD_LABELS for arm in ARM_LABELS
             if (method, arm) in summary.index]
    return summary.reorder_levels([1, 0], axis=1).loc[order]


def format_table(summary: pd.DataFrame) -> pd.DataFrame:
    def cell(metric, key):
        values = [fmt(metric, summary.loc[key, (metric, q)]) for q in ("median", "q25", "q75")]
        return "{} [{}, {}]".format(*values)

    return pd.DataFrame(
        {METRICS[metric]: [cell(metric, key) for key in summary.index] for metric in METRICS},
        index=[f"{METHOD_LABELS[method]}, {ARM_LABELS[arm]}" for method, arm in summary.index],
    )


def to_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Method, arm | " + " | ".join(table.columns) + " |",
        "| :--- |" + " ---: |" * len(table.columns),
    ]
    lines += [f"| {key} | " + " | ".join(row) + " |" for key, row in table.iterrows()]
    return "\n".join(lines)


def plot_metrics(summary: pd.DataFrame, path: Path) -> None:
    methods = list(dict.fromkeys(summary.index.get_level_values("method")))
    arms = list(dict.fromkeys(summary.index.get_level_values("arm")))
    fig, axes = plt.subplots(1, len(FIGURE_METRICS), figsize=(DOUBLE_COL[0], 1.9), sharey=True)
    positions = dict(zip(methods, range(len(methods))[::-1]))
    for ax, (metric, label) in zip(axes, FIGURE_METRICS.items()):
        scale = FIGURE_SCALE.get(metric, 1)
        for offset, arm in zip(np.linspace(0.2, -0.2, len(arms)), arms):
            rows = summary.xs(arm, level="arm").loc[methods]
            median = scale * rows[(metric, "median")]
            ax.errorbar(
                median,
                [positions[method] + offset for method in methods],
                xerr=[median - scale * rows[(metric, "q25")],
                      scale * rows[(metric, "q75")] - median],
                fmt="o",
                color=ARM_COLORS[arm],
                markersize=3,
                capsize=2,
                label=ARM_LABELS[arm],
            )
        ax.set_xlabel(label)
        ax.margins(x=0.15)
        ax.xaxis.set_major_locator(MaxNLocator(3))
    axes[0].set_yticks(list(positions.values()), [METHOD_LABELS[m] for m in positions])
    axes[0].set_ylim(-0.5, len(methods) - 0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1),
               ncol=len(arms), frameon=False)
    fig.tight_layout()
    save(fig, path)


def load_history(runs: pd.DataFrame, series: str, arm: str, method: str) -> pd.DataFrame:
    """Full timeline of one run (seed EXAMPLE_SEED), with explicit split membership."""

    row = runs[(runs["series"] == series) & (runs["arm"] == arm)
               & (runs["seed"] == EXAMPLE_SEED) & (runs["method"] == method)]
    run_dir = RESULTS_DIR / Path(row["run_dir"].item()).name
    return pd.read_csv(run_dir / f"full_history_{method}.csv", index_col=0, parse_dates=True)


def mark_splits(ax, predictions: pd.DataFrame, method: str, window_len: int) -> None:
    validation_start = predictions.index[predictions["split"] == "validation"][0]
    test_start = predictions.index[predictions["split"] == "test"][0]
    ax.axvspan(predictions.index[0], validation_start, color="0.92", label="Training")
    ax.axvspan(validation_start, test_start, color="#fff1cf", label="Validation")
    for boundary in (validation_start, test_start):
        ax.axvline(boundary, color="0.4", linestyle="--", linewidth=0.7)
    if method != "offline":
        test_position = predictions.index.get_loc(test_start)
        ax.axvline(
            predictions.index[test_position - window_len], color="0.2",
            linestyle=":", linewidth=0.8, label="RSR warm-up begins",
        )
    ax.set_xlim(predictions.index[0], predictions.index[-1])


def format_dates(ax, time: pd.DatetimeIndex) -> None:
    # Retain both endpoints, avoiding crowded neighboring calendar-year ticks.
    interior = pd.date_range(time[0], time[-1], freq="5YS")
    panel_width = ax.figure.get_figwidth() * ax.get_position().width
    margin = pd.Timedelta(days=1095 if panel_width < 3 else 550)
    ticks = [time[0], *interior[(interior > time[0] + margin) &
                              (interior < time[-1] - margin)], time[-1]]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def plot_band(ax, time, mean, std, label=None, linewidth=1) -> None:
    ax.fill_between(time, mean - std, mean + std, color="tab:blue", alpha=0.3,
                    linewidth=0, label=r"$\pm\sigma$" if label else None)
    ax.plot(time, mean, color="tab:blue", linewidth=linewidth, label=label)


def plot_example(runs, series: str, arms: list, window_len: int, path: Path) -> None:
    """Predeclared method and seed; one row per arm."""

    fig, axes = plt.subplots(len(arms), 1, figsize=(DOUBLE_COL[0], 1.2 * len(arms) + 1),
                             sharex=True,
                             sharey=True, squeeze=False)
    axes = axes[:, 0]
    for ax, arm in zip(axes, arms):
        predictions = load_history(runs, series, arm, EXAMPLE_METHOD)
        mark_splits(ax, predictions, EXAMPLE_METHOD, window_len)
        ax.plot(predictions.index, predictions["obs"], color="tab:red", label="observation")
        plot_band(ax, predictions.index, predictions["mean"], predictions["std"], "prediction")
        ax.set_ylabel(f"{ARM_LABELS[arm]}\n$y$ [std]")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1),
               ncol=3, frameon=False)
    format_dates(axes[-1], predictions.index)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    save(fig, path)


def plot_states(runs, series: str, arm: str, methods: list, slope: float,
                window_len: int, path: Path) -> None:
    """One series and arm per figure: observations and states, one column per method."""

    state_labels = {**STATE_LABELS, **MEMORY_LABELS.get(arm, {})}
    fig, axes = plt.subplots(
        1 + len(state_labels), len(methods),
        figsize=(DOUBLE_COL[0], 7.5 + 1.0 * (len(state_labels) - len(STATE_LABELS))),
        sharex=True, sharey="row", squeeze=False,
    )
    for column, method in zip(axes.T, methods):
        predictions = load_history(runs, series, arm, method)
        time = predictions.index
        for ax in column:
            mark_splits(ax, predictions, method, window_len)
        column[0].set_title(COLUMN_TITLES[method])
        column[0].plot(time, predictions["obs"], color="tab:red", linewidth=0.5,
                       label="observation")
        plot_band(column[0], time, predictions["mean"], predictions["std"], "prediction", 0.5)
        for ax, name in zip(column[1:], state_labels):
            plot_band(ax, time, predictions[f"{name}_mean"], predictions[f"{name}_std"],
                      linewidth=0.5)
        trend_ax = column[1 + list(state_labels).index("trend")]
        trend_ax.axhline(slope, color="tab:red", linestyle="--", label="added slope")
    for ax, label in zip(axes[:, 0], ["$y$ [std]", *state_labels.values()]):
        ax.set_ylabel(label)
    for ax in axes[-1]:
        format_dates(ax, time)
    fig.supxlabel("Date", fontsize=mpl.rcParams["font.size"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    trend_handles, trend_labels = axes[1 + list(state_labels).index("trend"), 0] \
        .get_legend_handles_labels()
    online_handles, online_labels = axes[0, -1].get_legend_handles_labels()
    extra = [(h, l) for h, l in zip(online_handles, online_labels) if l == "RSR warm-up begins"]
    fig.legend(
        handles + trend_handles[-1:] + [h for h, _ in extra],
        labels + trend_labels[-1:] + [l for _, l in extra],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="batch run tag; defaults to the latest batch")
    args = parser.parse_args()

    tag = args.tag or sorted(
        path.name for path in RESULTS_DIR.iterdir() if (path / "jobs.json").exists()
    )[-1]
    batch_dir = RESULTS_DIR / tag
    config = json.loads((batch_dir / "config.json").read_text())
    runs = pd.read_csv(batch_dir / "metrics.csv")
    runs["lstm_drift_abs"] = runs["lstm_drift_slope"].abs()
    keys = ["series", "method", "arm"]

    seed_summary = runs.groupby(keys)[list(METRICS)].agg(["mean", "std"])
    seed_summary.to_csv(batch_dir / "per_series_mean_std_over_seeds.csv")
    seed_lines = [
        f"Mean ± sample standard deviation over seeds {config['seeds']}; "
        "test split, data units. Standard deviation is unavailable for a single seed.",
        "",
        "| Series | Method | Arm | " + " | ".join(METRICS.values()) + " |",
        "| :--- | :--- | :--- |" + " ---: |" * len(METRICS),
    ]
    for (series, method, arm), row in seed_summary.iterrows():
        cells = []
        for metric in METRICS:
            std = row[(metric, "std")]
            cells.append(f"{fmt(metric, row[(metric, 'mean')])} ± "
                         + (fmt(metric, std) if pd.notna(std) else "N/A"))
        seed_lines.append(f"| {series} | {METHOD_LABELS[method]} | {ARM_LABELS[arm]} | "
                          + " | ".join(cells) + " |")
    (batch_dir / "per_series_mean_std_over_seeds.md").write_text("\n".join(seed_lines) + "\n")

    per_series = runs.groupby(keys, as_index=False)[list(METRICS)].mean()
    per_series.to_csv(batch_dir / "per_series_mean_over_seeds.csv", index=False)
    summary = summarize(per_series)
    summary.to_csv(batch_dir / "summary_median_iqr.csv")
    table = format_table(summary)
    (batch_dir / "summary_median_iqr.md").write_text(
        f"Median [Q1, Q3] over {per_series['series'].nunique()} series "
        f"(each series: mean over seeds {config['seeds']}); test split, data units. "
        "Abs. LSTM drift is averaged over seeds after taking each run's absolute value.\n\n"
        + to_markdown(table)
        + "\n"
    )
    latex = [r"\begin{tabular}{l" + "r" * len(table.columns) + r"}", r"\toprule",
             "Method, arm & " + " & ".join(table.columns).replace("%", r"\%") + r" \\",
             r"\midrule"]
    latex.extend(" & ".join([key, *row]) + r" \\" for key, row in table.iterrows())
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (batch_dir / "summary_median_iqr.tex").write_text("\n".join(latex) + "\n")
    print(to_markdown(table))

    FIGURES_DIR.mkdir(exist_ok=True)
    window_len = config["rsr_window_len"]
    plot_metrics(summary, FIGURES_DIR / f"{tag}_metrics")
    for series in config["series"]:
        plot_example(runs, series, config["arms"], window_len,
                     FIGURES_DIR / f"{tag}_example_{series}")
        for arm in config["arms"]:
            plot_states(runs, series, arm, config["methods"], config["slope"], window_len,
                        FIGURES_DIR / f"{tag}_states_{series}_{arm}")
    figure_links = [
        f"# Figures for {tag}",
        "",
        f"Seed {EXAMPLE_SEED} for every series. Prediction figures compare the arms for the "
        f"predeclared method ({METHOD_LABELS[EXAMPLE_METHOD]}); state figures show every "
        "method, plus the memories of the constrained arms. "
        "Bands show mean plus/minus one standard deviation. Shaded: training / "
        "validation; unshaded: test.",
        "",
        "Training and validation states are reconstructed with the selected offline model; "
        "RSR updates begin at the dotted line in the last validation cycle. Metrics remain "
        "test-only.",
        "",
        "| Series | Predictions (all arms) | "
        + " | ".join(f"States, {ARM_LABELS[arm]}" for arm in config["arms"]) + " |",
        "| :--- | :--- |" + " :--- |" * len(config["arms"]),
    ]
    for series in config["series"]:
        figure_links.append(
            f"| {series} | [PDF](../../figures/{tag}_example_{series}.pdf) | "
            + " | ".join(f"[PDF](../../figures/{tag}_states_{series}_{arm}.pdf)"
                         for arm in config["arms"]) + " |"
        )
    figure_links.extend([
        "", f"[Metric comparison](../../figures/{tag}_metrics.pdf) · "
        "[Test metrics](summary_median_iqr.md) · "
        "[Mean ± std across seeds](per_series_mean_std_over_seeds.md)",
        "", "Matching PGF files are available beside each PDF.",
    ])
    (batch_dir / "figures.md").write_text("\n".join(figure_links) + "\n")
    print(f"Saved tables in {batch_dir} and figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
