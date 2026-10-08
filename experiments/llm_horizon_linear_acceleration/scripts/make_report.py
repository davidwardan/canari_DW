"""Generate H-sweep tables and figures from a completed run."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR.parent / "llm_horizon_models/scripts"))
from models import MODELS, common, slug  # noqa: E402

LABELS = {model: label for model, label, *_ in MODELS} | {"acceleration_only": "Local acceleration + white noise"}
METRICS = ["rmse", "mae", "log_lik", "crps", "coverage90", "width90", "slope_mae", "acceleration_mae"]
SUMMARY = METRICS + ["n_scored", "foundation_calls", "elapsed_seconds"]


def table(frame, keys, labels):
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for key in keys:
            value = row[key]
            if key == "model": cells.append(LABELS.get(value, value))
            elif key in ("horizon", "n_series", "n_scored", "foundation_calls"): cells.append(f"{value:g}")
            else: cells.append("—" if pd.isna(value) else f"{value:.4g}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize(metrics):
    grouped = metrics.groupby(["model", "horizon"], sort=False)
    median = grouped[SUMMARY].median().reset_index()
    median["n_series"] = grouped.size().to_numpy()
    quantiles = grouped[METRICS].quantile([.25, .5, .75]).reset_index().rename(columns={"level_2": "quantile"})
    foundation = metrics[metrics.model != "acceleration_only"].copy()
    h1 = foundation[foundation.horizon == 1][["model", "series"] + SUMMARY]
    paired = foundation.merge(h1, on=["model", "series"], suffixes=("", "_h1"), validate="many_to_one")
    for key in ("rmse", "mae", "crps", "width90", "foundation_calls"):
        paired[f"{key}_ratio_h1"] = paired[key] / paired[f"{key}_h1"].replace(0, np.nan)
    paired["log_lik_difference_h1"] = paired.log_lik - paired.log_lik_h1
    paired["coverage90_difference_h1"] = paired.coverage90 - paired.coverage90_h1
    derived = [c for c in paired if c.endswith(("_ratio_h1", "_difference_h1"))]
    paired_summary = paired.groupby(["model", "horizon"], sort=False)[derived].median().reset_index()
    return median, quantiles, paired, paired_summary


def save_figure(fig, name, directory):
    for extension in ("pdf", "pgf", "png"):
        fig.savefig(directory / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)


def metric_figure(summary, key, ylabel, directory):
    fig, ax = plt.subplots()
    scale = 100 if key == "coverage90" else 1
    for model, label, color, linestyle in MODELS:
        rows = summary[summary.model == model].sort_values("horizon")
        if len(rows): ax.plot(rows.horizon, scale * rows[key], color=color, ls=linestyle, label=label)
    baseline = summary[summary.model == "acceleration_only"]
    if len(baseline): ax.axhline(scale * baseline.iloc[0][key], color="0.25", ls=":", label="Acceleration + white noise")
    if key == "coverage90":
        ax.axhline(90, color="0.6", ls="--", lw=.7, label="Nominal 90\%")
        ax.set_ylim(0, 102)
    horizons = sorted(summary.loc[summary.horizon > 0, "horizon"].unique())
    ax.set_xscale("log"); ax.set_xticks(horizons, labels=[str(h) for h in horizons]); ax.minorticks_off()
    ax.set_xlabel(r"Chronos refresh interval $H$ (weeks)"); ax.set_ylabel(ylabel)
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    save_figure(fig, f"{key}_vs_horizon", directory)


def examples(run):
    base = run / "raw" / slug("amazon/chronos-2")
    return {h: pd.read_csv(base / f"ts67_H{h}.csv", parse_dates=["date"])
            for h in (1, 52, 208) if (base / f"ts67_H{h}.csv").exists()}


def forecast_figure(frames, directory):
    if not frames: return
    fig, axes = plt.subplots(len(frames), 1, sharex=True, sharey=True, figsize=common.DOUBLE_COL, squeeze=False)
    for ax, (horizon, frame) in zip(axes[:, 0], frames.items()):
        view = frame.tail(208); dates = view.date.to_numpy(); mu = view.mu.to_numpy()
        radius = np.sqrt(view["var"].to_numpy())
        ax.plot(dates, view.y, color="tab:red", label="Observation")
        ax.plot(dates, mu, color="tab:blue", label="Prediction")
        ax.fill_between(dates, mu-radius, mu+radius, color="tab:blue", alpha=.3, linewidth=0, label=r"$\pm 1$ std")
        ax.set_ylabel(rf"$H={horizon}$" + "\nValue (s.u.)")
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=3, frameon=False)
    axes[-1, 0].xaxis.set_major_locator(mdates.YearLocator()); axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y")); axes[-1, 0].set_xlabel("Date (year)")
    fig.subplots_adjust(hspace=.18); save_figure(fig, "ts67_forecast_examples", directory)


def acceleration_figure(frames, directory):
    if not frames: return
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=common.DOUBLE_COL)
    styles = {1: "-", 52: "--", 208: ":"}
    for horizon, frame in frames.items():
        axes[0].plot(frame.date, frame.level_posterior, color="tab:blue", ls=styles[horizon], label=rf"Estimated, $H={horizon}$")
        axes[1].plot(frame.date, frame.slope_posterior, color="tab:blue", ls=styles[horizon])
        axes[2].plot(frame.date, frame.acceleration_posterior, color="tab:blue", ls=styles[horizon])
    first = next(iter(frames.values()))
    axes[0].plot(first.date, first.synthetic_acceleration_trend, color="0.3", ls="-.", label="Injected quadratic trend")
    axes[1].plot(first.date, first.synthetic_slope, color="0.3", ls="-.")
    axes[2].axhline(first.synthetic_acceleration.iloc[0], color="0.3", ls="-.")
    axes[0].set_ylabel("Level (s.u.)"); axes[1].set_ylabel("Slope (s.u./week)"); axes[2].set_ylabel("Acceleration (s.u./week$^2$)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6)); axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y")); axes[2].set_xlabel("Date (year)")
    fig.subplots_adjust(hspace=.16); save_figure(fig, "ts67_acceleration_diagnostics", directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run", type=Path); args = parser.parse_args()
    run = args.run.resolve(); config = json.loads((run / "config.json").read_text()); metrics = pd.read_csv(run / "metrics.csv")
    if metrics.duplicated(["model", "series", "horizon"]).any(): raise ValueError("Duplicate metric rows")
    figure_dir = EXP_DIR / "figures" / run.name
    if figure_dir.exists(): raise FileExistsError("Refusing to overwrite figure directory")
    median, quantiles, paired, paired_summary = summarize(metrics)
    run.joinpath("summary.csv").write_text(median.to_csv(index=False)); quantiles.to_csv(run / "summary_quantiles.csv", index=False); paired.to_csv(run / "paired_h1.csv", index=False); paired_summary.to_csv(run / "paired_h1_summary.csv", index=False)
    figure_dir.mkdir(parents=True)
    common.use_paper_style()
    for key, ylabel in (("rmse", "RMSE (standardized units)"), ("crps", "CRPS (standardized units)"), ("log_lik", "Mean log-likelihood"), ("coverage90", r"90\% predictive coverage (\%)")):
        metric_figure(median, key, ylabel, figure_dir)
    frames = examples(run); forecast_figure(frames, figure_dir); acceleration_figure(frames, figure_dir)
    horizons = sorted(median.loc[median.horizon > 0, "horizon"].unique()); h1 = median[median.horizon == 1]; endpoint = paired_summary[paired_summary.horizon == max(horizons)]; baseline = median[median.model == "acceleration_only"]
    injected = config["acceleration_per_year2"]
    lines = ["# Chronos refresh interval with LocalAcceleration", "", f"Run `{run.name}` uses the same ten preprocessed benchmark series, six frozen Chronos models, and H = {', '.join(map(str, horizons))} weeks.", "", "## Design", "", "Each model is LocalAcceleration + Chronos Auxiliary + WhiteNoise. LocalAcceleration has zero process noise, so its latent path is a constant-acceleration quadratic; level, slope and acceleration are updated weekly. Chronos is refreshed once per H-week block. The acceleration-only baseline uses the same warmup initialization and weekly updates without Chronos.", "", f"Each series is standardized on its first 52 observations, then receives the synthetic trend g(t)=0.5 a t^2 with a={injected} warmup-standardized units per year squared. The benchmark's existing full-series preprocessing is retained by explicit user choice; the added trend is synthetic and the residual can still contain drift. The injected acceleration is not provided to the model.", "", "Warmup OLS initializes the quadratic state and Chronos receives the warmup residual after that fitted curve is removed. Missing observations preserve predictive covariance and do not contribute to scores. Contexts are capped at 512 residual states and independent series use Chronos-2 cross_learning=False. Checkpoint revisions, hashes, input provenance and package versions are in config.json.", "", "The Gaussian working variance combines the LocalAcceleration state, Chronos quantile spread and WhiteNoise variance. These are total predictive intervals, not an epistemic/aleatoric decomposition. Scores use the existing Gaussian quantile adapter.", "", "## H comparison", "", "Medians give every series equal weight; the IQR is saved separately and is not a confidence interval.", "", table(h1, ["model", "rmse", "crps", "log_lik", "coverage90", "width90", "slope_mae", "acceleration_mae"], ["Model", "RMSE", "CRPS", "Mean log-likelihood", "90% coverage", "90% width", "Slope MAE", "Acceleration MAE"]), "", f"### H={max(horizons)} relative to H=1", "", "Ratios and log-likelihood changes are computed per series before taking medians.", "", table(endpoint, ["model", "rmse_ratio_h1", "crps_ratio_h1", "log_lik_difference_h1", "coverage90_difference_h1", "foundation_calls_ratio_h1"], ["Model", "RMSE ratio", "CRPS ratio", "Log-likelihood change", "Coverage change", "Call ratio"]), "", "### Acceleration-only baseline", "", table(baseline, ["model", "rmse", "crps", "log_lik", "coverage90", "width90", "slope_mae", "acceleration_mae"], ["Model", "RMSE", "CRPS", "Mean log-likelihood", "90% coverage", "90% width", "Slope MAE", "Acceleration MAE"]), "", "## Figures", ""]
    captions = {"rmse_vs_horizon": "Median RMSE across series.", "crps_vs_horizon": "Median Gaussian CRPS across series.", "log_lik_vs_horizon": "Median per-series mean Gaussian log-likelihood.", "coverage90_vs_horizon": "Median 90% predictive coverage; interval widths are in the summary table.", "ts67_forecast_examples": "Chronos-2 on ts67 with total predictive mean ± 1 std.", "ts67_acceleration_diagnostics": "Chronos-2 on ts67: estimated level, slope and acceleration against the imposed quadratic trend. This is a diagnostic, not uniquely identifiable component truth."}
    for name, caption in captions.items():
        if (figure_dir / f"{name}.png").exists(): lines += [f"![{caption}]({figure_dir / (name + '.png')})", "", f"{caption} [PDF]({figure_dir / (name + '.pdf')}) · [PGF]({figure_dir / (name + '.pgf')})", ""]
    lines += ["## Evidence and limits", "", "Raw per-series forecasts and state trajectories are stored under raw/. The previous processed benchmark uses full-series detrending, so this is a controlled H comparison and not a fully causal raw-series analysis. Trend and residual components are not uniquely identifiable. The imposed acceleration is one magnitude and does not establish performance for other curvatures or changing acceleration. The reports use total predictive intervals; no uncertainty decomposition is claimed.", ""]
    (run / "report.md").write_text("\n".join(lines)); print(f"Report: {run / 'report.md'}\nFigures: {figure_dir}")


if __name__ == "__main__": main()
