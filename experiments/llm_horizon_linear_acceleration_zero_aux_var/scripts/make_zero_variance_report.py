"""Generate tables, figures and a paired comparison with the reference run."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
REFERENCE_RUN = EXP_DIR.parent / "llm_horizon_linear_acceleration" / "results" / "run_20260925T103556_080836Z"
sys.path.insert(0, str(EXP_DIR.parent / "llm_horizon_linear_acceleration" / "scripts"))
from make_report import acceleration_figure, common, examples, forecast_figure, metric_figure, summarize, table  # noqa: E402

KEYS = ["model", "series", "horizon"]


def compare(metrics, reference):
    """Per-series ratios/differences against the non-zero-variance reference run."""
    paired = metrics.merge(reference, on=KEYS, suffixes=("", "_ref"), validate="one_to_one")
    if len(paired) != len(metrics):
        raise ValueError("Reference run does not cover every model/series/H row")
    for key in ("rmse", "crps", "width90"):
        paired[f"{key}_ratio_ref"] = paired[key] / paired[f"{key}_ref"]
    for key in ("log_lik", "coverage90"):
        paired[f"{key}_difference_ref"] = paired[key] - paired[f"{key}_ref"]
    derived = [c for c in paired if c.endswith(("_ratio_ref", "_difference_ref"))]
    return paired, paired.groupby(["model", "horizon"], sort=False)[derived].median().reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run", type=Path); args = parser.parse_args()
    run = args.run.resolve(); config = json.loads((run / "config.json").read_text()); metrics = pd.read_csv(run / "metrics.csv")
    if not (run / "COMPLETED").exists(): raise FileNotFoundError("Run is not completed")
    if metrics.duplicated(KEYS).any(): raise ValueError("Duplicate metric rows")
    figure_dir = EXP_DIR / "figures" / run.name
    if figure_dir.exists(): raise FileExistsError("Refusing to overwrite figure directory")
    median, quantiles, paired, paired_summary = summarize(metrics)
    median.to_csv(run / "summary.csv", index=False); quantiles.to_csv(run / "summary_quantiles.csv", index=False); paired.to_csv(run / "paired_h1.csv", index=False); paired_summary.to_csv(run / "paired_h1_summary.csv", index=False)
    versus, versus_summary = compare(metrics, pd.read_csv(REFERENCE_RUN / "metrics.csv"))
    versus.to_csv(run / "paired_reference.csv", index=False); versus_summary.to_csv(run / "paired_reference_summary.csv", index=False)
    baseline_var = {f.name.split("_")[0]: pd.read_csv(f)["var"].to_numpy() for f in (run / "raw" / "acceleration_only").glob("*.csv")}
    variance_gap = max(float(abs(pd.read_csv(f)["var"].to_numpy() / baseline_var[f.name.split("_")[0]] - 1).max())
                       for f in (run / "raw").glob("*/*_H*.csv") if f.parent.name != "acceleration_only")
    figure_dir.mkdir(parents=True)
    common.use_paper_style()
    for key, ylabel in (("rmse", "RMSE (standardized units)"), ("crps", "CRPS (standardized units)"), ("log_lik", "Mean log-likelihood"), ("coverage90", r"90\% predictive coverage (\%)")):
        metric_figure(median, key, ylabel, figure_dir)
    frames = examples(run); forecast_figure(frames, figure_dir); acceleration_figure(frames, figure_dir)
    horizons = sorted(median.loc[median.horizon > 0, "horizon"].unique()); h1 = median[median.horizon == 1]; hmax = median[median.horizon == max(horizons)]
    endpoint = paired_summary[paired_summary.horizon == max(horizons)]; baseline = median[median.model == "acceleration_only"]
    versus_view = versus_summary[versus_summary.horizon.isin([1, max(horizons)])].sort_values(["horizon"], kind="stable")
    score_keys = ["model", "rmse", "crps", "log_lik", "coverage90", "width90", "slope_mae", "acceleration_mae"]
    score_labels = ["Model", "RMSE", "CRPS", "Mean log-likelihood", "90% coverage", "90% width", "Slope MAE", "Acceleration MAE"]
    lines = ["# LocalAcceleration H sweep with zero Chronos variance", "", f"Run `{run.name}` repeats `{REFERENCE_RUN.relative_to(EXP_DIR.parent)}` with the Chronos variance forced to exactly zero ({config['chronos_variance']}). Same ten series, six frozen Chronos models and H = {', '.join(map(str, horizons))} weeks.", "", "## What zero variance implies", "", "The auxiliary transition is 0, so with zero Chronos variance the auxiliary state has zero prior variance and zero cross-covariance with the trend states. Its Kalman gain is zero: the auxiliary posterior equals the Chronos mean, and that mean is what is appended to the Chronos context. After the warmup Chronos therefore conditions only on its own past forecasts, never on observations, and level, slope and acceleration absorb the whole innovation. The predictive variance is the LocalAcceleration state variance plus WhiteNoise (0.01) only. This is total predictive variance, not an epistemic/aleatoric decomposition.", "", f"The covariance recursion is therefore that of the acceleration-only baseline: across all completed groups the maximum relative difference between the predictive variance and the baseline's is {variance_gap:.3g}. Only the predictive mean differs, by the self-fed Chronos forecast.", "", "## Scores", "", "Medians over series; the IQR is saved in summary_quantiles.csv and is not a confidence interval.", "", "### H=1", "", table(h1, score_keys, score_labels), "", f"### H={max(horizons)}", "", table(hmax, score_keys, score_labels), "", f"### H={max(horizons)} relative to H=1 (this run)", "", table(endpoint, ["model", "rmse_ratio_h1", "crps_ratio_h1", "log_lik_difference_h1", "coverage90_difference_h1"], ["Model", "RMSE ratio", "CRPS ratio", "Log-likelihood change", "Coverage change"]), "", "### Acceleration-only baseline", "", table(baseline, score_keys, score_labels), "", "## Zero variance relative to the reference run", "", "Computed per series (zero-variance / reference, or zero-variance − reference) before taking medians over the ten series.", "", table(versus_view, ["model", "horizon", "rmse_ratio_ref", "crps_ratio_ref", "log_lik_difference_ref", "coverage90_difference_ref", "width90_ratio_ref"], ["Model", "H", "RMSE ratio", "CRPS ratio", "Log-likelihood change", "Coverage change", "Width ratio"]), "", "All H values are in paired_reference_summary.csv.", ""]
    failures = run / "failures.csv"
    if failures.exists():
        failed = pd.read_csv(failures)
        lines += ["## Diverged groups", "", "These model/H groups stopped because the self-fed Chronos rollout overflowed float32 and returned non-finite or crossed quantiles. They have no metric rows and are absent from the tables and figures.", "", table(failed[["model", "horizon"]], ["model", "horizon"], ["Model", "H"]), ""]
    lines += ["## Figures", ""]
    captions = {"rmse_vs_horizon": "Median RMSE across series.", "crps_vs_horizon": "Median Gaussian CRPS across series.", "log_lik_vs_horizon": "Median per-series mean Gaussian log-likelihood.", "coverage90_vs_horizon": "Median 90% predictive coverage.", "ts67_forecast_examples": "Chronos-2 on ts67 with total predictive mean ± 1 std (Chronos variance zero).", "ts67_acceleration_diagnostics": "Chronos-2 on ts67: estimated level, slope and acceleration against the imposed quadratic trend. Diagnostic only; components are not uniquely identifiable."}
    for name, caption in captions.items():
        if (figure_dir / f"{name}.png").exists(): lines += [f"![{caption}]({figure_dir / (name + '.png')})", "", f"{caption} [PDF]({figure_dir / (name + '.pdf')}) · [PGF]({figure_dir / (name + '.pgf')})", ""]
    lines += ["## Evidence and limits", "", "Raw per-series forecasts, state trajectories and native Chronos quantiles are under raw/. The same preprocessing limits as the reference run apply: full-series detrending in the processed benchmark, a single synthetic curvature, and non-identifiable trend/residual components.", ""]
    (run / "report.md").write_text("\n".join(lines)); print(f"Report: {run / 'report.md'}\nFigures: {figure_dir}")


if __name__ == "__main__": main()
