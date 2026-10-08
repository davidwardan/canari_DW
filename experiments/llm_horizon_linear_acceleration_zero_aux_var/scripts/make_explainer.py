"""Write why_not_zero_variance.md and its figures from the saved runs."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
ZERO_RUN = EXP_DIR / "results" / "run_20260925T131113_839408Z"
REFERENCE_RUN = EXP_DIR.parent / "llm_horizon_linear_acceleration" / "results" / "run_20260925T103556_080836Z"
FIGURE_DIR = EXP_DIR / "figures" / "why_not_zero_variance"
sys.path.insert(0, str(EXP_DIR.parent / "llm_horizon_models/scripts"))
from models import MODELS, common, slug  # noqa: E402

FEEDBACK_SERIES, INTERVAL_SERIES = "ts57", "ts67"
Z90 = 1.6448536269514722


def raw(run, model, series, horizon):
    return pd.read_csv(run / "raw" / slug(model) / f"{series}_H{horizon}.csv", parse_dates=["date"])


def save(fig, name):
    for extension in ("pdf", "pgf", "png"):
        fig.savefig(FIGURE_DIR / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)


def feedback_figure():
    zero, reference = raw(ZERO_RUN, "amazon/chronos-2", FEEDBACK_SERIES, 1), raw(REFERENCE_RUN, "amazon/chronos-2", FEEDBACK_SERIES, 1)
    fig, ax = plt.subplots()
    ax.plot(reference.date, reference.aux_mu.abs(), color="0.5", label="Chronos mean, reference")
    ax.plot(zero.date, zero.aux_mu.abs(), color="tab:blue", label="Chronos mean, zero variance")
    ax.plot(zero.date, zero.level_prior.abs(), color="tab:blue", ls="--", label="Level, zero variance")
    ax.set_yscale("log"); ax.set_ylabel("Absolute value (s.u.)"); ax.set_xlabel("Date (year)")
    ax.xaxis.set_major_locator(mdates.YearLocator(4)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1, frameon=False)
    save(fig, "feedback_divergence")
    return zero, reference


def interval_figure():
    frames = {"Reference": raw(REFERENCE_RUN, "amazon/chronos-2", INTERVAL_SERIES, 208),
              "Zero variance": raw(ZERO_RUN, "amazon/chronos-2", INTERVAL_SERIES, 208)}
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=common.DOUBLE_COL)
    for ax, (label, frame) in zip(axes, frames.items()):
        view = frame.tail(156); mu = view.mu.to_numpy(); radius = np.sqrt(view["var"].to_numpy())
        ax.plot(view.date, view.y, color="tab:red", label="Observation")
        ax.plot(view.date, mu, color="tab:blue", label="Prediction")
        ax.fill_between(view.date, mu - radius, mu + radius, color="tab:blue", alpha=.3, linewidth=0, label=r"$\pm 1$ std")
        ax.set_ylabel(f"{label}\nValue (s.u.)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=3, frameon=False)
    axes[1].xaxis.set_major_locator(mdates.YearLocator()); axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y")); axes[1].set_xlabel("Date (year)")
    fig.subplots_adjust(hspace=.15)
    save(fig, "interval_collapse")
    return frames


def coverage(frame):
    valid = frame.y.notna()
    return float(np.mean(np.abs(frame.y[valid] - frame.mu[valid]) <= Z90 * np.sqrt(frame["var"][valid])))


def max_aux(run, horizon="*", model="*"):
    return max(pd.read_csv(f, usecols=["aux_mu"]).aux_mu.abs().max() for f in (run / "raw").glob(f"{model}/*_H{horizon}.csv"))


def divergence_table(zero_metrics, reference_metrics, failures, threshold):
    rows = []
    for horizon in sorted(reference_metrics.loc[reference_metrics.horizon > 0, "horizon"].unique()):
        diverged = 0
        for model, *_ in MODELS:
            failed = ((failures.model == model) & (failures.horizon == horizon)).any()
            files = (ZERO_RUN / "raw" / slug(model)).glob(f"*_H{horizon}.csv")
            diverged += failed or any(pd.read_csv(f, usecols=["aux_mu"]).aux_mu.abs().max() > threshold for f in files)
        c2 = lambda m: m[(m.model == "amazon/chronos-2") & (m.horizon == horizon)]
        rows.append((horizon, int(c2(reference_metrics).foundation_calls.median()), diverged,
                     c2(zero_metrics).rmse.median(), c2(reference_metrics).rmse.median()))
    lines = ["| H (weeks) | Chronos calls per series | Models diverged (of 6) | Chronos-2 RMSE, zero var. | Chronos-2 RMSE, reference |",
             "| --- | --- | --- | --- | --- |"]
    lines += [f"| {h} | {calls} | {d} | {z:.3g} | {r:.3g} |" for h, calls, d, z, r in rows]
    return "\n".join(lines)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    common.use_paper_style()
    zero_metrics, reference_metrics = pd.read_csv(ZERO_RUN / "metrics.csv"), pd.read_csv(REFERENCE_RUN / "metrics.csv")
    failures = pd.read_csv(ZERO_RUN / "failures.csv")
    # Blow-up: 10x beyond the largest Chronos mean seen anywhere in the reference run.
    reference_max = max_aux(REFERENCE_RUN)
    threshold = 10 * reference_max
    zero, reference = feedback_figure()
    intervals = interval_figure()
    h1 = [pd.read_csv(f, usecols=["aux_mu"]).aux_mu.abs().max() for f in (ZERO_RUN / "raw" / "chronos-2").glob("*_H1.csv")]
    baseline = {f.name.split("_")[0]: pd.read_csv(f)["var"].to_numpy() for f in (ZERO_RUN / "raw" / "acceleration_only").glob("*.csv")}
    gap = max(float(np.max(np.abs(pd.read_csv(f)["var"].to_numpy() / baseline[f.name.split("_")[0]] - 1)))
              for f in (ZERO_RUN / "raw").glob("*/*_H*.csv") if f.parent.name != "acceleration_only")
    median = lambda m, model, h, key: m[(m.model == model) & (m.horizon == h)][key].median()
    first = zero.date[zero.aux_mu.abs() > threshold].iloc[0].year
    fig = lambda name, alt: f"![{alt}](figures/why_not_zero_variance/{name}.png)\n\n[PDF](figures/why_not_zero_variance/{name}.pdf) · [PGF](figures/why_not_zero_variance/{name}.pgf)"
    text = f"""# Why the Chronos variance cannot simply be set to zero

Model: `LocalAcceleration + Auxiliary(Chronos) + WhiteNoise`, filtered weekly, with Chronos refreshed every H weeks.
Numbers below come from the zero-variance run `{ZERO_RUN.name}` and the reference run `{REFERENCE_RUN.name}`.
They are generated by `scripts/make_explainer.py`.

**In short:** the Chronos variance does two jobs. It widens the predictive interval, and it sets how much of each forecast error goes back into the Chronos state.
Setting it to zero removes both. Chronos stops seeing the data, its forecasts can blow up, and the intervals become overconfident.

## 1. The variance also sets the Kalman gain

The observation is $y_t = \\ell_t + c_t + v_t$, where $c_t$ is the Chronos (auxiliary) state. Because the auxiliary transition is 0, $c_t$ has no cross-covariance with the other states, and its gain is

$$K_c = \\frac{{\\sigma^2_c}}{{\\sigma^2_\\ell + \\sigma^2_c + \\sigma^2_v}}.$$

With $\\sigma^2_c = 0$, $K_c = 0$. The posterior of $c_t$ is exactly the Chronos forecast, and the whole innovation $y_t - \\hat y_t$ goes to level, slope and acceleration.

## 2. Zero gain: Chronos is fed its own forecasts

Canari appends the posterior of $c_t$ to the Chronos context. With $K_c = 0$, that value is the forecast Chronos just made. After the 52-week warmup, Chronos conditions only on its own output, never on an observation. That is an open-loop rollout, and any drift Chronos extrapolates is read back in and extended again.

Chronos-2, H = 1, series {FEEDBACK_SERIES}:

{fig("feedback_divergence", "Self-fed Chronos mean diverges; the level diverges with it")}

The loop exists in the reference too. Level and Chronos state are not separately identifiable (only their sum is observed), so at small H they drift in opposite directions, here up to {reference.aux_mu.abs().max():.2g}, and up to {reference_max:.2g} across the whole reference run. The observations keep that drift bounded. With zero variance nothing does: the Chronos mean passes 10× the reference maximum in {first} and reaches {zero.aux_mu.abs().max():.1e}. At H = 1, {sum(v > threshold for v in h1)} of the 10 Chronos-2 series blew up this way.

The level blows up with it because it must absorb $y_t - c_t$. The prediction is then the sum of two huge numbers of opposite sign that should nearly cancel. At this scale floating-point rounding is far larger than the signal, which is where the RMSE of about {median(zero_metrics, "amazon/chronos-2", 1, "rmse"):.0e} comes from.

## 3. More refreshes, more feedback

Each refresh is one pass around the feedback loop, so frequent refreshes diverge. A model/H group counts as diverged if any series exceeded |Chronos mean| > {threshold:.2g} (10× the largest value in the reference run), or if the run stopped with a float32 overflow (`failures.csv`).

{divergence_table(zero_metrics, reference_metrics, failures, threshold)}

Where the rollout stays bounded (large H), zero variance is still worse than the reference, because Chronos is only extrapolating the warmup.

## 4. The intervals collapse to the no-Chronos baseline

Removing $\\sigma^2_c$ also removes the variability Chronos was modeling, mostly seasonality. Since $c_t$ has zero variance and no cross-covariance, the covariance recursion is exactly that of `LocalAcceleration + WhiteNoise` without Chronos. Across all completed groups the maximum relative difference in predictive variance from that baseline is {gap:.3g}.

Chronos-2, H = 208, series {INTERVAL_SERIES}:

{fig("interval_collapse", "Predictive mean ± 1 std with and without the Chronos variance")}

| | Reference | Zero variance | No-Chronos baseline |
| --- | --- | --- | --- |
| 90% coverage, {INTERVAL_SERIES} | {coverage(intervals["Reference"]):.0%} | {coverage(intervals["Zero variance"]):.0%} | — |
| Median 90% coverage, 10 series | {median(reference_metrics, "amazon/chronos-2", 208, "coverage90"):.0%} | {median(zero_metrics, "amazon/chronos-2", 208, "coverage90"):.0%} | {median(zero_metrics, "acceleration_only", 0, "coverage90"):.0%} |
| Median log-likelihood | {median(reference_metrics, "amazon/chronos-2", 208, "log_lik"):.3g} | {median(zero_metrics, "amazon/chronos-2", 208, "log_lik"):.3g} | {median(zero_metrics, "acceleration_only", 0, "log_lik"):.3g} |

## A tiny variance does not fix it

A floor such as $10^{{-6}}$ gives $K_c \\approx 10^{{-6}}$, so Chronos is still fed almost exactly its own forecasts.

## What to do instead

Keep the two jobs separate:

- **Context from data.** Feed Chronos the observed residual $y_t - \\hat\\ell_t$ rather than the posterior of $c_t$. The Chronos variance can then be zero without cutting Chronos off from the observations.
- **Filter with the variance, drop it only for scoring.** Keep $\\sigma^2_c$ in the filter and remove it only when forming the reported interval. This measures what Chronos contributes to the uncertainty without changing the filter.
"""
    (EXP_DIR / "why_not_zero_variance.md").write_text(text)
    print(EXP_DIR / "why_not_zero_variance.md")


if __name__ == "__main__":
    main()
