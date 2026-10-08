# Acceleration H sweep with zero Chronos variance

## Question and hypothesis

What happens to the LocalAcceleration + Chronos Auxiliary + WhiteNoise model of
`experiments/llm_horizon_linear_acceleration` when the variance returned by
Chronos is deliberately set to exactly zero at every step, while the Chronos
mean is unchanged?

With zero auxiliary prior variance and zero cross-covariance (the auxiliary
transition is 0), the auxiliary Kalman gain is zero. Consequences that follow
from the filter equations, not hypotheses:

- the predictive variance is only the LocalAcceleration state variance plus the
  WhiteNoise variance (0.01);
- the auxiliary posterior equals the Chronos mean, so the context appended at
  each week is Chronos's own forecast. After the warmup, Chronos never sees the
  observations; the whole innovation is absorbed by level, slope and
  acceleration.

Hypotheses to measure: intervals become far narrower, coverage and
log-likelihood collapse toward the acceleration-only baseline (coverage 7.7%,
log-likelihood -62.8 in the reference run), and point accuracy depends on how
well a self-fed Chronos rollout tracks the residual, which the trend states then
compensate.

## Data and split

Identical to the reference experiment: the same ten processed weekly series,
standardized on the first 52 observations, plus the synthetic quadratic
g(t) = 0.5 a t^2 with a = 0.05/52^2. The first 52 weeks are warmup (scaling,
OLS initialization, Chronos context) and are not scored; the rest is
sequential evaluation. Nothing is fitted on the evaluation segment.

## Model and protocol

Same `core.py` (imported from the reference experiment), same six frozen
checkpoints, H grid `{1, 3, 6, 13, 26, 52, 104, 208}`, 512-step context cap,
`cross_learning=False`, float32 CPU, seed 0. The only change is that the
per-step Chronos variance passed to `Auxiliary` is `0` instead of
`max(((q_0.8413 - q_0.1587) / 2)^2, 1e-6)`. Native Chronos quantiles are still
saved for diagnosis.

## Baselines and metrics

- LocalAcceleration + WhiteNoise without Chronos (same initialization).
- The reference run `llm_horizon_linear_acceleration/results/run_20260925T103556_080836Z`
  (non-zero Chronos variance), compared per series, model and H.

Metrics: RMSE, MAE, Gaussian log-likelihood, CRPS, 90% coverage and width,
slope/acceleration MAE against the injected coefficients. The variance is total
predictive variance; no epistemic/aleatoric split is claimed.

## Verification

Unit tests check that the zero-variance wrapper keeps means and quantiles and
returns exact zeros, that the auxiliary posterior equals the Chronos mean, that
the context equals the served means, and that the predictive variance equals the
trend-plus-white-noise variance. A two-series smoke run precedes the full run.
