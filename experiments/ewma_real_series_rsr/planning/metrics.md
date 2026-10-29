# Metrics: what each one measures and how to read it

Formulas are in `maths.md`. Everything is on the test split (week 312 to the end),
in data units (the units of `weekly_values.csv`), mean ± std over seeds 1–5.

## Main metrics

| Metric | Column | Question | Ideal |
|---|---|---|---|
| Level RMSE | `level_rmse` | How far is the filtered level from the true baseline $b_t = m + 0.005\,t$? | 0 |
| Trend RMSE | `trend_rmse` | How far is the filtered trend from the true 0.005 per week? | 0 |

**How they are computed.** At every test week, take the filtered (causal) posterior
mean of the level $z^{\mathtt{L}}_t$ and of the trend $z^{\mathtt{T}}_t$, subtract the truth, and take
the root mean square over the test split. Weeks with a missing observation still count;
the states are predicted there.

**How to read them.**

- The model fits $y_t = z^{\mathtt{L}}_t + z^{(\mathtt{O})}_t + z^{\mathtt{V}}_t$ exactly at every step, so
  whatever the level misses is carried by the LSTM (or the noise). A growing level RMSE
  means the LSTM is taking part of the trend.
- Trend RMSE is in data units per week. For scale: $\delta = 0.005$, so a trend RMSE of
  0.001 is a 20% error on the slope.
- The level error accumulates the trend error: a trend off by $e$ per week for 600
  weeks moves the level by $600\,e$ unless the observations pull it back.
- The true level assumes the detrended series has no level change of its own. If a real
  series still wanders slowly, every arm pays the same level error for it; compare
  arms, not absolute values.

## Secondary metrics (predictive quality)

From the one-step-ahead forecast $\mathcal{N}(\mu_t, \sigma_t^2)$ made before $y_t$ is used,
on observed weeks only.

| Metric | Column | Ideal |
|---|---|---|
| RMSE of the forecast | `rmse` | small |
| Mean log-likelihood | `ll` | large |
| CRPS | `crps` | small |
| 95% coverage | `coverage_95` | 0.95 |

They measure the cost of the constraint. A model that lets the LSTM take the trend
can forecast one step ahead well while getting the level wrong; that is why level and
trend RMSE are the main metrics.

## Arms

- `frozen`: trained without constraint, LSTM frozen in the test.
- `frozen_ewma_k{κ}`: trained with the constraint at tolerance ratio κ, frozen in the
  test with the constraint still active.
- `rsr`: trained without constraint, LSTM keeps learning with RSR `replay_smoothed`.
- `rsr_ewma_k{κ}`: trained and run with RSR at the same κ.

κ is the tolerance relative to the seasonal leakage of the training split: κ = 1 means
τ equals the largest reading a pure seasonal output would give; smaller κ is tighter.
