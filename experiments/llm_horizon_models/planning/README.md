# Predictive degradation vs. `H`, across Chronos foundation models

## Research question

`experiments/llm_horizon_degradation` measured how much an `Auxiliary` +
`WhiteNoise` SSM loses when Chronos-2 is re-queried only every `H` weeks. This
experiment repeats that sweep **unchanged** and swaps only the foundation model:

**Does the degradation curve (RMSE and log-likelihood vs. `H`) depend on which
Chronos model backs the `Auxiliary` component, and does model size or family
matter more at short or at long `H`?**

## Hypothesis

- At every `H`, Chronos-2 beats the Chronos-Bolt models, and within a family
  larger is better. The gap should be widest at small `H`, where the model has
  the most recent context and model quality has the most room to show.
- At large `H`, all models fall back towards the same seasonal extrapolation,
  so the curves converge in RMSE.
- Log-likelihood separates the models more than RMSE does: Bolt was trained
  with a fixed 64-step output and extends past it autoregressively on its own
  median, so we expect its intervals to be overconfident for `H > 64`.

## Models

| id | family | params |
|----|--------|--------|
| `amazon/chronos-2` | Chronos-2 | 120M (reference, same as the original run) |
| `autogluon/chronos-2-small` | Chronos-2 | 28M |
| `amazon/chronos-bolt-tiny` | Chronos-Bolt | 9M |
| `amazon/chronos-bolt-mini` | Chronos-Bolt | 21M |
| `amazon/chronos-bolt-small` | Chronos-Bolt | 48M |
| `amazon/chronos-bolt-base` | Chronos-Bolt | 205M |

Chronos-T5 is left out: it forecasts by autoregressive token sampling, which
would make the `H = 1` runs orders of magnitude slower and turn the quantiles
into sample estimates, a different kind of approximation from the other two
families.

Every model goes through the same predictor
(`llm_horizon_degradation/scripts/common.py::make_predict_fn`): the mean
forecast is the location, and half the 15.87-84.13% quantile spread is the
scale. For Bolt, these quantile levels are interpolated from its decile head.

## Data, split, SSM, leakage

Identical to `llm_horizon_degradation/planning/README.md` and imported from
its `common.py`, not copied: the same 10 weekly series, a 52-week warmup used
as the initial context and for standardization, the evaluation segment
`[52, n)`, `sigma_v = 0.1`, and `H in {1, 3, 6, 13, 26, 52, 104, 208}`. No
parameter is fitted or tuned on the evaluation segment, and all models are
scored on the same time steps, so the comparison is paired across both `H` and
model.

## Baselines and metrics

- Climatology (warmup mean and std) as the floor.
- Chronos-2 as the reference model.
- **RMSE** and **mean Gaussian log-likelihood** of the one-step-ahead
  predictive distribution, in warmup-standardized units, median over the
  10 series. MAE and CRPS are still saved in `metrics.csv`.

## Reproducibility

All models are frozen and their quantile heads are deterministic, so a single
seed suffices (it is set and recorded). The Chronos-2 rows must reproduce
`llm_horizon_degradation/results/run_20260919_185524/metrics.csv`; the run
script checks this.

## Pipeline check

`scripts/model_check.py` runs the original AR(1) and sinusoid toy checks
against every model before the sweep.

## Expected outcome

A fan of curves ordered by model quality at `H = 1` that narrows in RMSE as
`H` grows, with the Bolt log-likelihood dropping faster than Chronos-2's
beyond `H = 64`.
