# Predictive degradation of an LLM-SSM as the re-forecast interval `H` grows

## Research question

An `Auxiliary` component queries a pretrained time-series foundation model
(Chronos-2) once every `H` filtered time steps, and the state-space model
consumes that block of `H` forecasts before asking again. Inside a block the
foundation model is blind: its context stops growing from its own point of view,
so the step at offset `j` of a block is really a `(j+1)`-step-ahead forecast.

**How much predictive accuracy is lost as `H` grows from 1 to 208 weeks?**

## Hypothesis

Accuracy degrades monotonically in `H`, but not linearly: the loss is
concentrated in the first few lead times of a block, so the aggregate penalty
grows roughly with `log H` until `H` reaches the scale at which the series is no
longer predictable, after which the curve flattens at the "climatological"
error level (predicting the unconditional mean). For weekly hydrological series
we expect the flattening to begin somewhere between `H = 26` and `H = 104`.

## Data

`data/hq_benchmark_data/weekly/weekly_values.csv` (+ `weekly_datetimes.csv`), the
*processed* benchmark files: 101 real weekly series that have been detrended and
standardized. They are not an affine rescaling of the `_raw` files -- fitting
`processed ~ a * raw + b` over sliding windows gives a near-constant `a` but an
intercept that drifts across the series (ts67: -0.65 to -7.72), i.e. a slowly
varying baseline has been subtracted. Working on the detrended version is what
makes a level-free model sensible: the auxiliary component is then asked to
explain a roughly stationary signal rather than to carry a multi-decade trend.

**Series selection** (deterministic, see `scripts/common.py::select_series`):
among series whose observed span is >= 500 weeks with <= 5 internal missing
values, take them longest-first and keep a series only if its absolute Pearson
correlation with every already-selected series is < 0.9. This yields exactly 10
series. On the raw files the filter would bite hard -- shared trends push many
pairs to |r| = 0.99 -- but after detrending the 10 longest series are already
mutually distinguishable (max |r| = 0.851), so the rule keeps all of them.

| series | weeks | first | last | missing |
|--------|-------|-------|------|---------|
| ts67 | 1078 | 2005-12-04 | 2026-07-26 | 3 |
| ts66 | 1061 | 2006-04-02 | 2026-07-26 | 2 |
| ts57 |  995 | 2007-07-15 | 2026-08-02 | 2 |
| ts25 |  974 | 2007-12-02 | 2026-07-26 | 0 |
| ts26 |  974 | 2007-12-02 | 2026-07-26 | 0 |
| ts27 |  974 | 2007-12-02 | 2026-07-26 | 0 |
| ts36 |  969 | 2008-01-06 | 2026-07-26 | 0 |
| ts61 |  969 | 2008-01-06 | 2026-07-26 | 0 |
| ts62 |  969 | 2008-01-06 | 2026-07-26 | 0 |
| ts58 |  953 | 2008-05-04 | 2026-08-02 | 2 |

Every selected series has at least 901 evaluation steps, so even `H = 208` gets
more than four forecast blocks.

**Split.** There is no training split: every component is frozen (Chronos-2 is
pretrained, the SSM has no learned parameters). The first `W = 52` weeks of each
series are the *warmup*: they are handed to the `Auxiliary` component directly as
its initial context and are never filtered and never scored. Weeks `[52, n)` are
the *evaluation segment*; every `H` is scored on exactly the same time steps of
the same series, so the comparison across `H` is paired.

**Leakage.** Each series is re-standardized with the mean and standard deviation
of its own 52-week warmup. This is not redundant with the file's own global
standardization and it does not reintroduce it: re-standardizing is invariant to
any affine map of the input, so the file's global constants cancel exactly and
what reaches the model is the detrended series scaled by warmup statistics only.
Nothing downstream of week 52 touches any hyper-parameter: `sigma_v` is a fixed
constant, `H` is the swept variable, and Chronos-2 is used exactly as released.

One caveat is inherited rather than introduced: the detrending baked into
`weekly_values.csv` was fitted by the benchmark authors, presumably over each
whole series, so early values may carry some information about later ones. It is
identical for every `H`, so it cannot bias the comparison this experiment is
about, but it does mean the absolute error levels are optimistic relative to a
strictly causal pipeline.

**Missing values.** Up to 3 gaps per series are left as `NaN`. Canari's update
step maps a `NaN` observation to a zero state correction, so the filter coasts on
the prior; those time steps are excluded from every metric.

## Model

```
y_t = auxiliary_t + white-noise_t
```

- `Auxiliary(predict_fn=chronos2, horizon=H)` -- the swept component. Its context
  starts as the 52 warmup observations and grows by one posterior auxiliary mean
  per filtered step.
- `WhiteNoise(std_error=sigma_v)` with `sigma_v = 0.1` in warmup-standardized
  units, fixed for every series and every `H`.

No level/trend component: the foundation model is asked to account for the whole
signal, which is what isolates its predictive capacity.

Chronos-2 returns quantiles; the `Auxiliary` interface needs two moments, so the
mean forecast supplies the location and half the 15.87%-84.13% quantile spread
supplies the scale. Forecasts are therefore scored as Gaussian, which understates
any skew Chronos-2 expresses -- the same approximation at every `H`, so it does
not bias the comparison.

## Sweep

`H in {1, 3, 6, 13, 26, 52, 104, 208}` x 10 series = 80 filter runs. With
`H = 1`, Chronos-2 is called once per week; with `H = 208`, roughly four times
over the whole series.

## Baselines

- **`H = 1`** is the reference: the best this architecture can do, one Chronos-2
  call per step. Every other `H` is reported both absolutely and as a ratio to it.
- **Climatology** (predict the warmup mean, with the warmup standard deviation)
  is the floor: in standardized units its RMSE is ~1. A curve reaching it means
  the foundation model has stopped contributing anything.

## Metrics

Computed on the evaluation segment in warmup-standardized units (so they are
comparable across series), on the one-step-ahead predictive distribution that the
filter emits at every step:

- **RMSE**, **MAE** -- point accuracy.
- **Mean Gaussian log-likelihood**, **CRPS** -- distribution quality; they are
  what will show whether Chronos-2 widens its intervals honestly as the lead time
  inside a block grows, or stays overconfident.

Each is reported aggregated over the evaluation segment, and broken down by lead
time within the block (offset `j+1`), which is what explains the aggregate curve.

## Reproducibility

`scripts/run_experiment.py` writes a timestamped directory under `results/`
containing `config.json` (all settings, package versions, seed), one raw
per-step CSV per (series, `H`), and `metrics.csv`. Nothing is overwritten.
Chronos-2's quantile head is deterministic, so a single seed suffices; the seed
is still set and recorded.

## Expected outcome

`RMSE(H)` increasing and concave in `log H`, flattening near the climatological
level for the largest `H`. The per-lead-time breakdown should show the error
saturating within the first ~10-25 weeks of a block, which would mean the cost of
a large `H` is paid almost entirely in its first quarter and that raising `H`
beyond the series' predictability horizon is nearly free.

## Pipeline check

`scripts/toy_check.py` runs the same machinery on a synthetic series whose
behaviour is known (a pure 52-week sinusoid plus small noise) and asserts:
Chronos-2 is queried exactly `ceil(N/H)` times, the context grows by one value
per filtered step, `H = 1` beats every larger `H`, and a deterministic constant
predictor reproduces the closed-form Kalman prediction. Run it before the sweep.
