# LLM-SSM predictive degradation vs. the re-forecast interval `H`

How much predictive accuracy does the `Auxiliary` + `WhiteNoise` state-space
model lose when Chronos-2 is queried once every `H` weeks instead of every week?

Pre-registration (question, hypothesis, data, model, metrics, leakage argument):
[`planning/README.md`](planning/README.md).

## Run it

```bash
python scripts/toy_check.py       # pipeline check on series with known behaviour
python scripts/run_experiment.py  # 10 series x 8 horizons, ~17 min on CPU
python scripts/make_figures.py    # figures + tables from the newest run
```

Results land in a fresh `results/run_<timestamp>/`; `make_figures.py` reads the
most recent one unless given a path.

## Headline

Degradation is real, monotone, and **much gentler than the horizons suggest**:
stretching `H` from 1 week to 208 weeks costs a factor of only **1.87 in RMSE**.
Across the sweep the median follows a clean power law,

```
RMSE(H) / RMSE(1)  ~  H^0.11        i.e. +7.7% RMSE per doubling of H
```

and `H = 208` still sits at **0.63 of the climatological RMSE**, so a four-year-old
forecast is a long way from useless on these series. CRPS tracks RMSE almost
exactly (`H^0.11` as well), so the penalty is not hiding in the intervals.

| | H=1 | H=3 | H=6 | H=13 | H=26 | H=52 | H=104 | H=208 |
|---|---|---|---|---|---|---|---|---|
| RMSE | 0.342 | 0.405 | 0.449 | 0.497 | 0.560 | 0.605 | 0.702 | 0.783 |
| CRPS | 0.186 | 0.217 | 0.239 | 0.262 | 0.293 | 0.318 | 0.361 | 0.433 |
| Log-likelihood | -0.350 | -0.514 | -0.630 | -0.777 | -0.814 | -0.896 | -0.948 | -1.073 |
| RMSE / climatology | 0.327 | 0.401 | 0.446 | 0.475 | 0.502 | 0.523 | 0.568 | 0.631 |
| RMSE / RMSE(H=1) | 1.000 | 1.179 | 1.293 | 1.427 | 1.509 | 1.510 | 1.678 | 1.867 |

Median over the 10 series, warmup-standardized units. Per-series numbers in
`results/<run>/rmse_skill_by_series.csv`, LaTeX in `summary_table.tex`.

Two practical readings:

- **The cheap range is wide.** Going from `H = 1` to `H = 13` costs 43% more RMSE
  while cutting Chronos-2 calls by 13x. `H = 26` to `H = 52` is nearly free
  (1.509 -> 1.510) -- doubling the interval across the annual cycle costs nothing
  measurable.
- **Where it does hurt is early.** Half the total penalty is paid by `H = 13`.

## Figures

| file | content |
|------|---------|
| `figures/skill_vs_horizon.*` | RMSE as a fraction of climatology vs `H`, median and IQR |
| `figures/degradation_vs_horizon.*` | RMSE and CRPS relative to `H=1` |
| `figures/error_by_lead.*` | error growth *inside* a block, against the block averages |
| `figures/example_ts67.*` | ts67 at `H=1` and `H=52`, observations and `mu +/- sigma` |

`error_by_lead` is the one that explains the rest: the per-lead-time error inside
a single `H=208` block rises from 0.30 to ~0.65 of climatology and the block
averages at each `H` lie right on that curve, confirming that the aggregate
penalty is just the running average of one underlying error-growth profile.

## Against the hypothesis

- **Held:** monotone degradation, concave in `log H`, well approximated by a power
  law.
- **Wrong:** no flattening at climatology. We expected saturation between `H = 26`
  and `H = 104`; instead the curve is still climbing at `H = 208` and has only
  reached 0.63 of climatology. Detrended weekly hydrological series retain enough
  annual structure that Chronos-2 keeps beating the unconditional mean even four
  years out.
- **Wrong:** the loss is not concentrated in the first few lead times. It grows
  steadily in `log(lead)` across the whole 1-208 range.

## The ts61 outlier

Nine of ten series degrade monotonically (1.69x to 3.44x at `H = 208`). **ts61
improves**: 0.69x, i.e. forecasting 208 weeks at a time beats forecasting weekly.
It is by far the smoothest series in the set (ACF(1) = 0.99, ACF(52) = 0.98), and
on it Chronos-2's long smooth seasonal extrapolation tracks the signal better
(corr 0.984, bias +0.005) than its step-by-step forecasts, which lag slightly
behind the curve (corr 0.969, bias +0.063). The structural checks in
`toy_check.py` pass for every `H`, and dropping ts61 moves the fitted exponent
from `H^0.108` to `H^0.112`, so the headline is not sensitive to it.

## Caveats

- Chronos-2 emits quantiles; the `Auxiliary` interface takes two moments, so
  forecasts are scored as Gaussian (mean, half the 15.87-84.13% spread). The same
  approximation applies at every `H`.
- The model has no level or trend component by design, so results describe the
  foundation model's own predictive capacity on a signal the benchmark already
  detrended -- not what a `LocalTrend` + `Auxiliary` model would achieve.
- `weekly_values.csv`'s detrending was fitted by the benchmark authors over whole
  series. It is identical across `H` and cannot bias the comparison, but absolute
  error levels are optimistic relative to a strictly causal pipeline.
- `results/run_20260919_183624/` is a superseded run on `weekly_values_raw.csv`;
  see the note inside it.
