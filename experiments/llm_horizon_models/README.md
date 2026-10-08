# LLM-SSM degradation vs. `H`, across Chronos models

The `llm_horizon_degradation` sweep run unchanged: same 10 series, SSM, split,
metrics and code, imported from its `common.py`. The only thing that changes
is the foundation model behind `Auxiliary`: Chronos-2, Chronos-2 small, and
Chronos-Bolt tiny/mini/small/base. Pre-registration: [`planning/README.md`](planning/README.md).

## Run it

```bash
python scripts/model_check.py     # pipeline checks + per-model toy checks
python scripts/run_experiment.py  # 6 models x 10 series x 8 horizons, ~50 min on CPU
python scripts/make_figures.py    # figures + tables from the newest run
```

The Chronos-2 rows reproduce
`llm_horizon_degradation/results/run_20260919_185524` to a maximum absolute
difference of 2.2e-16.

## Results (`results/run_20260924_142849`)

Median over the 10 series, warmup-standardized units.

**RMSE**

| model | H=1 | 3 | 6 | 13 | 26 | 52 | 104 | 208 |
|---|---|---|---|---|---|---|---|---|
| Chronos-2 | **0.342** | **0.405** | **0.449** | **0.497** | 0.560 | 0.605 | 0.702 | 0.783 |
| Chronos-2 small | 0.353 | 0.407 | 0.451 | 0.499 | **0.542** | 0.584 | 0.641 | 0.775 |
| Bolt tiny | 0.384 | 0.445 | 0.493 | 0.535 | 0.580 | 0.557 | 0.604 | 0.729 |
| Bolt mini | 0.381 | 0.442 | 0.488 | 0.527 | 0.566 | **0.538** | **0.575** | 0.720 |
| Bolt small | 0.373 | 0.445 | 0.496 | 0.541 | 0.586 | 0.554 | 0.591 | 0.734 |
| Bolt base | 0.365 | 0.432 | 0.481 | 0.522 | 0.564 | 0.548 | 0.582 | **0.685** |

**Mean log-likelihood**

| model | H=1 | 3 | 6 | 13 | 26 | 52 | 104 | 208 |
|---|---|---|---|---|---|---|---|---|
| Chronos-2 | -0.350 | -0.514 | **-0.630** | -0.777 | -0.814 | -0.896 | **-0.948** | **-1.073** |
| Chronos-2 small | **-0.342** | **-0.491** | -0.640 | **-0.733** | **-0.801** | **-0.879** | -1.010 | -1.150 |
| Bolt tiny | -0.457 | -0.627 | -0.772 | -0.894 | -0.925 | -0.901 | -0.994 | -1.245 |
| Bolt mini | -0.455 | -0.639 | -0.782 | -0.927 | -0.961 | -0.960 | -1.030 | -1.178 |
| Bolt small | -0.430 | -0.654 | -0.811 | -0.937 | -0.950 | -0.964 | -1.050 | -1.255 |
| Bolt base | -0.422 | -0.626 | -0.767 | -0.907 | -0.934 | -0.968 | -1.034 | -1.141 |

LaTeX version: `results/<run>/summary_table.tex`. Per-series values are in
`metrics.csv`.

## Findings

- **Short `H` (1-13): Chronos-2 wins on both metrics, as hypothesized.** Bolt
  base, the best Bolt model, is 7% worse in RMSE at `H = 1`. Chronos-2 beats it
  on 9 of 10 series at every `H <= 13`.
- **Long `H` (>= 52): the RMSE ranking flips on the medians, but only on the
  medians.** Every Bolt model has a lower median RMSE than Chronos-2, but
  paired by series it is a coin flip: Bolt base is better on 5, 6 and 5 of
  10 series at `H = 52, 104, 208`. The model has no clear RMSE effect at long
  `H`.
- **Log-likelihood separates the families at every `H`.** Chronos-2 beats
  Bolt base on 8 of 10 series at `H = 208`. At that horizon Bolt base's mean
  predictive sigma is 0.59, against 0.87 for Chronos-2, while its point
  forecast is equally good (correlation with observations 0.77 vs 0.79). Bolt
  is overconfident, as hypothesized, and at every `H`, not just beyond its
  64-step native output.
- **Chronos-2 small tracks Chronos-2 closely throughout,** with a
  slightly better LL for `H <= 52` and a worse one for `H >= 104`. At a quarter
  of the parameters and about 3x faster, it is a cheap substitute here.
- **Size barely matters within Bolt.** Tiny (9M) to base (205M) spans 0.02-0.05
  in RMSE at every `H`, not monotonically.

## Against the toy checks

Only Chronos-2 passes the 208-week sinusoid check (`results/model_check_output.txt`):
from a 52-week context, the other models lose the annual cycle (RMSE/clim
0.71-0.83, against 0.38). This did not carry over into worse RMSE on the real
series at long `H`. There, a forecast that relaxes towards the mean is about as
good as one that keeps extrapolating the seasonal cycle.

## Caveats

- All inherited from `llm_horizon_degradation`: Gaussian scoring of a quantile
  forecast, no level component, and benchmark-side detrending.
- For Bolt, the 15.87/84.13% quantiles are interpolated from its deciles, and
  `H > 64` is an autoregressive roll-out of its median.
- Chronos-T5 is not included (see the planning README).
