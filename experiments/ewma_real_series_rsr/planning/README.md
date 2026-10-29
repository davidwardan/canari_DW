# EWMA constraint on real series with an added trend: frozen vs RSR

## Research question and hypothesis

On five real weekly series with a known added trend, does the EWMA constraint keep the
baseline estimate correct when the LSTM keeps learning online with RSR, and how do the
tolerances change this?

**Hypotheses.**

1. Online RSR without constraint lets the LSTM take part of the trend, so the
   level and trend RMSE grow compared with the frozen model.
2. RSR with the EWMA constraint keeps the level and trend RMSE close to, or below, the
   frozen model.
3. Tighter tolerances lower the trend error until they start fighting the seasonal
   pattern; below that, the predictive scores and possibly the level error get worse.

The toy experiment `ewma_drift_attribution` found a useful range at
$\kappa \le \tfrac12$ (LSTM share of the slope from 12% to under 3%, CRPS cost under 2%).

## Data and split

Five series of `data/hq_benchmark_data/weekly/weekly_values.csv` (already detrended and
standardized by the provider): `ts67`, `ts66`, `ts57`, `ts25`, `ts26`, the first five of
`lstm_local_trend_rsr_ewma`. A trend $\delta t$, $\delta = 0.005$ per week, is added from
the first sample:

$$y_t = y^{\text{det}}_t + \delta t.$$

Train: weeks 0–207 (4 cycles); validation: 208–311 (2 cycles); test: 312 to the end
(662 to 766 weeks). Standardization uses training data only. Missing values stay
missing.

**True baseline.** The detrended part has no trend by construction; its level is taken
as its training mean $m = \frac{1}{n}\sum_{t<208} y^{\text{det}}_t$ (between −0.015 and
0.012 here). The reference baseline is therefore

$$b_t = m + \delta t,\qquad \text{trend } \delta .$$

The provider detrended each full series with a smooth baseline, so a real series can
still wander slowly. Such wander counts as level error for every arm alike.

## Correct initial level and trend

Every training epoch starts from the correct baseline: level $b_{-1}$ and trend $\delta$
(the state before the first observation), in normalized units, with canari's default
initial variances ($10^{-4}$ level, $10^{-6}$ trend) and no cross-covariance. canari
normally restarts each epoch from the smoothed trend of the previous epoch; this is
overridden. The selected snapshot carries these initial states, so the frozen and RSR
runs also start from them at week 0.

## Model

As in `lstm_local_trend_rsr_ewma`: `LocalTrend(std_error=0)`, a 50-unit LSTM (look-back
52, `infer_len` 52, week-of-year covariate), `WhiteNoise(sigma_v)` with
$\sigma_v \in \{0.01, 0.0825, 0.155, 0.2275, 0.3\}$ selected on validation
log-likelihood together with the epoch (at most 50 epochs, patience 10). EWMA spans 104
and 208 weeks. RSR: `replay_smoothed`, window 52.

## Tolerances

$\tau = \kappa\lambda$ with $\kappa \in \{4, 2, 1, \tfrac12, \tfrac14, \tfrac18\}$, where
$\lambda_m$, $\lambda_d$ are the peak mean and slope readings the EWMAs give for the
average seasonal cycle of the training split
($y_t - b_t$ averaged over the 4 training cycles). Only training data are used.
Each tolerance is used for both offline training and the test.

## Arms (per series and seed)

| Arm | Offline training | Test |
|---|---|---|
| `frozen` | no constraint | frozen LSTM |
| `frozen_ewma_k{κ}` | constraint at κ | frozen LSTM, constraint at κ |
| `rsr` | no constraint | RSR `replay_smoothed` |
| `rsr_ewma_k{κ}` | constraint at κ | RSR `replay_smoothed`, constraint at κ |

5 series × 5 seeds (1–5) × 7 snapshots = 175 trainings; 5 × 5 × 14 = 350 test runs.
Seeds fix the LSTM initialization.

## Metrics

Main: **level RMSE** and **trend RMSE** on the test split, from the filtered posterior
means, in data units (see `metrics.md`). Secondary: one-step predictive RMSE, log
likelihood, CRPS and 95% coverage. Mean ± std over seeds per series, and the mean over
series of the seed means.

## Follow-ups (added 2026-10-28, after the first real run)

**Longer inferred start (`--infer-len`).** At week 0 the LSTM starts from a look-back that
canari's LSTM smoother infers before the data (`3 × infer_len` weeks; 155 weeks for the
default 52). The first year of LSTM output was poor, so the user asked for a longer inferred
stretch. Toy pilot (5 seeds, `frozen`, `rsr`, `frozen_ewma_k1`, `rsr_ewma_k1`) with
`infer_len` 52, 104, 208; `startup.py` reports the look-back error against the true past,
the year-0 LSTM error, the largest trend deviation in training and the level error at the
test start. The real series are then re-run with `--infer-len 208` (tag suffix `_infer208`).

**Level-offset test (`offset.py`, in `results/level_offset/` and `figures/level_offset/`).**
The snapshots of a finished run are re-run over the whole series with the week-0 level
shifted by +0.5 (data units) and given a prior standard deviation of 0.5, so the filter can
correct it. Each run is paired with the same arm, snapshot and seed without offset; since
level + LSTM + noise = y exactly, the share of the offset the level keeps is the share the
LSTM or the noise takes instead. RSR only learns from week 260, so the week-0 offset is
first met by the frozen pass in every arm.

## Verification before the real run

1. Focused tests: the baseline reset before every epoch, the conversion of the true
   baseline to normalized units, the leakage estimate, NaN-safe scores.
2. Toy series (`--series toy`: the toy pattern plus noise, no real data) through the
   whole pipeline with 5 seeds; expected: small level and trend RMSE for the frozen
   model.
3. The five real series.

## Reproduce

The EWMA component lives in the uncommitted `src/` of the `feature/online-lstm` working
copy; each run records hashes of the source files it imported. Data are read from the
main checkout.

```sh
SRC=/Users/davidwardan/canari_DW/src
PYTHONPATH=$SRC /opt/miniconda3/envs/canari/bin/python -m pytest experiments/ewma_real_series_rsr/scripts/test_experiment.py -q
PYTHONPATH=$SRC /opt/miniconda3/envs/canari/bin/python experiments/ewma_real_series_rsr/scripts/run_experiment.py --series toy
PYTHONPATH=$SRC /opt/miniconda3/envs/canari/bin/python experiments/ewma_real_series_rsr/scripts/run_experiment.py
PYTHONPATH=$SRC /opt/miniconda3/envs/canari/bin/python experiments/ewma_real_series_rsr/scripts/summarize.py --tag RUN_TAG
PYTHONPATH=$SRC /opt/miniconda3/envs/canari/bin/python experiments/ewma_real_series_rsr/scripts/plot_states.py --tag RUN_TAG
# follow-ups
PYTHONPATH=$SRC /opt/miniconda3/envs/canari/bin/python experiments/ewma_real_series_rsr/scripts/run_experiment.py --series toy --infer-len 208 --arms frozen rsr frozen_ewma_k1 rsr_ewma_k1 --tag-suffix _infer208
PYTHONPATH=$SRC /opt/miniconda3/envs/canari/bin/python experiments/ewma_real_series_rsr/scripts/startup.py --tags TAG [TAG ...]
PYTHONPATH=$SRC /opt/miniconda3/envs/canari/bin/python experiments/ewma_real_series_rsr/scripts/offset.py --tag RUN_TAG
PYTHONPATH=$SRC /opt/miniconda3/envs/canari/bin/python experiments/ewma_real_series_rsr/scripts/plot_states.py --tag level_offset/OFFSET_TAG
```
