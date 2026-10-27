# Local-trend RSR with an auxiliary-SSM constraint on the LSTM output

## Question and hypothesis

`lstm_local_trend_rsr_ewma` keeps two EWMAs of the LSTM output in the joint state and
pseudo-observes a zero long-term mean and slope. This experiment replaces the two
EWMAs with a second state-space model run on the LSTM output: a local trend
(level $L_t$, slope $S_t$) plus a white residual that absorbs the periodic pattern.
Its steady-state Kalman estimates of $L_t$ and $S_t$ become the two joint-state
memories, and the pseudo-observations say both are zero. The pseudo-observation
noise is the auxiliary filter's own steady-state posterior covariance $\mathbf{P}$,
so the method has no hand-picked tolerances.

Does this auxiliary-SSM constraint stop the online LSTM from absorbing the drift as
well as the EWMA constraint, at a similar prediction and calibration cost?
Hypothesis: against the unconstrained model it lowers the absolute LSTM drift and the
late trend deviation under RSR, close to the EWMA arm, with a small cost in RMSE,
log-likelihood (LL) or continuous ranked probability score (CRPS).

A literal joint version, where the auxiliary level and slope are latent states linked
by $R_t = L_t + E_t$, was considered and rejected before implementation: with a white
residual it penalizes an offset, a ramp and the seasonal cycle equally (equivalent
std $\sigma_E$ at every frequency), so it cannot target drift. See `maths.md` and
`scripts/penalty_spectrum.py`.

## Data and split

Identical to `lstm_local_trend_rsr_ewma`: the ten weekly benchmark series (`ts67`,
`ts66`, `ts57`, `ts25`, `ts26`, `ts27`, `ts36`, `ts61`, `ts62`, `ts58`) from
`weekly_values.csv`, with an added slope $0.005t$ per week and period 52. Training is
weeks 0--207, validation 208--311, and evaluation starts at 312. Normalization,
baseline initialization and the auxiliary residual std use training observations only.
RSR's first window includes the final validation cycle. Evaluation is prequential.

The toy check uses the same synthetic series as the EWMA experiment: two yearly
harmonics (amplitudes 1 and 0.5), white noise 0.1, slope 0.005, 900 weeks.

## Arms and methods

- `unconstrained`: native CANARI model.
- `ewma`: CANARI's `EwmaConstraint` with the EWMA experiment's settings
  (spans 104 and 208 weeks, $\tau_m=0.05$, $\tau_d=0.0005(24/52)$).
- `ssm`: the auxiliary local-trend constraint (`scripts/ssm_constraint.py`).

All arms use the same model otherwise: `LocalTrend()` with zero process noise, a
50-unit TAGI LSTM with 52-step look-back and week-of-year covariate, `infer_len=52`,
and white-noise candidates `linspace(0.01, 0.3, 5)`. Constraints are active in offline
training, validation forecasting, frozen filtering and online replay. Each arm selects
its own epoch and noise level on the validation split (maximum 50 epochs, patience 10).
Methods: `offline` (frozen weights), `rsr_replay_filtered`, `rsr_replay_smoothed`.

Auxiliary-SSM settings, fixed before any result is inspected:

- residual std $\sigma_E$: std of the least-squares-detrended, standardized training
  observations of each series (the share the residual must absorb);
- process-noise std $\sigma_W$: the ratio $\sigma_W/\sigma_E = 7.07\times10^{-6}$
  for which the filtered level of a unit step reaches $1-e^{-1}$ after 208 weeks,
  the EWMA arm's slow span;
- pseudo-observation noise: steady-state posterior covariance
  $\mathbf{P}=\sigma_E^2\,\tilde{\mathbf{P}}$, with standard deviations
  $(0.061\,\sigma_E,~1.63\times10^{-4}\sigma_E)$ and correlation 0.71.

Seasonal leakage into the memories per unit amplitude is
$(0.031,~5.9\times10^{-5})$, about 0.7 and 0.5 times the tolerance for a pure sinusoid.
This is looser than the EWMA arm, whose $\tau_m$ is close to its own leakage. It is
reported as a known difference, not tuned.

No `src/` change is made. The constraint subclasses `EwmaConstraint`, so CANARI's
existing coupling, conditioning, smoothing, RSR replay and LSTM learning messages are
used unchanged.

## Seeds, outputs and metrics

Seeds 1--5 for every series and arm. Each series/arm/seed run has its own folder
`results/<YYYYMMDD-HHMMSS>_<mode>_<series>_<arm>_seed<N>/` with config, selection,
snapshot, and full predictions and states, plus a log in `logs/`. A batch folder holds
the manifest and the metrics; existing runs are never overwritten.

Metrics on the test split: RMSE, MAE, mean Gaussian LL, CRPS, 95% coverage, mean
trend, RMS deviation of the trend from 0.005 over the first and last 156 weeks, and
the LSTM drift (slope of 52-week block means of the LSTM posterior). Tables report
mean ± std across seeds per series, and median [IQR] across series of seed means.
Figures use seed 1; arm comparisons use `rsr_replay_filtered`.

## Verification plan

1. Unit tests: steady state solves the Riccati equation; memories equal a standalone
   Kalman filter; a ramp is tracked exactly; seasonal leakage matches the analytic
   frequency response; the single-step constrained LSTM moments match a hand-computed
   update; serialization, epoch reset and RSR run.
2. Toy with five seeds: expect a trend near 0.005, RMSE near the noise scale, and a
   smaller LSTM drift than the unconstrained arm.
3. Benchmark with the same code and settings; tables and figures from saved results.

## Reproduce

Run from the repository root. `PYTHONPATH` must point at this branch's `src`.

```sh
PY=/opt/miniconda3/envs/canari/bin/python
export PYTHONPATH=src MPLCONFIGDIR=/tmp/canari-ssm-matplotlib
$PY -m pytest experiments/lstm_local_trend_rsr_ssm/scripts -q
$PY experiments/lstm_local_trend_rsr_ssm/scripts/penalty_spectrum.py
$PY experiments/lstm_local_trend_rsr_ssm/scripts/run_experiment.py --toy --workers 8
$PY experiments/lstm_local_trend_rsr_ssm/scripts/run_experiment.py --workers 8
$PY experiments/lstm_local_trend_rsr_ssm/scripts/summarize.py --tag RUN_TAG
```
