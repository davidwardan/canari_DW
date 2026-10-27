# Harmonic-fit level/slope constraint on the LSTM output (toy)

## Question and hypothesis

The EWMA constraint reads the LSTM output's long-term level and slope from two
EWMAs, which attenuate but never remove the seasonal wave: for a unit annual wave the
slow memory keeps an amplitude of 0.040 and the slope summary $3.8\times10^{-4}$.
The tolerances must therefore stay near that leakage, or the constraint fights the
cycle.

This experiment replaces the two EWMAs with an exponentially weighted joint
least-squares fit of a line and $K$ seasonal harmonics to the past LSTM outputs.
Only the fitted level $\hat\alpha_t$ and slope $\hat\beta_t$ are pseudo-observed as
zero; the harmonic coefficients stay free. For a signal made of a line and the modelled
harmonics, the readings are exactly the line's level and slope, so the wave
contributes nothing.

On a toy where every component is known: does the harmonic-fit constraint remove
the wave from the readings, and does it stop the online LSTM from absorbing the drift
at least as well as the EWMA constraint, under both RSR strategies? The earlier
auxiliary-SSM experiment found that a coupled-memory constraint lost most of its
effect under `rsr_replay_smoothed`. The harmonic fit also has coupled memories, so
that strategy is the key check.

## Toy data and split

The toy series of `lstm_local_trend_rsr_ewma`: 900 weeks,
$y_t = 0.005t + \sin(2\pi t/52) + 0.5\sin(4\pi t/52+\pi/4) + v_t$,
$v_t\sim\mathcal{N}(0,0.1^2)$, noise seed 0. The true level, trend, seasonal
component and noise are saved next to every run. Training is weeks 0--207,
validation 208--311, test 312--899. Standardization uses the training split only.

## Protocol

1. Offline training of `LocalTrend()` (zero process noise), a 50-unit TAGI LSTM
   (52-step look-back, week-of-year covariate, `infer_len=52`) and white noise, with
   no constraint. Noise-grid search and early stopping on the validation split as
   in the EWMA experiment. One trained snapshot per seed, shared by all arms.
2. Frozen filtering of weeks 0--259, which ends where the first RSR window
   begins (the last validation cycle).
3. The constraint is added, and its memories are warm-started from the frozen LSTM
   posterior means of weeks 0--259: the exact weighted fit for the harmonic arm and
   the EWMA recursion for the EWMA arm. The memories are treated as known (zero
   variance). A zero start would make the harmonic readings leak 0.14 in the first
   year and still 0.075 after four years, more than the EWMA, because all poles sit at
   $\rho$.
4. Online RSR from week 260, predicting weeks 312--899, with `rsr_replay_filtered`
   and `rsr_replay_smoothed`.

Arms:

- `unconstrained`: steps 1, 2 and 4 only.
- `ewma`: `EwmaConstraint`, spans 104 and 208 weeks, $\tau_m=0.05$,
  $\tau_d=0.0005(24/52)$.
- `harmonic`: harmonic fit with $K=2$, period 52, discount span 208 weeks
  ($\rho=e^{-1/208}$, as the slow EWMA), and the same $\tau_m$, $\tau_d$ on
  $\hat\alpha_t$, $\hat\beta_t$.

The constraint is not active during offline training, unlike in the benchmark
experiments; this isolates its effect on online learning, where the drift is absorbed.

## Seeds, outputs and metrics

Seeds 1--5. Each seed's folder `results/<YYYYMMDD-HHMMSS>_toy_seed<N>/` holds the
config, selection, snapshot, and one CSV per arm and method with the observations,
predictions, all hidden states, readings and true components. The batch folder holds
the metrics, and logs go to `logs/`.

Metrics on the test split: RMSE, MAE, mean log-likelihood (LL), CRPS, 95% coverage,
mean trend, RMS trend deviation from 0.005 over the first and last 156 weeks, LSTM
drift (slope of 52-week block means), and RMS error of the level and of the LSTM
against the true baseline and seasonal component over the last 156 weeks. Readings:
mean and within-cycle ripple of the level and slope readings over the last 156 weeks.
Tables report mean ± std over seeds. Figures show seed 1 (declared in advance).

## Expected outcome

The harmonic readings carry no visible annual ripple, and the fitted harmonic
amplitudes track the LSTM's wave (about 1 and 0.5 in data units). The LSTM drift is
no larger than with the EWMA under `rsr_replay_filtered`. Whether that holds under
`rsr_replay_smoothed` is open.

## Verification

Unit tests (`scripts/test_harmonic_constraint.py`): the steady-state recursion's
weights equal the batch weighted fit; a line plus modelled harmonics, from a
warm-started fit, gives $\hat\alpha_t=a+bt$ and $\hat\beta_t=b$ to numerical precision
through `Model.forward`; the finite-window fit is exact from a quarter cycle (shorter
windows are exact in algebra but ill-conditioned: condition number $4\times10^{11}$
and error $10^{-6}$ at 7 weeks); snapshots and both RSR strategies run.

## Runs

- `results/20261027-174411_toy`: five seeds, all candidates trained without numerical
  failure. Table `mean_std_over_seeds.md`, figure index `figures.md`.

Observed: the harmonic readings carry no annual ripple, and the harmonic arm improves
on the EWMA arm on every seed and in both RSR strategies. The readings still settle
well away from zero: the LSTM starts the test split with an offset inherited from
offline training without a constraint, and $\tau_m$, $\tau_d$ were kept at the EWMA
values, which had to stay near the EWMA's leakage. Tighter tolerances are now possible
because there is no ripple to fight; they were not tried.

## Reproduce

Run from the repository root:

```sh
PY=/opt/miniconda3/envs/canari/bin/python
export PYTHONPATH=src MPLCONFIGDIR=/tmp/canari-harmonic-matplotlib
$PY -m pytest experiments/lstm_harmonic_fit_constraint/scripts -q
$PY experiments/lstm_harmonic_fit_constraint/scripts/run_toy.py --workers 5
$PY experiments/lstm_harmonic_fit_constraint/scripts/summarize_toy.py --tag RUN_TAG
```
