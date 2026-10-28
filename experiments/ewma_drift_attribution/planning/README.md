# Does the EWMA constraint keep a slope out of the LSTM? (toy)

## Research question and hypothesis

A model `LocalTrend` + `LstmNetwork` + `WhiteNoise` has two components that can
both explain a linear ramp: the trend, and an LSTM that keeps learning online with
Recurrent Smoothing and Replay (RSR). Without extra information the split of a ramp
between them is not identifiable. The `EwmaConstraint` component softly observes the
slow mean and the slope of the LSTM output as zero, with tolerances $\tau_m$ and
$\tau_d$.

**Question.** With a rigid baseline (process noise $\sigma_w = 0$) and RSR
`replay_smoothed`, how much of a slope present in the whole record ends up in the
LSTM, and how does this depend on the tolerances?

**Hypotheses.**

1. The LSTM's share of the slope decreases as the tolerances decrease, and the
   trend's share approaches 1.
2. Below the seasonal leakage of the EWMA readings (tolerance ratio $\kappa < 1$)
   the constraint starts fighting the seasonal cycle, which costs predictive
   accuracy on a series without any slope.
3. A zero-mean change of the seasonal pattern is still learned online, because the
   constraint only acts on the low-frequency part of the LSTM output.

## Fixed design choices (agreed with the user, 2026-10-28)

- Baseline `LocalTrend(std_error=0)`: no process noise. No switching Kalman filter.
  The baseline can only carry a slope that is present from the start of the record,
  so every scenario keeps the slope constant over the whole record.
- RSR strategy `replay_smoothed`, window 52. It gave the lowest LSTM drift with EWMA
  in `lstm_local_trend_rsr_ewma`; all four strategies tie without the constraint.
- One offline snapshot per scenario family and seed, trained with the constraint at
  $\kappa = 1$. The tolerance is changed only for the online RSR phase.
- Toy only. Benchmark series come later.

## Data and split

Weekly toy series, 900 steps, period $P = 52$:

$$y_t = \delta\, t + p_t + v_t,\qquad
p_t = \sin\!\left(\tfrac{2\pi t}{52}\right) + 0.5\,\sin\!\left(\tfrac{4\pi t}{52} + \tfrac{\pi}{4}\right),\qquad
v_t \sim \mathcal{N}(0, 0.1^2).$$

Train 0–207 (4 cycles), validation 208–311 (2 cycles), test 312–899 (588 steps,
11 complete cycles). The noise realization uses the run seed, so seed $s$ fixes both
the LSTM initialization and the noise, and every scenario and arm with seed $s$
sees the same noise. Standardization uses the training split only.

| Scenario | $\delta$ (data units / week) | Seasonal pattern | Trend prior at $t=0$ | Snapshot |
|---|---|---|---|---|
| **A** null | 0 | $p_t$ | from `auto_initialize_baseline_states` | A |
| **B** slope | 0.005 | $p_t$ | from `auto_initialize_baseline_states` | B |
| **C** slope, unknown trend | 0.005 | $p_t$ | mean 0, std 0.01 (normalized, $\approx 1.6\,\delta$) | C |
| **D** slope + phase shift | 0.005 | $p_t$, then $p_{t+13}$ from $t_c = 416$ | as B | B |

D is identical to B up to $t_c = 416$ (test start + 2 cycles), so the two share
the offline snapshot and the online run is identical before the change. A
quarter-cycle phase shift keeps the pattern's amplitude, hence its EWMA leakage,
unchanged; only the mapping from week of year to value changes.

## Model and training

`LocalTrend(0)`, `LstmNetwork(look_back_len=52, num_features=2, infer_len=52,
num_hidden_unit=50, num_layer=1, manual_seed=seed, smoother=True)`,
`WhiteNoise(sigma_v)`, `EwmaConstraint(source="lstm", fast_span=104,
slow_span=208)`. Covariate: week of year.

Offline training as in `lstm_local_trend_rsr_ewma`: grid
$\sigma_v \in \{0.01, 0.0825, 0.155, 0.2275, 0.3\}$, at most 50 epochs, patience 10,
early stopping on validation log-likelihood (first 5 epochs skipped), best
$(\sigma_v, \text{epoch})$ by validation log-likelihood. Test data are never used
for training, selection, normalization or tolerance choice.

## Tolerances

Tolerances are expressed relative to the seasonal leakage, $\tau = \kappa\,
\lambda$, where $\lambda_m$ and $\lambda_d$ are the peak absolute mean and slope
readings that the EWMA memories produce from the true pattern $p_t$ alone
(normalized units; see `maths.md`). For the toy, $\lambda_m \approx 0.061$ and
$\lambda_d \approx 5.9\times10^{-4}$ per week. The previous experiment's
tolerances (0.05, $2.3\times10^{-4}$) correspond to $\kappa \approx 0.82$ and $0.39$.
Using the true pattern is an oracle that only exists on the toy; it fixes the
scale of the sweep and does not use test observations.

## Arms (per scenario and seed)

| Arm | Offline snapshot | Online phase |
|---|---|---|
| `none` | trained without constraint | RSR, no constraint component |
| `frozen` | trained at $\kappa=1$ | frozen LSTM weights, constraint at $\kappa=1$ |
| `rsr_kinf` | trained at $\kappa=1$ | RSR, both tolerances `None` (memories kept, no constraint) |
| `rsr_k8` … `rsr_k0.25` | trained at $\kappa=1$ | RSR, both tolerances at $\kappa \in \{8,4,2,1,\tfrac12,\tfrac14\}$ |
| `rsr_mean_k1` | trained at $\kappa=1$ | RSR, $\tau_m$ at $\kappa=1$, $\tau_d$ = `None` |
| `rsr_slope_k1` | trained at $\kappa=1$ | RSR, $\tau_m$ = `None`, $\tau_d$ at $\kappa=1$ |

`none` is the full pipeline without any constraint. It was added after the design
discussion: if the $\kappa=1$ snapshot already gives the trend most of the slope,
`rsr_kinf` alone cannot show how much an unconstrained LSTM would take.

Runs: 4 scenarios $\times$ 11 arms $\times$ 5 seeds = 220 evaluations, plus
2 snapshot types $\times$ 3 families (A, B, C) $\times$ 5 seeds = 30 trainings.
Seeds 1–5.

## Metrics

All metrics, their formulas and how to read them are in `metrics.md`. In short:
slope shares $a_{\mathtt{LSTM}}$, $a_{\mathtt{L}}$, $a_{\mathtt{W}}$, $a_{\mathtt{T}}$
(B, C, D); LSTM drift (all); LSTM pattern error; RMSE, MAE, log-likelihood, CRPS,
95% coverage; adaptation kept $\eta$ (D vs B); reading ratio (diagnostic). Each is
reported as mean ± std over the 5 seeds.

## Pre-registered criteria

Evaluated on seed means at $\kappa = 1$ (the training reference), and reported for
every $\kappa$:

1. **Drift kept out (B, C).** $a_{\mathtt{LSTM}}(\kappa) \le \tfrac14\,
   a_{\mathtt{LSTM}}(\texttt{none})$ and $|1-a_{\mathtt{T}}(\kappa)| <
   |1-a_{\mathtt{T}}(\texttt{none})|$. Not evaluable if
   $a_{\mathtt{LSTM}}(\texttt{none}) < 0.05$ (nothing to prevent).
2. **Low cost (A).** CRPS at most 5% above `rsr_kinf`; 95% coverage within
   0.02 of `rsr_kinf`.
3. **Pattern change still learned (D).** $\eta(\kappa) \ge 0.8$. Not evaluable if
   RSR does not beat the frozen model after the change.
4. **Monotone.** Spearman correlation between $\kappa$ and seed-mean
   $a_{\mathtt{LSTM}}$ over the `rsr_k*` arms $\ge 0.8$ in B and C.

The constraint is judged to **work** at a tolerance if criteria 1–3 all hold
there, and to **not work** if no tolerance in the sweep satisfies 1 and 3 together.

## Expected outcome

From earlier runs: the unconstrained pipeline leaves about 17% of the slope in the
LSTM on this toy, and an EWMA at about leak level reduces it several-fold. We expect
$a_{\mathtt{LSTM}}$ to fall with $\kappa$, a CRPS cost in A at
$\kappa \le \tfrac12$, and $\eta \approx 1$ for $\kappa \ge 1$.

## Verification before the full run

1. Focused tests (`scripts/test_experiment.py`): block-slope estimator exact on
   ramp + periodic signals; shares add up; analytic leakage of one harmonic;
   scenario pairing; tolerance swap in the snapshot; posterior components add up to
   $y_t$ in canari. The equivalence of `None` tolerances with a model without the
   component is covered by `test/test_ewma.py`.
2. Smoke run with 1 seed and 3 epochs.
3. Full run.

## Reproduce

The EWMA component lives in the uncommitted `src/` of the `feature/online-lstm`
working copy, so runs use that source tree. Each run records SHA-256 hashes of the
source files it imported.

```sh
SRC=/Users/davidwardan/canari_DW/src
PYTHONPATH=$SRC /opt/miniconda3/envs/canari/bin/python -m pytest experiments/ewma_drift_attribution/scripts/test_experiment.py -q
PYTHONPATH=$SRC /opt/miniconda3/envs/canari/bin/python experiments/ewma_drift_attribution/scripts/run_experiment.py --workers 6
PYTHONPATH=$SRC /opt/miniconda3/envs/canari/bin/python experiments/ewma_drift_attribution/scripts/summarize.py --tag RUN_TAG
```
