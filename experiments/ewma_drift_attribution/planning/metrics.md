# Metrics: what each one measures and how to read it

Derivations are in `maths.md`; this file explains the metrics in words.

## Where the numbers come from

- **Window.** Every metric is computed on the test split (weeks 312–899) unless
  stated otherwise. Block-based metrics use its 11 complete 52-week cycles.
- **Hidden states.** We use the *filtered* posterior mean of each component: what the
  model believes at time $t$ after seeing $y_t$, as stored when the step first arrives
  in RSR. No smoothing, no future data.
- **Predictions.** Predictive scores use the one-step-ahead forecast made *before*
  $y_t$ is used.
- **Units.** Everything is converted back to data units (the units of $y_t$), so
  $\delta = 0.005$ per week in scenarios B, C, D.
- **Seeds.** Each metric is computed per run, then reported as mean ± standard
  deviation over seeds 1–5.

The model splits every observation into exactly three parts:

$$y_t = \underbrace{\ell_t}_{\text{level}} + \underbrace{u_t}_{\text{LSTM}} + \underbrace{w_t}_{\text{white noise}}.$$

This holds *exactly* at every step, because canari has no separate observation noise
(the white noise is a state). The trend $x^{\mathtt{T}}$ is not in this sum; it is the
slope the level is moving with.

## Summary

| Metric | Question it answers | Scenarios | Ideal | Column |
|---|---|---|---|---|
| LSTM slope share $a_{\mathtt{LSTM}}$ | How much of the slope did the LSTM take? | B, C, D | 0 | `share_lstm` |
| Level slope share $a_{\mathtt{L}}$ | How much did the baseline level take? | B, C, D | 1 | `share_level` |
| Residual slope share $a_{\mathtt{W}}$ | How much was left unexplained, in the noise? | B, C, D | 0 | `share_noise` |
| Share sum | Check: the three shares add up to the data's slope | B, C, D | ≈ 1 | `share_sum` |
| Trend share $a_{\mathtt{T}}$ | Does the SSM's trend state know the true slope? | B, C, D | 1 | `share_trend` |
| LSTM drift | Does the LSTM output drift? (data units / week) | all | 0 | `lstm_drift` |
| LSTM pattern error | Is the LSTM output the seasonal pattern, and only that? | all | small | `pattern_rmse` |
| RMSE, MAE | How far are the one-step forecasts? | all | small | `rmse`, `mae` |
| Log-likelihood (LL) | How probable are the observations under the forecast? | all | large | `ll` |
| CRPS | Accuracy and uncertainty together | all | small | `crps` |
| 95% coverage | Are the forecast intervals honest? | all | 0.95 | `coverage_95` |
| Adaptation kept $\eta$ | Can the LSTM still learn a real pattern change? | D vs B | 1 | `eta` |
| Reading ratios $\varrho_m$, $\varrho_d$ | Is the constraint fighting the seasonal cycle? | A (diagnostic) | ≈ 1 | `reading_ratio_mean`, `reading_ratio_slope` |

## 1. Slope shares — the main result (B, C, D)

**Question.** The data contain a slope $\delta$. Which component carries it?

**How it is computed.**

1. Average each component over each 52-week cycle of the test split. One whole cycle
   of a seasonal pattern averages to zero, so what remains is the slow part only.
2. Fit a straight line through the 11 cycle averages. Its slope is that component's
   slope, $\hat\beta$.
3. Divide by the true slope $\delta$.

$$a_{\mathtt{LSTM}} = \hat\beta(u)/\delta,\qquad a_{\mathtt{L}} = \hat\beta(\ell)/\delta,\qquad a_{\mathtt{W}} = \hat\beta(w)/\delta.$$

Because $\ell_t + u_t + w_t = y_t$ exactly and the line fit is linear, the three shares
add up to $\hat\beta(y)/\delta$, the slope actually present in the data. That is
1 up to about ±0.5% of noise. **They are fractions of one pie.**

**How to read them.**

- $a_{\mathtt{LSTM}} = 0.17$: the LSTM explains 17% of the slope. This is what the
  constraint should prevent.
- $a_{\mathtt{L}} \approx 1$, $a_{\mathtt{LSTM}} \approx 0$: the slope sits in the baseline.
  **The constraint works.**
- $a_{\mathtt{W}}$ clearly above 0: neither the baseline nor the LSTM explains part of
  the slope, so the residuals grow over time. With a rigid baseline ($\sigma_w = 0$),
  that happens when the trend state is wrong and cannot correct itself fast enough.
  A constraint that blocks the LSTM without the baseline taking over moves the slope
  here: the LSTM is kept clean, but forecasts suffer.
- Negative shares are possible: a component moving against the slope, compensated
  by another one.

## 2. Trend share (B, C, D)

**Question.** Does the trend state of the SSM estimate the true slope?

$$a_{\mathtt{T}} = \frac{\text{mean of the filtered trend over the test split}}{\delta}.$$

$a_{\mathtt{T}} = 1$ means the baseline's own slope parameter is right. It differs from
$a_{\mathtt{L}}$ because the level can also be moved directly by the observation
updates. With $\sigma_w = 0$ the trend barely moves during the test, so $a_{\mathtt{T}}$
mostly shows what offline training left it with.

## 3. LSTM drift (all scenarios)

$\hat\beta(u)$ in data units per week: the same line fit as the share, without dividing
by $\delta$. It is the only drift measure in scenario A, where $\delta = 0$ and shares
are undefined. In A it should be 0 for every arm. In B–D it equals
$a_{\mathtt{LSTM}}\,\delta$.

## 4. LSTM pattern error (all scenarios)

RMSE between the LSTM output $u_t$ and the true seasonal pattern $p_t$ (after the
phase shift in D), in data units. It goes up if the LSTM carries drift or an offset
(B, C without constraint), and if a tight constraint distorts the cycle (A at small
$\kappa$). Small values mean the LSTM learned the pattern and nothing else.

## 5. Predictive scores (all scenarios)

All computed from the one-step-ahead forecast $\mathcal{N}(\mu_t, \sigma_t^2)$.

- **RMSE, MAE.** Typical forecast error. The noise floor is $\sigma = 0.1$.
- **LL.** Average Gaussian log density of $y_t$. Penalizes over- and underconfidence.
- **CRPS.** Distance between the forecast distribution and the observation, in data
  units. It equals MAE for a point forecast and rewards sharp, well-placed intervals.
  This is the main cost metric.
- **95% coverage.** Share of observations inside $\mu_t \pm 1.96\sigma_t$.
  Below 0.95: intervals too narrow. Above: too wide.

**How to read them.** These measure the *cost* of the constraint. In A there is no
slope to keep out, so any loss against `rsr_kinf` is pure cost. In B and C an LSTM
that takes the slope can *improve* short-term forecasts, so better scores without the
constraint are not evidence against it.

## 6. Adaptation kept $\eta$ (D vs B)

**Question.** In D the seasonal pattern shifts by a quarter cycle at week 416. The
LSTM *should* learn this. Does the constraint stop it?

**How it is computed.** B and D are identical before week 416. For each arm, take the
CRPS after week 416 in D minus the CRPS over the same weeks in B: that is the extra
error caused by the change, $E$. Then compare arms:

$$\eta = \frac{E_{\texttt{frozen}} - E_{\text{arm}}}{E_{\texttt{frozen}} - E_{\texttt{rsr\_kinf}}}.$$

**How to read it.**

- $\eta = 1$: the arm recovers from the change as well as unconstrained RSR.
- $\eta = 0$: no better than an LSTM that does not learn at all.
- $\eta$ falling as $\kappa$ falls means tight tolerances also block real pattern changes.

## 7. Reading ratios (diagnostic, scenario A)

**Question.** Is the constraint fighting the seasonal cycle itself?

Even a perfect seasonal LSTM output gives the EWMA readings a small ripple, the
*leakage*: the memories don't average a cycle out exactly. The ratio compares the
readings of the run (RMS over the test split) with the readings the true pattern alone
would give:

- $\varrho \approx 1$: the LSTM's low-frequency content matches the true pattern.
- $\varrho \ll 1$: the constraint flattens the cycle, a sign that $\tau$ is below the
  leakage. Expect a CRPS cost.
- $\varrho \gg 1$: the LSTM carries an offset or drift.

Shown for A, where there is no drift to confuse the reading.

## How the metrics combine into a verdict

| Observation | Meaning |
|---|---|
| $a_{\mathtt{LSTM}}$ falls with $\kappa$, $a_{\mathtt{L}}$ rises, $a_{\mathtt{W}}$ stays ≈ 0 | The constraint moves the slope from the LSTM to the baseline. **Works.** |
| $a_{\mathtt{LSTM}}$ falls but $a_{\mathtt{W}}$ rises | The LSTM is kept clean, but the rigid baseline doesn't take the slope over. **Partial.** |
| $a_{\mathtt{LSTM}}$ flat in $\kappa$ | The constraint has no effect on attribution. **Does not work.** |
| CRPS in A rises and $\varrho \ll 1$ at small $\kappa$ | Tolerance below the leakage: the constraint fights the cycle. |
| $\eta$ near 1 down to some $\kappa$ | Down to that tolerance, real pattern changes are still learned. |

The pre-registered thresholds are in `README.md`.
