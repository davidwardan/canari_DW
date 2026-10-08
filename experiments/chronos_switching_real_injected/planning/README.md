# Chronos-2 small on real weekly data with an injected anomaly

## Research question and hypothesis

Does the switching particle filter identify a synthetic level anomaly embedded in
an otherwise unchanged real Canari weekly series, while preserving the same
aleatoric/epistemic variance identity? The hypothesis is that posterior abnormal
probability will rise during the injected interval, remain low on the clean
counterfactual, and return to normal after the shift is removed.

## Data and leakage control

The data are the repository's raw weekly benchmark series `ts67` and `ts66`.
The original first-104-point standardization and train-only noise estimates are
read from the completed Chronos-2 real-data experiment. The model is evaluated
online on a short held-out 2022 test window. No test observations are used to
fit scaling, (Q), (R), the regime transition matrix, or the anomaly size.

For each series, six real observations before the anomaly are used as online
pre-anomaly context. A +2.5 standardized-unit level shift is added for 12 weeks,
then removed for six weeks. The clean control uses exactly the same window without
the added shift. The underlying real observation remains the counterfactual target
for injected-window forecast diagnostics; it is not treated as a latent truth.

## Model

Chronos-2 small supplies the frozen one-step median transition from a 32-week
latent history. The two regimes use the train-only per-series (Q,R) estimates,
the transition matrix from the mechanism experiment,

\[
P=\begin{pmatrix}.995&.005\\.08&.92\end{pmatrix},\qquad
\delta_N=0,\quad \delta_A=2.5.
\]

Each particle stores the latent context and regime. Candidate regimes are scored
against the observed data, resampled to a fixed particle count, and updated with
the exact scalar Gaussian conditional. After assimilation, the detected additive
regime residual is removed from the newest history value before the next Chronos
query; this keeps Chronos focused on the baseline dynamics and prevents a
persistent injected level shift from being fed back as if it were normal history.
Forecast variance is saved as

\[
T=A+E_{\mathrm{state}}+E_{\mathrm{regime}}.
\]

## Baselines and metrics

The report compares switching and fixed-normal filters on clean and injected
series. It reports observed-data NLL/CRPS, counterfactual-clean-target NLL,
90% coverage, Brier score, log loss, AUROC, detection/recovery delay, false
alarms, particle sensitivity, and decomposition reconstruction error.

## Expected outcome

The injected series should produce a high abnormal posterior during the shift,
with a low posterior on the clean series and after recovery. This is a detection
mechanism test, not evidence that the same synthetic level shift represents every
real anomaly.
