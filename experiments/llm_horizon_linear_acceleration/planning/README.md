# Chronos refresh interval with LocalAcceleration

## Question and hypothesis

How does the Chronos refresh interval (H) behave when the Canari state-space
model contains a zero-process-noise `LocalAcceleration`, Chronos `Auxiliary`,
and `WhiteNoise`? The filter still assimilates one observation per week; only
the foundation model is refreshed every (H) weeks. We expect a constant-
acceleration baseline to preserve a quadratic trend over a block more reliably
than a constant-slope model, while residual forecast quality can still decline
as H grows. These are hypotheses to measure.

## Data construction and split

Use exactly the same ten preprocessed weekly benchmark series as the preceding
linear-trend H experiment. Standardize each on its first 52 observations only,
then add the synthetic quadratic trend

\[
g_t=\frac12 a t^2,\qquad a=0.05/52^2
\]

in warmup-standardized units, where t is the weekly sample index. The implied
acceleration is 0.05 warmup standard deviations per year squared; the trend
reaches approximately 10.5 units at the end of a 21-year series, comparable to
the earlier linear perturbation. Do not renormalize after adding the trend.
The processed benchmark's original full-series detrending is retained because
the user requested the same benchmark series; that inherited preprocessing is a
limitation for causal real-world interpretation. The added trend is synthetic,
and residual variation can still contain low-frequency drift.

The first 52 observations initialize scaling, state estimates, and Chronos
context and are not scored. The remainder is a sequential evaluation segment.
Missing observations are excluded from scores and preserve the predictive state
covariance; no test-period parameter is fitted or tuned.

## Model and initialization

The state equations are

\[
\begin{bmatrix}\ell_{t+1}\\b_{t+1}\\a_{t+1}\end{bmatrix}
=
\begin{bmatrix}1&1&1/2\\0&1&1\\0&0&1\end{bmatrix}
\begin{bmatrix}\ell_t\\b_t\\a_t\end{bmatrix}
 + \eta_t,
\qquad y_t=\ell_t+c_t+v_t.
\]

`LocalAcceleration(std_error=0)` supplies the constant-acceleration dynamics,
Chronos forecasts the residual c_t, and `WhiteNoise(std_error=0.1)` matches the
previous benchmark. A warmup OLS quadratic initializes level at the final
warmup index, slope, acceleration, and their full coefficient covariance. The
Chronos context is the warmup residual after subtracting that fitted quadratic.
The injected acceleration is not supplied to the model. The generic FFT-based
context initializer is deliberately avoided because it can absorb a polynomial
trend as spurious seasonality.

Use the same six frozen checkpoints and H grid
`{1, 3, 6, 13, 26, 52, 104, 208}`. Contexts are capped at the most recent 512
residual states. Independent series are batched with Chronos-2
`cross_learning=False`; all model/H combinations use float32 CPU inference and
seed 0. For H beyond a model's native length, use the installed library rollout
and record the native length and checkpoint hash.

The Chronos location and 0.1587/0.8413 quantile spread are converted to a
Gaussian working mean and variance as in the earlier experiments. The combined
variance is a total predictive variance, not an epistemic/aleatoric split.

## Baselines, metrics, and verification

Compare every foundation model and H to a LocalAcceleration + WhiteNoise
baseline with the same warmup initialization. Save per-series RMSE, MAE,
Gaussian log-likelihood, CRPS, 90% coverage, width, refresh counts, posterior
level/slope/acceleration diagnostics, and native forecast quantiles. Also save
slope and acceleration errors against the injected polynomial coefficients as
diagnostics; they are not uniquely identifiable component truth on real series.

Before inference, tests cover exact quadratic propagation across all H values,
OLS covariance and residual contexts, missing observations, manual versus
library updates, future-data independence, and H refresh cadence. A smoke run
uses two series and H={1,13,208} for every model. The full run is timestamped,
never overwrites earlier experiments, and produces PDF/PGF/PNG H curves and
ts67 diagnostic figures using the project plotting defaults.
