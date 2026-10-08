# Bayesian latent-history propagation through frozen Chronos

## Research question and hypotheses

Can posterior uncertainty about the latent input signal produce a calibrated
state-uncertainty component when propagated through a frozen Chronos transition?
Does a coherent observation/dynamics model separate it from future process and
measurement noise? Exact variance identities, numerical inference accuracy, and
empirical calibration are distinct claims. A failed calibration result is valid.

All code is experiment-local. No `src/` changes. Native Chronos quantiles are
not uniquely decomposed: these are explicitly specified new predictive models.
Chronos weights are fixed; no weight-posterior uncertainty is inferred.

## Definitions and stages

### A: analytic controls

Known seasonal mean $s_t$, residual $U_t = a U_{t-1} + W_t$, $X_t = s_t + U_t$,
$Y_t = X_t + V_t$. Independent $W \sim \mathcal{N}(0, Q)$, $V \sim \mathcal{N}(0, R)$,
stationary prior for $|a| < 1$. Static control has $a = 1$, $Q = 0$ and
$U \sim \mathcal{N}(0, 1)$. Forward filtering/backward sampling
draws JOINT latent histories conditioned only on observations in the past
context. Smoothing inside that past context is allowed; future observations
are never available. Verify against direct Gaussian conditioning.

At lead $h$ the exact current-state contribution is $E_h = a^{2h} P_t$; future
noise is

$$A_h = R + Q \sum_{j=0}^{h-1} a^{2j}.$$

Linear forecasts applied to posterior draws
must recover these components. An independent-marginal sampling control shows
why preserving history covariance matters for history-dependent maps.

### B: AR-filtered Chronos conditional predictive model

Frozen $g_h$ is explicitly the Chronos MEDIAN at lead $h$, used as the conditional
Gaussian mean. The installed pipeline's return named mean is actually median.
Let $Z$ be a past latent history drawn from the AR model above. Define

$$q(Y_{t+h} \mid \mathcal{D}_t) = \int \mathcal{N}\big(g_h(Z), A_h\big)\, p_{\mathrm{AR}}(Z \mid \mathcal{D}_t)\, \mathrm{d}Z.$$

Then $T_h = A_h + \mathrm{Var}_Z[g_h(Z)]$ exactly under this conditional model.

The past posterior comes from AR dynamics and the future map comes from Chronos.
This is a valid per-origin HYBRID conditional forecast, not the posterior of a
sequential Chronos dynamical model. At $h > 1$, prescribed AR future noise is not
noise propagation through recursive nonlinear Chronos transitions. Comparing
its state variance with the AR oracle tests suitability, not identical models.

Methods: raw Chronos on observed history (Gaussian moment baseline, components
unspecified); $g(\text{posterior mean})$ with $A$ only; joint posterior propagation with
known noise; joint propagation with training-fitted $Q, R$ (plug-in uncertainty);
independent marginal histories and twice as many joint draws on paired first
test origins as sensitivities. Never assign native interval variance to $Q$ or $R$.

### C: self-consistent Chronos state-space model (primary mechanism test)

$$X_{t+1} = g_1(X_{t-31:t}) + W_t, \qquad Y_t = X_t + V_t, \qquad Q = 0.01,\ R = 0.04.$$

Initial 32 latent values are known $\sin(2\pi t/16)$; no measurement noise is
assimilated there. Both artificial truth and inference use exactly the same
frozen Chronos median transition. Five independently seeded 64-step episodes.
There is no training/tuning on these evaluation episodes; $Q, R$ are known.

Use a fully adapted particle filter retaining the full 32-point context in
each particle. Given previous context $z_i$ and equal old weights, predictive
ancestor weights are proportional to $\mathcal{N}(y; g(z_i), Q + R)$.
Systematically resample, then append

$$x \sim \mathcal{N}\left(g(z_i) + \frac{Q}{Q + R}\big(y - g(z_i)\big),\ \frac{QR}{Q + R}\right).$$

Every prediction uses the posterior BEFORE the target observation is assimilated.

Primary particle count 64; repeat the identical seed0 observations with 256
particles as a numerical sensitivity, not an analytic oracle. Save ESS and
ancestral diversity. One-step $A = Q + R$, $E = \mathrm{Var}_{\text{particle}}[g(Z)]$,
$T = A + E$. The observed target is $Y_{\text{next}}$; the predictable target is
$g(\text{true latent history})$. Latent $X_{\text{next}}$ also contains future
process noise and uses $E + Q$. State bands alone target $g$, not $X_{\text{next}}$.
Finite-particle $E$ can be underestimated by particle degeneracy.

This establishes a known-noise mechanism under the model's own dynamics. It
does not identify noise in arbitrary real series or establish real Canari gains.

### D: explicit parameter posterior in the analytic model

Known $a$/seasonality, unknown $Q, R$. Use a finite log-spaced grid prior with
9 $Q$ nodes in $[10^{-4}, 1]$ and 9 $R$ nodes in $[0.005, 1]$; static $Q = 0$
uses 9 $R$ nodes.
These fixed supports are specified in the synthetic signal's amplitude-one units.
Equal node prior mass is a declared discretized log-scale prior. Update weights using exact training
innovation likelihood. Save every node/weight, posterior means/intervals and
edge mass (coarse-grid/truncation diagnostic). No validation/test tuning.

Forecast from the training endpoint (origin768, first validation target), using
the parameter and conditional state posteriors from the SAME training prefix.
Compute exact posterior-predictive moments with three terms:
$\mathbb{E}_\theta[A_h]$, $\mathbb{E}_\theta[a^{2h} P_t(\theta)]$, and
$\mathrm{Var}_\theta[\mu_h(\theta)]$. Correlated
state/parameter inference is respected by conditional state posteriors.
This is an ordinary posterior predictive distribution under the finite grid
prior; no validation or test target has been assimilated. Discretization can
exclude the true continuous $Q$/$R$ values and must be reported as a limitation.
Chronos plug-in fits in B do not silently include this parameter uncertainty.

## Datasets, train/validation/test and fixed configuration

Reuse saved synthetic data from the completed uncertainty-decomposition run
`run_20260924T123843_788223Z`; use the context-floor loader to verify source
provenance, then retain synthetic series only. Same four cases, five seeds,
1536 weekly steps. Train [0,768), validation [768,1024), test [1024,1536).
Deterministic seasonal mean and a are privileged known structure in B/D.
Fit $Q$/$R$ on training observations only. Static $R$ uses the training residual
sample variance. Stage B context64, prediction block52, leads1,4,13,52;
validation origins768,820; test origins1024,1076,...,1440. Blocks stay inside
their split and do not overlap within a split. Test contexts may include already
observed test weeks, as in sequential forecasting.

64 posterior histories per primary call, 128 on the paired first test origins.
Separate univariate inputs, cross_learning=False, batch_size64. Verify batched
versus single-input transition invariance. Seeds for data, posterior sampling,
particle proposals and resampling are separately recorded. Save full series,
posterior-history example, per-origin member forecasts, raw baseline quantiles,
filter diagnostics, per-target scores, noise fits, parameters, hashes/versions.
Run a two-seed reduced smoke pipeline before full inference. Never overwrite runs.

## Metrics, figures and success criteria

Report Gaussian moment RMSE/MAE, NLL, CRPS and 90% coverage for observations and
predictable targets. Additionally score the explicitly defined Gaussian-mixture
observation forecast using exact finite-mixture NLL/CRPS. Zero state variance
has no Gaussian state score; save that denominator. All bands shown in forecast
figures use plus/minus ONE standard deviation. Variances add; SDs do not.

Compare hybrid methods on identical test targets; sensitivities use paired
subsets only. Aggregate within seed and lead, then leads within seed, then seeds
within case, then cases equally. Save seed SDs and counts; no IID significance
claims across serial targets. Stage C/D summaries remain separate from hybrid.
Reference errors in B are explicitly errors relative to a different AR model.

Figures tell a sequence: noisy observations/latent truth; smoothed past histories;
their covariance; propagated forecast means versus plug-in; analytic recovery;
hybrid horizon components; coherent filtering/forecast bands; $E$/$Q$/$R$ contributions;
particle-count sensitivity; calibration; $Q$/$R$ identification/posterior; three-term
parameter decomposition. All use project Matplotlib defaults, no figure titles,
clear units, compact legends and PDF/PGF exports plus PNG previews. Captions
explain the target and assumptions. Inspect final exports visually.

Exact controls and source/split/variance checks must pass. Coherent particle
calibration and particle-count sensitivity are reported even when disappointing.
Expected outcome is a sound computational mechanism, with empirical validity
remaining conditional on correct dynamics/noise and sufficiently resolved
posterior inference. The entire experiment may reject Chronos as latent dynamics.
