# Mathematical basis

Notation follows `docs/ewma.rst` for the constraint. Superscripts name model
components: $\mathtt{L}$ level, $\mathtt{T}$ trend, $\mathtt{LSTM}$, $\mathtt{W}$
white noise. A tilde marks standardized (normalized) units.

## 1. Generating process and model

The toy series is

$$y_t = c + \delta\,t + p_t + v_t,\qquad v_t \sim \mathcal{N}(0,\sigma^2),\quad \sigma = 0.1,$$

with $c = 0$, $P = 52$ and the zero-mean, $P$-periodic pattern $p_t$ given in the
README. The model works on $\tilde y_t = (y_t - \mu_y)/\sigma_y$, where $\mu_y$ and
$\sigma_y$ are the training-split mean and standard deviation. Its hidden state is

$$\mathbf{x}_t = [x^{\mathtt{L}}_t~~x^{\mathtt{T}}_t~~x^{\mathtt{LSTM}}_t~~x^{\mathtt{W}}_t~~m_{\mathrm{f},t}~~m_{\mathrm{s},t}]^{\intercal}.$$

- Baseline without process noise: $x^{\mathtt{L}}_t = x^{\mathtt{L}}_{t-1} + x^{\mathtt{T}}_{t-1}$,
  $x^{\mathtt{T}}_t = x^{\mathtt{T}}_{t-1}$. With $\sigma_w = 0$ the filtered trend is a
  recursive least-squares slope: it can carry a slope present from the start, and
  it moves less and less as data accumulate.
- LSTM: $X^{\mathtt{LSTM}}_t \sim \mathcal{N}(\mu^{\mathtt{LSTM}}_t,(\sigma^{\mathtt{LSTM}}_t)^2)$
  from the TAGI network, given the look-back of its own past posterior outputs and the
  week-of-year covariate. Its weights are updated with the Gaussian message from the
  raw prediction to the final posterior (constraint and observation included).
- White noise: $X^{\mathtt{W}}_t \sim \mathcal{N}(0,\sigma_v^2)$, independent over time.
- Observation, with **no additional noise term**:

$$\tilde y_t = \mathbf{h}^{\intercal}\mathbf{x}_t,\qquad \mathbf{h} = [1~~0~~1~~1~~0~~0]^{\intercal}.$$

## 2. EWMA memories, readings and leakage

With source $r_t \equiv x^{\mathtt{LSTM}}_t$ and $j \in \{\mathrm{f},\mathrm{s}\}$,

$$m_{j,t} = \rho_j\, m_{j,t-1} + (1-\rho_j)\, r_t,\qquad \rho_j = \exp(-\Delta t/s_j),\qquad \ell_j = \frac{\Delta t\,\rho_j}{1-\rho_j},$$

with $s_{\mathrm{f}} = 104$, $s_{\mathrm{s}} = 208$, $\Delta t = 1$. The two readings are

$$g^{m}_t = m_{\mathrm{s},t},\qquad g^{d}_t = \frac{m_{\mathrm{f},t} - m_{\mathrm{s},t}}{\ell_{\mathrm{s}} - \ell_{\mathrm{f}}}.$$

**Ramp.** For $r_t = \beta t$, let $e_{j,t} = m_{j,t} - \beta(t-\ell_j)$. Substituting,

$$e_{j,t} = \rho_j e_{j,t-1} + \beta\left[\ell_j(1-\rho_j) - \rho_j\Delta t\right] = \rho_j e_{j,t-1},$$

so $e_{j,t}\to 0$, $m_{j,t}\to\beta(t-\ell_j)$, and $g^{d}_t \to \beta$ exactly: the slope
reading measures a steady LSTM ramp without bias, after a transient of about $s_{\mathrm{s}}$.

**Harmonic.** For $r_t = A\cos(\omega t + \phi)$ the steady state is
$m_{j,t} = A\,|H_j(\omega)|\cos(\omega t + \phi + \arg H_j(\omega))$ with

$$H_j(\omega) = \frac{1-\rho_j}{1-\rho_j e^{-i\omega}}.$$

A seasonal LSTM output therefore leaks into both readings, with peak values
$A|H_{\mathrm{s}}|$ (mean) and $A|H_{\mathrm{f}} - H_{\mathrm{s}}|/(\ell_{\mathrm{s}}-\ell_{\mathrm{f}})$ (slope).

**Leakage used to scale the tolerances.** For the two-harmonic pattern, the peaks
are computed numerically: run both memories on $\tilde p_t = p_t/\sigma_y$ for 60 cycles
from zero and take

$$\lambda_m = \max_{t \in \text{last cycle}} |g^{m}_t|,\qquad \lambda_d = \max_{t \in \text{last cycle}} |g^{d}_t|.$$

For $\sigma_y \approx 0.79$ (B, C, D): $\lambda_m \approx 0.061$, $\lambda_d \approx 5.9\times10^{-4}$
per week. A tolerance arm with ratio $\kappa$ uses $\tau_m = \kappa\lambda_m$ and
$\tau_d = \kappa\lambda_d$; $\kappa = \infty$ removes both pseudo-observations.

## 3. Constraint update

After the LSTM prediction and the memory coupling, the prior is
$\mathcal{N}(\mathbf{x}_t;\boldsymbol{\mu}^{-}_t,\boldsymbol{\Sigma}^{-}_t)$. The pseudo-observations are

$$\mathbf{0} = \mathbf{C}\mathbf{x}_t + \boldsymbol{\epsilon}_t,\qquad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0},\mathbf{R}),\qquad \mathbf{R} = \operatorname{diag}(\tau_m^2,\tau_d^2),$$

where the rows of $\mathbf{C}$ form $g^{m}_t$ and $g^{d}_t$ (a row is dropped when its
tolerance is `None`). Gaussian conditioning gives

$$\boldsymbol{\mu}^{c}_t = \boldsymbol{\mu}^{-}_t - \mathbf{K}_t\mathbf{C}\boldsymbol{\mu}^{-}_t,\qquad
\boldsymbol{\Sigma}^{c}_t = \boldsymbol{\Sigma}^{-}_t - \mathbf{K}_t\mathbf{C}\boldsymbol{\Sigma}^{-}_t,\qquad
\mathbf{K}_t = \boldsymbol{\Sigma}^{-}_t\mathbf{C}^{\intercal}(\mathbf{C}\boldsymbol{\Sigma}^{-}_t\mathbf{C}^{\intercal}+\mathbf{R})^{-1}.$$

The real observation then updates $\mathcal{N}(\boldsymbol{\mu}^{c}_t,\boldsymbol{\Sigma}^{c}_t)$.

**The tolerance acts as a gain, not a dead-band.** For a single reading $G$ with
prior mean $\mu_G$ and variance $\sigma_G^2$,

$$\mu_G^{c} = \frac{\tau^2}{\sigma_G^2+\tau^2}\,\mu_G .$$

The reading is shrunk by a factor that does not depend on $\mu_G$. Within the
linear-Gaussian approximation the update is linear in the data, so the fraction of a
slope the constraint removes from the LSTM does not depend on the size of the slope.
A small drift is not let through just because it is small. Any dependence on
$\delta$ in the results comes from TAGI's nonlinear weight updates.

The per-step shift is small because $\sigma_G^2$, a slow average of LSTM variance, is
small. The effect accumulates through the weights: RSR trains the LSTM towards the
constrained posterior at every step.

**Why the split needs the constraint.** A ramp $\delta t$ can be produced by the
trend, or by the LSTM through its recurrent look-back (a slowly growing output).
The observation only constrains the sum $x^{\mathtt{L}}_t + x^{\mathtt{LSTM}}_t + x^{\mathtt{W}}_t$,
so without the readings the data cannot tell the two apart. The constraint adds
prior information: a persistent nonzero mean or slope of the LSTM output is
improbable.

## 4. Exact sum of the components

With no observation noise, the Kalman update for $\tilde y_t$ gives

$$\mathbf{h}^{\intercal}\boldsymbol{\mu}_{t|t} = \mathbf{h}^{\intercal}\boldsymbol{\mu}^{c}_t + \mathbf{h}^{\intercal}\boldsymbol{\Sigma}^{c}_t\mathbf{h}\,(\mathbf{h}^{\intercal}\boldsymbol{\Sigma}^{c}_t\mathbf{h})^{-1}(\tilde y_t - \mathbf{h}^{\intercal}\boldsymbol{\mu}^{c}_t) = \tilde y_t,$$

whenever $\mathbf{h}^{\intercal}\boldsymbol{\Sigma}^{c}_t\mathbf{h}>0$ (always, since $\sigma_v>0$). The memories
have zero entries in $\mathbf{h}$ and are conditioned before the observation, so this
holds with the constraint. In data units, with the level carrying $\mu_y$,

$$\underbrace{\sigma_y\mu^{\mathtt{L}}_{t|t}+\mu_y}_{\ell_t} + \underbrace{\sigma_y\mu^{\mathtt{LSTM}}_{t|t}}_{u_t} + \underbrace{\sigma_y\mu^{\mathtt{W}}_{t|t}}_{w_t} = y_t \quad\text{exactly, for every } t.$$

These are the filtered (causal) posteriors stored when each step first arrives in RSR.

## 5. Slope shares

**Block-mean slope.** For a series $z_t$ over the test split, with $K = 11$ complete
blocks $\mathcal{B}_k = \{t_0 + kP, \cdots, t_0 + (k+1)P - 1\}$ and centres $c_k$,

$$\bar z_k = \frac{1}{P}\sum_{t\in\mathcal{B}_k} z_t,\qquad
\hat\beta(z) = \frac{\sum_k (c_k-\bar c)(\bar z_k - \bar{\bar z})}{\sum_k (c_k-\bar c)^2}.$$

$\hat\beta$ is linear in $z$. If $z_t = \beta t + q_t$ with $q_t$ $P$-periodic and zero-mean,
then $\bar z_k = \beta c_k$ and $\hat\beta(z) = \beta$ exactly: the seasonal part does not
bias the estimate.

**Shares.** For $\delta \ne 0$,

$$a_{\mathtt{L}} = \frac{\hat\beta(\ell)}{\delta},\qquad a_{\mathtt{LSTM}} = \frac{\hat\beta(u)}{\delta},\qquad a_{\mathtt{W}} = \frac{\hat\beta(w)}{\delta},\qquad
a_{\mathtt{L}} + a_{\mathtt{LSTM}} + a_{\mathtt{W}} = \frac{\hat\beta(y)}{\delta}.$$

The last equality follows from Section 4 and linearity. $\hat\beta(y)/\delta$ differs from 1
only through the noise: its standard deviation is
$\sigma/\sqrt{P}\,/\sqrt{\sum_k(c_k-\bar c)^2}\,/\delta \approx 0.5\%$. In D the phase shift
starts on a block boundary, so the pattern still contributes nothing.

The trend share is the SSM's own slope estimate:

$$a_{\mathtt{T}} = \frac{1}{\delta}\,\frac{1}{n}\sum_{t\in\text{test}} \sigma_y\,\mu^{\mathtt{T}}_{t|t}.$$

In scenario A ($\delta = 0$) shares are undefined. There, the drift is reported as
$\hat\beta(u)$ in data units per week.

## 6. Other quantities

**One-step-ahead predictive scores.** RSR and the frozen filter give
$Y_t \mid y_{1:t-1} \sim \mathcal{N}(\mu_t,\sigma_t^2)$ before $y_t$ is used. With
$z_t = (y_t-\mu_t)/\sigma_t$:

$$\mathrm{LL} = \frac{1}{n}\sum_t\left[-\tfrac12\ln(2\pi\sigma_t^2) - \tfrac12 z_t^2\right],\qquad
\mathrm{CRPS} = \frac{1}{n}\sum_t \sigma_t\left[z_t(2\Phi(z_t)-1) + 2\varphi(z_t) - \tfrac{1}{\sqrt{\pi}}\right],$$

RMSE and MAE of $y_t-\mu_t$, and 95% coverage $\frac1n\sum_t \mathbb{I}(|z_t|\le1.96)$.
$\sigma_t^2$ is the total predictive variance; no epistemic/aleatoric split is reported.

**LSTM pattern error.** $\sqrt{\frac1n\sum_t (u_t - p_t)^2}$ over the test split, with
$p_t$ the true pattern (phase-shifted after $t_c$ in D).

**Adaptation kept (D).** For an arm $\mathsf{a}$, the excess CRPS caused by the phase
shift is the paired difference over $t \ge t_c$:
$E_{\mathsf{a}} = \mathrm{CRPS}_{\mathsf{a}}(\text{D}) - \mathrm{CRPS}_{\mathsf{a}}(\text{B})$. Then

$$\eta(\mathsf{a}) = \frac{E_{\texttt{frozen}} - E_{\mathsf{a}}}{E_{\texttt{frozen}} - E_{\texttt{rsr\_kinf}}}.$$

$\eta = 1$: the arm recovers from the change as well as unconstrained RSR on the
same snapshot. $\eta = 0$: no better than a frozen LSTM.

**Reading ratio (diagnostic).** With readings computed from the posterior memory
means of a run, and readings $g^{\star}$ computed from the true pattern
$\tilde p_t$ with the same recursion from $t = 0$,

$$\varrho_m = \frac{\operatorname{RMS}_{\text{test}}(g^{m})}{\operatorname{RMS}_{\text{test}}(g^{m\star})},\qquad
\varrho_d = \frac{\operatorname{RMS}_{\text{test}}(g^{d})}{\operatorname{RMS}_{\text{test}}(g^{d\star})}.$$

In A an LSTM that learned exactly $\tilde p_t$ gives $\varrho = 1$. $\varrho \ll 1$ means the
constraint flattens the low-frequency content of the seasonal output: it is fighting
the cycle. $\varrho \gg 1$ means the LSTM carries an offset or a drift.

## 7. Uncertainty across seeds

Each metric, including $\eta$, is computed per run (or per seed for the paired
$\eta$), then reported as mean ± sample standard deviation over the 5 seeds (LSTM
initialization and noise realization). The criteria use these seed means.

## 8. Assumptions and approximations

- TAGI propagates Gaussian moments with diagonal approximations; the weight update
  is an approximate Bayesian update.
- RSR replays each observation in up to 53 windows and restarts from a smoothed
  boundary that already contains the overlap. Evidence is reused; RSR is an
  algorithm, not exact inference.
- Shares use filtered posteriors (what the model believes online), not smoothed
  ones.
- The leakage scale uses the true pattern (toy oracle). On real data it must be
  estimated from training data only.
- Spans are fixed at 2 and 4 cycles; their effect is not part of this experiment.
