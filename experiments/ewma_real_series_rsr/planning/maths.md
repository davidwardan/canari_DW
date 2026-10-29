# Mathematical basis

Notation of the EWMA deck: level $z^{\mathtt{L}}_t$, trend $z^{\mathtt{T}}_t$, LSTM output
$z^{(\mathtt{O})}_t$, white noise $z^{\mathtt{V}}_t$, EWMA memories $z^{\mathtt{M}_{\mathtt{f}}}_t$ and
$z^{\mathtt{M}_{\mathtt{s}}}_t$, slope reading $s_t$. The constraint itself (memories,
pseudo-observations, Gaussian conditioning) is derived in
`../../ewma_drift_attribution/planning/maths.md`; only what differs is given here.

## 1. Data and reference baseline

$$y_t = y^{\text{det}}_t + \delta t,\qquad \delta = 0.005 \text{ per week},$$

where $y^{\text{det}}_t$ is the provider's detrended, standardized series. With the
training mean $m = \frac{1}{n_{\text{tr}}}\sum_{t<208,\,y_t\text{ observed}} y^{\text{det}}_t$, the reference
baseline and trend are

$$b_t = m + \delta t,\qquad z^{\mathtt{T}}\text{ reference} = \delta .$$

## 2. Model

$$y_t = z^{\mathtt{L}}_t + z^{(\mathtt{O})}_t + z^{\mathtt{V}}_t,\qquad
z^{\mathtt{L}}_t = z^{\mathtt{L}}_{t-1} + z^{\mathtt{T}}_{t-1},\qquad
z^{\mathtt{T}}_t = z^{\mathtt{T}}_{t-1},$$

in standardized units $\tilde y_t = (y_t - \mu_y)/\sigma_y$, with $\mu_y$, $\sigma_y$ from the
training split. There is no process noise on the baseline ($\sigma_w = 0$).

## 3. Correct initial baseline

The initial state is the state one step before the first observation, because the
first prediction applies the transition once. In standardized units:

$$\mathbb{E}[z^{\mathtt{L}}_{-1}] = \frac{b_{-1} - \mu_y}{\sigma_y} = \frac{m - \delta - \mu_y}{\sigma_y},\qquad
\mathbb{E}[z^{\mathtt{T}}_{-1}] = \frac{\delta}{\sigma_y},$$

$$\operatorname{var}[z^{\mathtt{L}}_{-1}] = 10^{-4},\qquad \operatorname{var}[z^{\mathtt{T}}_{-1}] = 10^{-6},\qquad
\operatorname{cov}(z^{\mathtt{L}}_{-1}, \cdot) = \operatorname{cov}(z^{\mathtt{T}}_{-1}, \cdot) = 0 .$$

The first prediction is then $\mathbb{E}[z^{\mathtt{L}}_0] = (b_0 - \mu_y)/\sigma_y$. These values are
set before the first epoch and again before every later epoch, replacing canari's
restart from the previous epoch's smoothed estimate. The EWMA memories restart at
known zero, as canari does.

With $\sigma_w = 0$ the trend can still move: it is a Gaussian state with prior standard
deviation $10^{-3}$ (about 20% of $\delta/\sigma_y \approx 0.0048$), updated by every
observation, with a gain that falls as data accumulate.

## 4. Tolerance scale from the training split

The average seasonal cycle of the training split, with the reference baseline removed,

$$\hat p_k = \operatorname{mean}_{c = 0..3}\left(y_{52c + k} - b_{52c + k}\right),\qquad k = 0,\cdots,51,$$

(missing values skipped) is run through both EWMAs from zero for 60 cycles, in
standardized units $\hat p_k/\sigma_y$. The peak absolute readings over the last cycle,

$$\lambda_m = \max_t |z^{\mathtt{M}_{\mathtt{s}}}_t|,\qquad \lambda_d = \max_t |s_t|,\qquad
s_t = \frac{z^{\mathtt{M}_{\mathtt{f}}}_t - z^{\mathtt{M}_{\mathtt{s}}}_t}{\ell_{\mathtt{s}} - \ell_{\mathtt{f}}},$$

are the seasonal leakage. A tolerance arm uses $\tau_m = \kappa\lambda_m$,
$\tau_d = \kappa\lambda_d$. Only training data enter $\hat p$, $\sigma_y$ and $\lambda$.

## 5. Metrics

Over the $n$ test weeks $t \in \mathcal{T}$, with filtered posterior means in data units
($\sigma_y\,\mathbb{E}[z^{\mathtt{L}}_t \mid y_{0:t}] + \mu_y$ for the level, $\sigma_y\,\mathbb{E}[z^{\mathtt{T}}_t \mid y_{0:t}]$ for the
trend):

$$\text{RMSE}_{\mathtt{L}} = \sqrt{\frac1n\sum_{t\in\mathcal{T}}\left(\mathbb{E}[z^{\mathtt{L}}_t\mid y_{0:t}] - b_t\right)^2},\qquad
\text{RMSE}_{\mathtt{T}} = \sqrt{\frac1n\sum_{t\in\mathcal{T}}\left(\mathbb{E}[z^{\mathtt{T}}_t\mid y_{0:t}] - \delta\right)^2}.$$

Missing observations do not remove a week from these sums: the states are still
predicted. Secondary scores use the one-step predictive
$\mathcal{N}(y_t;\mu_t,\sigma_t^2)$ on observed test weeks only:
RMSE, mean log-likelihood $\frac1n\sum_t\ln\mathcal{N}(y_t;\mu_t,\sigma_t^2)$, CRPS
$\sigma_t[z_t(2\Phi(z_t)-1)+2\varphi(z_t)-1/\sqrt\pi]$ with $z_t = (y_t-\mu_t)/\sigma_t$, and
coverage of $\mu_t\pm1.96\sigma_t$. $\sigma_t^2$ is the total predictive variance.

Each metric is computed per run and reported as mean ± sample standard deviation over
the 5 seeds, per series. The overall table averages the seed means over the series.

## 6. Assumptions and approximations

- The reference level of the detrended part is its training mean; slow residual
  wander in the provider's detrending counts as level error for all arms.
- TAGI's Gaussian approximations and RSR's reuse of evidence, as in the toy.
- $\lambda$ comes from the average training cycle, not from what the LSTM learns.
