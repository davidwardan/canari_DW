# Mathematical basis

Random scalars are uppercase italic, matrices and random vectors bold uppercase, vectors
bold lowercase, and vector/matrix moments bold Greek. Transpose is $^{\intercal}$.

## Main model

As in `lstm_local_trend_rsr_ewma`, the observation is $Y_t=B_t+R_t+V_t$ with baseline
level $B_t$, Bayesian LSTM output $R_t$ and white noise
$V_t\sim\mathcal{N}(v_t;0,\sigma_V^2)$. The baseline is a local trend with zero
process noise,
$$
\begin{bmatrix}B_t\\D_t\end{bmatrix}
=\begin{bmatrix}1&1\\0&1\end{bmatrix}
\begin{bmatrix}B_{t-1}\\D_{t-1}\end{bmatrix}.
$$
The LSTM prediction $R_t\sim\mathcal{N}(r_t;\mu_{R,0},\sigma_{R,0}^2)$ is assumed
independent of the previous joint state, as in native CANARI.

## Auxiliary local-trend model on the LSTM output

The constraint model describes the LSTM output as a local trend plus a white residual
$E_t$ that absorbs the periodic pattern:
$$
R_t=\mathbf{h}^{\intercal}\mathbf{Z}_t+E_t,\qquad
\mathbf{Z}_t=\mathbf{A}\mathbf{Z}_{t-1}+\mathbf{W}_t,\qquad
\mathbf{Z}_t=\begin{bmatrix}L_t\\S_t\end{bmatrix},
$$
$$
\mathbf{A}=\begin{bmatrix}1&1\\0&1\end{bmatrix},\quad
\mathbf{h}=\begin{bmatrix}1\\0\end{bmatrix},\quad
E_t\sim\mathcal{N}(e_t;0,\sigma_E^2),\quad
\mathbf{W}_t\sim\mathcal{N}\!\left(\mathbf{w}_t;\mathbf{0},\sigma_W^2
\begin{bmatrix}1/4&1/2\\1/2&1\end{bmatrix}\right).
$$
$\mathbf{A}$ and the process covariance $\mathbf{Q}$ are CANARI's `LocalTrend`
matrices. The constraint is that the LSTM output has no level and no slope:
$L_t=0$ and $S_t=0$.

### Steady-state Kalman filter

The one-step prediction covariance $\mathbf{P}^{-}$ solves the discrete algebraic
Riccati equation
$$
\mathbf{P}^{-}=\mathbf{A}\mathbf{P}^{-}\mathbf{A}^{\intercal}
-\mathbf{A}\mathbf{P}^{-}\mathbf{h}
\left(\mathbf{h}^{\intercal}\mathbf{P}^{-}\mathbf{h}+\sigma_E^2\right)^{-1}
\mathbf{h}^{\intercal}\mathbf{P}^{-}\mathbf{A}^{\intercal}+\mathbf{Q}.
$$
The gain, posterior covariance and memory transition are
$$
\mathbf{k}=\frac{\mathbf{P}^{-}\mathbf{h}}{\mathbf{h}^{\intercal}\mathbf{P}^{-}\mathbf{h}+\sigma_E^2},
\qquad
\mathbf{P}=(\mathbf{I}-\mathbf{k}\mathbf{h}^{\intercal})\mathbf{P}^{-},
\qquad
\mathbf{F}=(\mathbf{I}-\mathbf{k}\mathbf{h}^{\intercal})\mathbf{A}.
$$
They depend on $\sigma_W/\sigma_E$ only, apart from $\mathbf{P}\propto\sigma_E^2$.
For a realized LSTM sequence $r_{1:t}$ and prior
$\mathbf{Z}_0\sim\mathcal{N}(\mathbf{z}_0;\mathbf{0},\mathbf{P})$, the filter is exactly
stationary from the first step:
$$
f(\mathbf{z}_t\mid r_{1:t})=\mathcal{N}(\mathbf{z}_t;\mathbf{m}_t,\mathbf{P}),\qquad
\mathbf{m}_t=\mathbf{F}\mathbf{m}_{t-1}+\mathbf{k}\,r_t,\qquad \mathbf{m}_0=\mathbf{0}.
$$

### Memories in the joint state

The estimates become two zero-observation memories
$\mathbf{M}_t=[M_{L,t}~M_{S,t}]^{\intercal}$ of the joint state
$\mathbf{X}_t=[B_t~D_t~R_t~V_t~M_{L,t}~M_{S,t}]^{\intercal}$, with
$$
\mathbf{M}_t=\mathbf{F}\mathbf{M}_{t-1}+\mathbf{k}\,R_t .
$$
With the block transition $\mathbf{T}=\operatorname{blkdiag}(\mathbf{A},0,0,\mathbf{F})$,
process covariance $\mathbf{\Sigma}_W$, fresh LSTM moments inserted at the $R_t$
position, and the coupling $\mathbf{G}=\mathbf{I}+\mathbf{k}\,\mathbf{e}_R^{\intercal}$
restricted to the memory rows, the joint prior before any pseudo-observation is
$$
\boldsymbol{\mu}_0=\mathbf{G}\,\mathbf{u}_t,\qquad
\boldsymbol{\Sigma}_0=\mathbf{G}\,\mathbf{U}_t\,\mathbf{G}^{\intercal},
$$
where $(\mathbf{u}_t,\mathbf{U}_t)$ are $\mathbf{T}\boldsymbol{\mu}_{t-1}$ and
$\mathbf{T}\boldsymbol{\Sigma}_{t-1}\mathbf{T}^{\intercal}+\mathbf{\Sigma}_W$ with
the $R_t$ row and column replaced by the LSTM moments. The memories therefore carry
covariance with $R_t$, the baseline and their own past. This is exactly CANARI's
`EwmaConstraint` mechanism, with $\operatorname{diag}(\rho_f,\rho_s)$ replaced by
$\mathbf{F}$ and $(1-\rho_f,1-\rho_s)$ by $\mathbf{k}$.

### Pseudo-observation

The Kalman estimation error $\boldsymbol{\Delta}_t=\mathbf{Z}_t-\mathbf{m}_t$ is
independent of $r_{1:t}$ with covariance $\mathbf{P}$. The statement
$\mathbf{Z}_t=\mathbf{0}$ is therefore the pseudo-observation
$$
\mathbf{0}=\mathbf{C}\mathbf{X}_t+\boldsymbol{\Delta}_t,\qquad
\boldsymbol{\Delta}_t\sim\mathcal{N}(\boldsymbol{\delta};\mathbf{0},\mathbf{P}),
$$
where the rows of $\mathbf{C}$ select $M_{L,t}$ and $M_{S,t}$. There is no extra
tolerance: the auxiliary model's own uncertainty sets the strength.

**Approximation.** Errors $\boldsymbol{\Delta}_t$ at neighbouring steps are strongly
correlated, but each step's pseudo-observation is treated as fresh evidence, as the
EWMA constraint also does. The repeated evidence makes the constraint stronger than a
single $\mathbf{P}$ suggests.

**Rejected joint alternative.** Keeping $\mathbf{Z}_t$ as latent states and
conditioning on $0=R_t-L_t-E_t$, $L_t=0$, $S_t=0$ is the exact joint treatment.
Marginalizing $\mathbf{Z}_{1:\mathsf{T}}$ leaves the Gaussian factor
$\mathcal{N}(\mathbf{r};\mathbf{0},\mathbf{H}\boldsymbol{\Sigma}_{\mathbf{Z}}\mathbf{H}^{\intercal}+\sigma_E^2\mathbf{I})$
on $\mathbf{r}=r_{1:\mathsf{T}}$. Because the constrained $\mathbf{Z}$ has a small
variance, this is nearly white noise of variance $\sigma_E^2$. Its precision is the
same for an offset, a ramp and the seasonal cycle, so it cannot target drift.

### Frequency response and leakage

For $r_t=\operatorname{Re}(e^{i\omega t})$ the steady memory response is
$$
\mathbf{H}(\omega)=\left(\mathbf{I}-\mathbf{F}e^{-i\omega}\right)^{-1}\mathbf{k},
$$
with $\mathbf{H}(0)=[1~0]^{\intercal}$. The filter has two integrators, so a ramp
$r_t=a+bt$ is tracked with zero steady-state error, $\mathbf{m}_t\to[a+bt~~b]^{\intercal}$.
A seasonal amplitude $\mathsf{A}$ at $\omega_P=2\pi/52$ leaks
$\mathsf{A}\,|H_L(\omega_P)|$ into the level memory and
$\mathsf{A}\,|H_S(\omega_P)|$ into the slope memory.

### Settings

$\sigma_E$ is the std of the least-squares-detrended, standardized training
observations. The ratio $\sigma_W/\sigma_E=7.07\times10^{-6}$ is the one for which
$M_{L,t}$ first reaches $1-e^{-1}$ after 208 weeks for a unit step $r_t=1$. Then
$$
\mathbf{k}=\begin{bmatrix}3.75\times10^{-3}\\7.06\times10^{-6}\end{bmatrix},\qquad
\sqrt{\operatorname{diag}(\mathbf{P})}=\sigma_E
\begin{bmatrix}0.0613\\1.63\times10^{-4}\end{bmatrix},\qquad
\rho_{\mathbf{P}}=0.71,
$$
$$
|\mathbf{H}(\omega_P)|=\begin{bmatrix}0.031\\5.9\times10^{-5}\end{bmatrix}.
$$

### Penalty spectrum

`scripts/penalty_spectrum.py` compares how strongly each constraint penalizes a
sequence $\mathbf{r}$, using only the constraint factors and no observation. Each
mechanism defines a Gaussian factor $\exp(-\tfrac12\mathbf{r}^{\intercal}\boldsymbol{\Lambda}\mathbf{r})$:

- Kalman memories: $\boldsymbol{\Lambda}=\sum_t\mathbf{J}_t^{\intercal}\mathbf{P}^{-1}\mathbf{J}_t$,
  where $\mathbf{m}_t=\mathbf{J}_t\mathbf{r}$;
- EWMA: the same, with the slow-mean and slope rows and
  $\operatorname{diag}(\tau_m^2,\tau_d^2)$;
- joint alternative: $\boldsymbol{\Lambda}=(\mathbf{H}\boldsymbol{\Sigma}_{\mathbf{Z}}\mathbf{H}^{\intercal}+\sigma_E^2\mathbf{I})^{-1}$.

The equivalent std of a pattern $\mathbf{r}$ is
$\sqrt{\mathsf{T}^{-1}\mathbf{r}^{\intercal}\mathbf{r}\,/\,\mathsf{T}^{-1}\mathbf{r}^{\intercal}\boldsymbol{\Lambda}\mathbf{r}}$.
A smaller value means a stronger penalty. Drift should be penalized much more strongly
than the seasonal cycle.

## Conditioning and learning

With $(\boldsymbol{\mu}_0,\boldsymbol{\Sigma}_0)$ the joint prior above,
$$
\mathbf{K}_c=\boldsymbol{\Sigma}_0\mathbf{C}^{\intercal}
(\mathbf{C}\boldsymbol{\Sigma}_0\mathbf{C}^{\intercal}+\mathbf{P})^{-1},\qquad
\boldsymbol{\mu}_c=\boldsymbol{\mu}_0-\mathbf{K}_c\mathbf{C}\boldsymbol{\mu}_0,
$$
$$
\boldsymbol{\Sigma}_c=(\mathbf{I}-\mathbf{K}_c\mathbf{C})\boldsymbol{\Sigma}_0
(\mathbf{I}-\mathbf{K}_c\mathbf{C})^{\intercal}+\mathbf{K}_c\mathbf{P}\mathbf{K}_c^{\intercal}.
$$
The one-step prediction of $Y_t$ uses $(\boldsymbol{\mu}_c,\boldsymbol{\Sigma}_c)$, and
the real observation is assimilated afterwards. TAGI receives the message from the raw
LSTM moments to the final posterior marginal $(\mu_{R,+},\sigma_{R,+}^2)$:
$$
\delta_\mu=\frac{\mu_{R,+}-\mu_{R,0}}{\sigma_{R,0}^2},\qquad
\delta_{\sigma^2}=\frac{\sigma_{R,+}^2-\sigma_{R,0}^2}{(\sigma_{R,0}^2)^2}.
$$
The RTS smoother uses the joint prior before the pseudo-observation,
$(\boldsymbol{\mu}_0,\boldsymbol{\Sigma}_0)$, and its cross-covariance with
$\mathbf{X}_{t-1}$, so constraint evidence propagates to earlier states. The memories
restart at known zero, with zero covariance, at every training epoch.

## Metrics

For finite test observations, $e_t=y_t-\hat{\mu}_{Y,t}$,
$z_t=e_t/\hat{\sigma}_{Y,t}$ and $\mathsf{N}$ samples:
$$
\operatorname{RMSE}=\sqrt{\mathsf{N}^{-1}\textstyle\sum_t e_t^2},\qquad
\operatorname{MAE}=\mathsf{N}^{-1}\textstyle\sum_t|e_t|,\qquad
\overline{\mathcal{L}}=\mathsf{N}^{-1}\textstyle\sum_t
\ln\mathcal{N}(y_t;\hat{\mu}_{Y,t},\hat{\sigma}_{Y,t}^2),
$$
$$
\operatorname{CRPS}=\mathsf{N}^{-1}\textstyle\sum_t\hat{\sigma}_{Y,t}
\left[z_t(2\Phi(z_t)-1)+2\varphi(z_t)-1/\sqrt{\pi}\right].
$$
Coverage is the fraction with $|z_t|\le1.96$. Trend deviation is the RMS distance of
the posterior trend mean from 0.005 over the first and last 156 test weeks. LSTM drift
is the least-squares slope through successive 52-week means of the LSTM posterior mean.
Predictions are unstandardized with the training mean and std; bands show mean
$\pm$ one marginal posterior or predictive std. No epistemic/aleatoric decomposition
is claimed. Per-series values are mean $\pm$ sample std over the five seeds; the
across-series summary is the median [IQR] of seed means.
