# Mathematical basis

Random scalars are uppercase italic, matrices and random vectors bold uppercase, vectors
bold lowercase. Transpose is $^{\intercal}$.

## Main model

$Y_t=B_t+R_t+V_t$ with a zero-noise local trend $(B_t,D_t)$, the Bayesian LSTM
output $R_t$ and white noise $V_t\sim\mathcal{N}(v_t;0,\sigma_V^2)$, as in
`lstm_local_trend_rsr_ewma`. The fresh LSTM prediction is independent of the previous
joint state.

## Exponentially weighted fit of a line and harmonics

With $\omega_0=2\pi/52$ and $K$ harmonics, define the basis at lag $j$ (time $i=t+j$)
$$
\mathbf{f}(j)=\left[1~~j~~\cos(\omega_0 j)~~\sin(\omega_0 j)~\cdots~
\cos(K\omega_0 j)~~\sin(K\omega_0 j)\right]^{\intercal}\in\mathbb{R}^{2+2K}.
$$
At time $t$, the fit of past LSTM outputs $r_i$ with discount $\rho\in(0,1)$ is
$$
\hat{\boldsymbol{\theta}}_t=\operatorname*{arg\,min}_{\boldsymbol{\theta}}
\sum_{i\le t}\rho^{\,t-i}\left(r_i-\boldsymbol{\theta}^{\intercal}\mathbf{f}(i-t)\right)^2,
\qquad
\hat{\boldsymbol{\theta}}_t=[\hat\alpha_t~~\hat\beta_t~~\hat a_{1,t}~~\hat b_{1,t}~\cdots]^{\intercal}.
$$
$\hat\alpha_t$ is the line's value at $t$ and $\hat\beta_t$ its slope per week.
Expressing the harmonics relative to $t$ only rotates their coefficients; it does not
change $\hat\alpha_t$ or $\hat\beta_t$.

**Exact rejection.** If $r_i=a+bi+\sum_k[c_k\cos(k\omega_0 i)+s_k\sin(k\omega_0 i)]$,
the residual is zero for one $\boldsymbol{\theta}$, which has $\alpha=a+bt$, $\beta=b$.
Whenever $\sum_i\rho^{t-i}\mathbf{f}(i-t)\mathbf{f}(i-t)^{\intercal}$ has full rank,
that $\boldsymbol{\theta}$ is the unique minimizer. This holds even over less than one
cycle. Harmonics that are not modelled, or a wrong period, still leak.

### Recursion

Let $\mathbf{L}$ satisfy $\mathbf{f}(j+1)=\mathbf{L}\,\mathbf{f}(j)$:
$$
\mathbf{L}=\operatorname{blkdiag}\!\left(
\begin{bmatrix}1&0\\1&1\end{bmatrix},
\begin{bmatrix}\cos\omega_0&-\sin\omega_0\\\sin\omega_0&\cos\omega_0\end{bmatrix},
\cdots,
\begin{bmatrix}\cos K\omega_0&-\sin K\omega_0\\\sin K\omega_0&\cos K\omega_0\end{bmatrix}
\right).
$$
The fit at $t-1$ predicts $r_t$ by $\hat{\boldsymbol{\theta}}_{t-1}^{\intercal}\mathbf{f}(1)$,
and in the basis centred at $t$ the predicted coefficients are
$\mathbf{L}^{\intercal}\hat{\boldsymbol{\theta}}_{t-1}$. With the weighted Gram matrix
$$
\mathbf{S}=\sum_{j\ge0}\rho^{j}\mathbf{f}(-j)\mathbf{f}(-j)^{\intercal},
\qquad
\mathbf{S}=\rho\,\mathbf{L}^{-1}\mathbf{S}\,\mathbf{L}^{-\intercal}+\mathbf{f}(0)\mathbf{f}(0)^{\intercal}
$$
(solved as a discrete Lyapunov equation), the steady-state fit is the fixed-gain recursion
of Brown's general exponential smoothing:
$$
\hat{\boldsymbol{\theta}}_t=\mathbf{A}\,\hat{\boldsymbol{\theta}}_{t-1}+\mathbf{h}\,r_t,
\qquad
\mathbf{h}=\mathbf{S}^{-1}\mathbf{f}(0),
\qquad
\mathbf{A}=\left(\mathbf{I}-\mathbf{h}\,\mathbf{f}(0)^{\intercal}\right)\mathbf{L}^{\intercal}.
$$
All eigenvalues of $\mathbf{A}$ have modulus $\rho$. For an input of the model form,
the innovation $r_t-\mathbf{f}(0)^{\intercal}\mathbf{L}^{\intercal}\hat{\boldsymbol{\theta}}_{t-1}$
is zero once $\hat{\boldsymbol{\theta}}_{t-1}$ equals the exact fit, so exact
rejection persists from any exact starting fit.

### Warm start

From $\hat{\boldsymbol{\theta}}_0=\mathbf{0}$ the recursion has a transient that decays
like $\rho^t$ times a polynomial (six poles at $\rho$ for $K=2$). The memories are
therefore started at the finite-window fit of the frozen LSTM posterior means
$\bar r_{0:n-1}$,
$$
\hat{\boldsymbol{\theta}}_{n-1}=\left(\mathbf{X}\mathbf{W}\mathbf{X}^{\intercal}\right)^{-1}
\mathbf{X}\mathbf{W}\bar{\mathbf{r}},
\qquad
[\mathbf{X}]_{:,i}=\mathbf{f}(i-n+1),\quad
\mathbf{W}=\operatorname{diag}(\rho^{\,n-1-i}),
$$
with zero variance. The EWMA arm starts at $m_{j,n-1}$ computed by its own recursion over
the same $\bar r_{0:n-1}$ from zero.

### Joint state and constraint

The memories $\mathbf{M}_t=[M_{\alpha,t}~M_{\beta,t}~M_{a_1,t}~M_{b_1,t}~\cdots]^{\intercal}$
enter the joint state $\mathbf{X}_t=[B_t~D_t~R_t~V_t~\mathbf{M}_t^{\intercal}]^{\intercal}$
with $\mathbf{M}_t=\mathbf{A}\mathbf{M}_{t-1}+\mathbf{h}R_t$: transition block
$\mathbf{A}$, coupling $\mathbf{h}$ onto the fresh $R_t$, zero process noise and zero
observation coefficients. This is CANARI's `EwmaConstraint` mechanism with
$\operatorname{diag}(\rho_f,\rho_s)\to\mathbf{A}$ and $(1-\rho_f,1-\rho_s)\to\mathbf{h}$.
The pseudo-observation is
$$
\begin{bmatrix}0\\0\end{bmatrix}=
\begin{bmatrix}M_{\alpha,t}\\M_{\beta,t}\end{bmatrix}+\mathbf{v}^c_t,
\qquad
\mathbf{v}^c_t\sim\mathcal{N}\!\left(\mathbf{v};\mathbf{0},\operatorname{diag}(\tau_m^2,\tau_d^2)\right),
$$
conditioned with the Joseph form before the real observation. TAGI receives the message
from the raw LSTM moments to the final posterior marginal, and the RTS smoother uses the
moments before the pseudo-observation, both unchanged from `EwmaConstraint`.

### Reading weights

Each reading is linear in the LSTM outputs, $\hat\alpha_t=\sum_{j\ge0}w^{\alpha}_j r_{t-j}$
with $\mathbf{w}_j=\rho^j\mathbf{S}^{-1}\mathbf{f}(-j)$, and similarly for
$\hat\beta_t$. With $K=2$, $\rho=e^{-1/208}$: $|H_\alpha|=|H_\beta|=0$ at $\omega_0$ and
$2\omega_0$, the ramp lag of $\hat\alpha_t$ is zero, and the current-output weights are
$w^{\alpha}_0=9.5\times10^{-3}$ and $w^{\beta}_0=2.3\times10^{-5}$. For the slow EWMA and the
EWMA slope summary they are $4.8\times10^{-3}$ and $4.6\times10^{-5}$.

## Metrics

On test weeks with $e_t=y_t-\hat\mu_{Y,t}$, $z_t=e_t/\hat\sigma_{Y,t}$:
RMSE, MAE, $\overline{\mathcal{L}}=\mathsf{N}^{-1}\sum_t\ln\mathcal{N}(y_t;\hat\mu_{Y,t},\hat\sigma_{Y,t}^2)$,
CRPS $=\mathsf{N}^{-1}\sum_t\hat\sigma_{Y,t}[z_t(2\Phi(z_t)-1)+2\varphi(z_t)-1/\sqrt\pi]$,
and coverage of $|z_t|\le1.96$. Against the known truth: RMS of $\mu_{D,t}-0.005$
(first and last 156 weeks), RMS of $\mu_{B,t}-0.005t$ and of
$\mu_{R,t}-s_t$ with $s_t$ the true seasonal component (last 156 weeks), and LSTM drift as
the least-squares slope through 52-week means of $\mu_{R,t}$. Readings are summarized
by their mean and the mean within-cycle standard deviation (ripple) over the last 156
weeks. All quantities are in data units, using the training mean and std. Bands are
mean $\pm$ one marginal standard deviation. Tables show mean $\pm$ sample standard
deviation over five seeds.
