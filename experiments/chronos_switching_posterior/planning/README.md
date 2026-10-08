# Switching Chronos particle-filter experiment

## Research question and hypothesis

Can a discrete normal/abnormal regime posterior be added to the latent-history
particle filter while retaining a mathematically explicit uncertainty
decomposition? The first controlled test uses the bounded Chronos median as a
frozen baseline and gives the abnormal regime a known residual level shift.
The hypothesis is that the switching filter will move posterior mass toward
the abnormal regime after a shift, return mass toward normal after recovery,
and preserve the variance identity

$$
T=A+E_{\text{state}}+E_{\text{regime}}.
$$

This is a mechanism test. It does not tune or retrain Chronos and does not
claim that the residual shift is the right real-world abnormal model.

## Model

The latent regime is a two-state Markov chain $S_t\in\{N,A\}$ with a
declared transition matrix. The recent latent context $Z_t$ contains the
last 32 values. For regime $s$,

$$
M_{t+1}=g_{\mathrm{bounded}}(Z_t)+\delta_s,
\qquad
X_{t+1}=M_{t+1}+W_{t+1},
\qquad
Y_{t+1}=X_{t+1}+V_{t+1}.
$$

The normal shift is zero and the abnormal shift is fixed at +0.75 in the
synthetic amplitude-one units. Both regimes use known \(Q=0.01\) and
\(R=0.04\) in the first run. Truth and inference use the same bounded map
\(g_{\mathrm{bounded}}(z)=3\tanh(g_{\mathrm{Chronos}}(z)/3)\).

Each particle stores a latent context and its regime. At every observation,
candidate regimes are propagated through the Markov matrix, scored by their
Gaussian observation likelihood, resampled, and updated with the exact scalar
Gaussian \(X\mid Z,S,Y\) proposal. This is a switching particle filter.

## Decomposition

Let \(p_s=P(S_{t+1}=s\mid D_t)\), \(\mu_s\) be the regime-conditional
forecast mean, and \(E_s\) its within-regime history variance. Then

\[
A=\sum_s p_s(Q_s+R_s),
\]

\[
E_{\mathrm{state}}=\sum_s p_sE_s,
\]

\[
E_{\mathrm{regime}}=\sum_s p_s(\mu_s-\mu)^2,
\qquad
\mu=\sum_s p_s\mu_s,
\]

and \(T=A+E_{\mathrm{state}}+E_{\mathrm{regime}}\). The regime term is
reported separately from within-regime latent-history uncertainty.

## Data, splits, and fixed configuration

The benchmark is synthetic and generated online from the pinned Chronos-2
checkpoint already used by the latent-posterior experiment. The initial
32-point context is a known period-16 sine wave. Each 96-step episode has a
normal segment, an abnormal segment, and a return to normal, with fixed switch
times known only to the evaluator. Five independent data seeds are used. The
primary filter has 64 particles; seed 0 is repeated with 128 particles.

The regime matrix, shift, noise variances, schedule, particle counts, model
revision, and random seeds are saved in `config.json`. No observations from a
held-out episode are used to choose parameters. The first run uses declared
parameters rather than validation tuning; a later acceleration experiment can
introduce SKF-style abnormal residual dynamics after this mechanism passes.

## Baselines and metrics

The report compares the switching particle filter with the current
non-switching bounded particle filter on the same generated episodes. It
reports regime posterior Brier score, log loss, AUROC, detection delay, false
alarms, forecast RMSE/CRPS, observation coverage, predictable-target
coverage, particle-count sensitivity, and decomposition reconstruction error.
The exact synthetic regime labels are used only for held-out scoring.

## Expected outcome

The switching posterior should rise after the declared abnormal shift and
fall after the return to normal. A successful run must pass constant-regime
and no-switch controls, preserve the variance identity to numerical
precision, and remain stable as particle count increases. Failure would mean
that a regime label is not identifiable from the chosen residual shift or
that the particle approximation is too degenerate.
