# Exploratory stability follow-up

The prespecified raw Chronos recursion in run_20261007T175118_194212Z diverged.
Some filter predictions reached roughly 10^15 while Q=.01 and R=.04 stayed
fixed, and floating-point state spacing eventually exceeded the conditional
noise standard deviation. Arithmetic identities alone did not establish
reliable posterior inference. Retain the entire failed run and all its scores.

This explicitly exploratory extension changes the generative model:

    g_bounded(Z) = 3 * tanh(median_Chronos(Z) / 3)

The cap is a declared amplitude prior in the benchmark's amplitude-one units,
not estimated from held-out targets, optimized, or a unique natural property of
Chronos. No cap sweep or winner selection. Both artificial truth and inference
use this same bounded mean transition; independent Q/R shocks remain unchanged.
Bounded mean does not strictly bound latent X (Gaussian shocks are unbounded),
guarantee correct real dynamics, or guarantee adequate particle approximation.

Repeat the five seeded 64-step coherent episodes, primary64 particles and
seed0/256-particle sensitivity, and the two-origin recursive nested simulations.
Use the same noise/proposal seeds as the original design; the observed series
are different because the dynamics changed, so model accuracy scores cannot be
compared as a paired head-to-head forecast contest. Report all results.

Reuse the unchanged AR hybrid, analytic and training-parameter artifacts in a
new run directory, recording their source hashes. Their g_h remains the raw
Chronos median. Only the coherent model is bounded. Add plot_states-style prior
forecast and posterior X/M/W/V rows, preserving posterior cross-covariances.
The posterior of realized noise W,V is distinct from independent future noise
Q,R; posterior noise/state variances cannot simply be added after conditioning.

This follow-up tests whether an explicit stable mean prior makes the proposed
mechanism numerically useful. Its post-failure origin makes it exploratory;
external-data validation and independently prespecified replication remain open.
