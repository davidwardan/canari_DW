# What κ means in the EWMA constraint

κ sets how tight the EWMA constraint is, measured against the natural ripple a seasonal pattern causes in the constraint's readings.

## What the constraint does

The constraint pushes two readings of the LSTM output towards zero, each as a soft pseudo-observation with its own tolerance (standard deviation):

- the slow memory, with tolerance τ_m:

$$
0 = z^{\mathtt{M}_{\mathtt{s}}}_t + \epsilon_m, \qquad \epsilon_m \sim \mathcal{N}(0, \tau_m^2)
$$

- the slope reading, with tolerance τ_d:

$$
0 = s_t + \epsilon_d, \qquad s_t = \frac{z^{\mathtt{M}_{\mathtt{f}}}_t - z^{\mathtt{M}_{\mathtt{s}}}_t}{d}, \qquad \epsilon_d \sim \mathcal{N}(0, \tau_d^2)
$$

## Why the tolerance needs a reference: seasonal leakage

Even an LSTM whose output is only the seasonal pattern, with no drift, makes these readings ripple: EWMAs with 104- and 208-week memories do not fully average out a 52-week cycle. The leakage λ is the peak of that ripple. It is computed by running the two EWMAs on the average seasonal cycle of the training split (training data only):

$$
\lambda_m = \max_t \left| z^{\mathtt{M}_{\mathtt{s}}}_t \right|, \qquad \lambda_d = \max_t \left| s_t \right|
$$

## Definition of κ

κ expresses the tolerance in units of that ripple. The same κ is used for both readings:

$$
\tau_m = \kappa\, \lambda_m, \qquad \tau_d = \kappa\, \lambda_d
$$

| κ | Meaning | Effect seen on the 5 real series |
|---|---|---|
| κ > 1 | looser than the seasonal ripple | κ = 2 and 4 too weak to stop the LSTM taking the trend |
| κ = 1 | tolerance equals the largest reading a pure seasonal output gives | lowest level and trend error, no forecast cost |
| κ = 1/2 | half the seasonal ripple | same errors, no forecast cost, level/LSTM split stays stable after a level offset |
| κ < 1/2 | tighter than the pattern's own ripple: the constraint fights the seasonal pattern | forecast cost: CRPS +12% at κ = 1/4, +51% at κ = 1/8 |

Using κ rather than a raw τ makes the setting comparable across series: a series with a larger seasonal cycle gets a proportionally larger τ at the same κ.

## Values for the five series (data units)

| Series | λ_m | λ_d (per week) | τ_m at κ = 1/2 | τ_d at κ = 1/2 (per week) |
|---|---|---|---|---|
| ts67 | 0.050 | 4.8e-4 | 0.025 | 2.4e-4 |
| ts66 | 0.048 | 4.6e-4 | 0.024 | 2.3e-4 |
| ts57 | 0.068 | 6.5e-4 | 0.034 | 3.2e-4 |
| ts25 | 0.047 | 4.5e-4 | 0.024 | 2.3e-4 |
| ts26 | 0.045 | 4.4e-4 | 0.023 | 2.2e-4 |

For scale: the added trend is 0.005 per week, so at κ = 1/2 the slope tolerance is about 1/20 of the trend.

Source: `experiments/ewma_real_series_rsr` (run 20261028-212716, level-offset runs 20261028-223804 and 20261028-232316).
