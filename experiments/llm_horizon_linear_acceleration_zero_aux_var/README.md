# LocalAcceleration H sweep with zero Chronos variance

Repeats `experiments/llm_horizon_linear_acceleration` (reference run
`run_20260925T103556_080836Z`) with the variance returned by Chronos forced to
exactly zero. Means, data, models, H grid and filtering protocol are unchanged;
`core.py` and the runner helpers are imported from the reference experiment.

- [Protocol](planning/README.md)
- [Full report](results/run_20260925T131113_839408Z/report.md)
- [Figures](figures/run_20260925T131113_839408Z/)
- [Why the Chronos variance cannot be set to zero](why_not_zero_variance.md)

With zero auxiliary variance the auxiliary Kalman gain is zero, so Chronos is fed
its own forecasts instead of filtered residuals. At H=1 (all models) and H=3 (three
Bolt models) the self-fed rollout diverged; the Bolt groups overflowed float32 and
are listed in `failures.csv`. The completed run has 420 model/series/H rows.

`results/run_20260925T130427_794317Z` is an earlier attempt that stopped at the
first overflow (no `COMPLETED` marker); it is kept unmodified.

The experiment reports total predictive uncertainty only.
