# Chronos refresh interval with LocalAcceleration

This is the acceleration counterpart to
`experiments/llm_horizon_linear_trend`. It replaces `LocalTrend` with a
zero-process-noise `LocalAcceleration` and replaces the added linear trend with
a synthetic constant-acceleration quadratic trend. The same ten processed
benchmark series, six Chronos models, H grid, warmup, context cap, Gaussian
adapter, and weekly filtering protocol are used.

- [Protocol](planning/README.md)
- [Full report](results/run_20260925T103556_080836Z/report.md)
- [Figures](figures/run_20260925T103556_080836Z/)

The completed run contains 490 per-series model/H rows, saved forecasts and
state trajectories, checkpoint hashes, batch-agreement checks, and generated
PDF/PGF/PNG figures.
The experiment reports total predictive uncertainty. It does not identify
epistemic and aleatoric components.
