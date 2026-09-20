# LLM anomaly-detection benchmark

## Research question

The upstream benchmark (Bayes-Works/canari @ `feature/global-model`,
`experiments/anomaly_detection_lstm.py` + `experiments/benchmark_anomaly_detection.py`)
compares three ways of filling the recurrent-pattern slot of a Switching Kalman
Filter: a **local** LSTM trained per series, a **global** LSTM fine-tuned from
pretrained weights, and the same global LSTM **zero-shot**.

**How does a pretrained time-series foundation model do on the same benchmark?**

The LSTM is replaced by Chronos-2 in canari's `Auxiliary` slot. Everything else --
the sigma_v grid search, the SKF parameter search, the testing scheme and the
anomaly magnitudes -- follows upstream, so the numbers drop into the same table as
the local, global-finetune and global-zeroshot conditions.

## What is held identical to upstream

| | upstream | here |
|---|---|---|
| Normal model | `LocalTrend` + LSTM + `WhiteNoise` | `LocalTrend` + **`Auxiliary`** + `WhiteNoise` |
| Abnormal model | `LocalAcceleration` + LSTM + `WhiteNoise` | `LocalAcceleration` + **`Auxiliary`** + `WhiteNoise` |
| Trend at init | zeroed after `auto_initialize_baseline_states` | same |
| sigma_v grid | `[0.02 … 0.2]`, 9 values | same |
| sigma_v selection | smallest within `delta` of best validation CRPS | same |
| SKF search space | `std_transition_error` loguniform `[5e-6, 1e-4]`, `norm_to_abnorm_prob` loguniform `[1e-5, 1e-4]`, `abnorm_to_norm_prob` quniform `[0.1, 0.2]`, `threshold` quniform `[0.05, 0.5]` | same |
| SKF search | Ray Tune + Optuna TPE, 50 trials, 20 startup | same |
| SKF objective | `cdf`: maximize `SKF.objective(detection_rate, false_rate, slope)` on train+validation | same |
| Objective shape | `detection_rate_cdf_std=0.2`, `anm_mag_cdf_median=0.2`, `anm_mag_cdf_shape=0.6` | same, **passed explicitly** |
| Tuned `slope` | `tune.choice(slope_search_space)` | same |
| Anomaly magnitudes | `[0.025, 0.05, 0.075, 0.225, 0.5, 0.75, 1.0]` sigma/year | same |
| Realizations | 25 per magnitude, injected **up and down** -> 50 | same |
| Anomaly window | `[test_start, test_end - 156]` | same |
| Detection window | 156 steps (three years) from onset | same |
| Detection rule | any step above threshold inside the window | same |
| False alarms | **count of time steps** above threshold on the clean series, divided by the calendar span | same |
| Time to detection | mean and std over detected realizations, in years | same |

## What necessarily differs

- **No training.** Upstream retrains the LSTM at every sigma_v candidate and picks
  the early-stopping epoch. Chronos-2 is frozen, so a candidate is scored by
  filtering train+validation once and measuring the same validation metric. The
  selection rule is unchanged.
- **No seed spread.** Upstream sweeps seeds because LSTM initialization is random.
  Chronos-2's quantile head is deterministic, so a single seed suffices and the
  spread reported here is across series, not across seeds.
- **No covariates.** Upstream attaches `week_of_year` for the LSTM; the `Auxiliary`
  interface takes no covariates, so none is used.
- **Splits.** Upstream pins `validation_start` / `test_start` as timestamps in a
  per-series config. Those configs are gitignored, so the splits here are ratios
  (`validation_ratio`, `test_ratio`) applied to each series.
- **One extra knob, `llm_horizon`.** The number of filtered steps between Chronos-2
  calls. `1` is the fair comparison, matching the LSTM's per-step cadence, and is
  the default. It is exposed because the companion experiments found it to matter
  enormously.
- **`SKF.objective` defaults.** The installed canari ships different defaults from
  upstream for three of the objective's shape parameters, so they are pinned in the
  config and passed explicitly rather than inherited.
- **Two reimplemented helpers.** The installed canari predates upstream's
  `SKF.detect_synthetic_anomaly(synthetic_data=, n_jobs=, test_only=)` and its
  `likelihood_covariance_floor`. The detection routine is reimplemented in
  `scripts/common.py` with upstream's exact counting rules; the covariance floor
  defaults to 0.0 upstream, so omitting it is equivalent.

## Data

The same 10 weekly HQ-benchmark series as the companion experiments, loaded
through the same code path, so the three experiments are mutually comparable.

## Metric conventions worth knowing before reading results

Upstream's false-alarm count is **per time step, not per alarm episode**: a single
sustained alarm lasting 40 weeks counts as 40 false alarms. Rates are therefore
much larger than an episode-based count would give, and are not comparable to the
companion experiment `llm_horizon_anomaly`, which counts up-crossings. Both are
defensible; they are simply different, and mixing them would be wrong.

Detection also counts an alarm that was already running when the anomaly began.
With a high false-alarm rate this inflates the detection probability, so the two
columns have to be read together.

## Expected outcome

Unknown, and worth stating plainly rather than guessing: the companion experiments
found that Chronos-2's growing context absorbs a slow drift and hides it from the
switching filter, which at `llm_horizon = 1` left no usable operating point. If
that carries over, the benchmark should show low detection at the small magnitudes
(0.025-0.075 sigma/year) where the LSTM conditions are expected to do their work,
and the honest result is a poor showing rather than a tuned-up one. The sigma_v
and SKF searches are exactly the upstream ones, so the comparison is fair either
way.

## Cost

`python scripts/benchmark.py --dry_run True` prints the plan. With the `cdf`
objective at `llm_horizon = 1` and upstream defaults it is roughly **643 hours**
(27 days): 35.8M Chronos-2 calls, of which 32.4M are the SKF search, because every
trial runs a full detection sweep of `1 + 2 x cdf_num_anomaly = 101` filters.

That protocol is inexpensive upstream, where a recurrent-pattern step is an LSTM
forward pass costing microseconds. A Chronos-2 call is 63 ms, four orders of
magnitude more, so the identical protocol is 27 days rather than an afternoon.
This is a property of the model being benchmarked, not of the benchmark.

`cdf_num_anomaly` is the first reduction to reach for: it changes only the tuning
objective, never the reported results, and TPE tolerates a noisier objective.
Dropping it to 10 costs nothing in fidelity and saves 460 hours. See the run
README for the measured scenario table.
