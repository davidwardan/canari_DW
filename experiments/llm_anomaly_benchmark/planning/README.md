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
| sigma_v selection | smallest within `delta` of best validation CRPS | same rule, but on best validation **log-likelihood** (`early_stopping_metric: ll`) |
| SKF search space | `std_transition_error` loguniform `[5e-6, 1e-4]`, `norm_to_abnorm_prob` loguniform `[1e-5, 1e-4]`, `abnorm_to_norm_prob` quniform `[0.1, 0.2]`, `threshold` quniform `[0.05, 0.5]` | same, except **`std_transition_error` up to `1e-3`** |
| SKF search | Ray Tune + Optuna TPE, 50 trials, 20 startup | same |
| SKF objective | `cdf`: maximize `SKF.objective(detection_rate, false_rate, slope)` on train+validation | same, except **0 when `detection_rate == 0`** |
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
  (`validation_ratio`, `test_ratio`) applied to each series after its 52-week
  warmup: train 35% / validation 15% / test 50%, so train+validation and test are
  equal in length. Each must span at least 4 years, or the series is rejected.
- **One extra knob, `llm_horizon`.** The number of filtered steps between Chronos-2
  calls. `1` is the fair comparison, matching the LSTM's per-step cadence, and is
  the default. It is exposed because the companion experiments found it to matter
  enormously.
- **`SKF.objective` defaults.** The installed canari ships different defaults from
  upstream for three of the objective's shape parameters, so they are pinned in the
  config and passed explicitly rather than inherited.
- **The `cdf` objective scores 0 when nothing is detected.** Upstream's
  `j1 = norm.cdf(detection_rate, 0.5, 0.2)` is 0.0062 at zero detections, and `j3`
  rewards small slopes, so "detect nothing at 0.025/yr" (0.0062) outscores "detect
  everything at 1.0/yr" (0.0037). In `results/sweep_bolt_tiny` 26 of 29 cases
  settled on exactly that trial and evaluated to P(detect) = 0 at every magnitude;
  the 3 that found a detecting trial reached P(detect) ~ 1 at 0.75-1.0/yr.
- **`std_transition_error` searched up to `1e-3`, not `1e-4`.** The detecting
  trials in that sweep sat near the old upper bound (~8e-5).
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

## The global-LSTM condition (`global_finetune`)

**Question.** On the same series and the same synthetic realizations, how does
upstream's global LSTM, fine-tuned per series, compare with the foundation model?

**Hypothesis.** The LSTM is trained on each series' own training split, so it
should track the recurrent pattern closely there. It runs at every step, so there
is no `H`, and a drift cannot hide inside a growing context the way it does for
Chronos. Detection at the small magnitudes should therefore be at least as good as
Chronos at its best `H`. That is a hypothesis, not a measurement.

Config: `config/full_global_finetune.yaml`, which starts from the global model
`saved_params/global_model.bin`. The cases are (seed, series), not (H, series),
and results go to `<run>/seed<s>/<series>/`.

**Four runs, do not mix them up:**

| results | starting weights | network |
|---|---|---|
| `full_global_finetune` | `saved_params/global_model.bin`, the global model | 1 x 50 |
| `full_global_finetune_256` | `saved_params/global_52_256.bin`, the global model (`config/full_global_finetune_256.yaml`) | 1 x 256 |
| `full_global_finetune_256_trials150` | as `full_global_finetune_256`, with 150 SKF trials (100 startup) instead of 100 (50) (`config/full_global_finetune_256_trials150.yaml`) | 1 x 256 |
| `full_local` | `/home/dw/canari/saved_params/seed_variability/model_256_52_seed1.bin` | 1 x 256 |

`full_local` ran first, from the wrong starting point. It is kept under that name
at the user's request, with a README inside saying what it holds. It is not
upstream's `local` condition, which is a 1 x 64 LSTM from random initialization.

**Held identical to the LLM sweep** (`full_chronos2_small_zerovar_residual`):
series, warmup, splits, standardization of `y`, the sigma_v grid, the SKF search
space, trials and objective, the evaluation magnitudes and every realization.
`scripts/check_same_inputs.py` asserts this series by series against that run's
`config_used.json`. It checks identical `y` in every split, identical split
indices, and identical onsets and values for both the evaluation and the
cdf-tuning realizations.

**Follows upstream's `global_finetune`** (`benchmark_anomaly_detection.py`):

| | upstream | here |
|---|---|---|
| Weights | `seed_variability/model_256_52_seed{seed}.bin`, one per seed | `saved_params/global_model.bin`, one file |
| Network | 1 LSTM layer x 256, lookback 52, `week_of_year` covariate, no smoother | 1 x **50** (what the file holds), otherwise same |
| Transfer | `increase_output_variance`: pretrained means, initialization variances | same, reimplemented (the installed canari predates it) |
| Warmup | 52 standardized rows as the lookback, variance 0.1 | same |
| Training | per sigma_v candidate, `lstm_train` each epoch, early stopping (patience 20, `skip_epoch=0`), up to 500 epochs | same |
| Early-stopping metric | open-loop validation forecast, LL here | same |
| SKF normal model | the trained model at t=0 (its smoothed states, trend not re-zeroed) | same |
| Date shift | `date_shift_weeks: 0` in the upstream configs | no shift |

**Differs from upstream:**

- **Baseline-state variances.** Both conditions run on the installed canari,
  whose `auto_initialize_baseline_states` uses level/trend variances of 1e-4/1e-5
  against 1e-3/1e-6 on upstream's branch. This keeps the two conditions here
  consistent with each other, not with upstream.
- **SKF search concurrency.** Upstream hard-sets one trial at a time. Here Ray runs
  4 trials side by side per case, as in the LLM sweep, so TPE behaves the same way
  in both conditions.
- **sigma_v is scored differently from the LLM.** The LLM is scored by a one-step
  filter over train+validation. The LSTM uses upstream's open-loop validation
  forecast at the early-stopping epoch. Both keep upstream's selection rule.

**Seeds.** pytagi's initial *variances* do not depend on the seed; only the
means do, and the means are overwritten by the pretrained ones. So the seed acts
only through which weights file is loaded. With the single `global_model.bin`,
more seeds would repeat the same run, so it uses seed 1 only.

**Open point.** Whether the global weights were pretrained on data that overlaps
these series' test periods is not recorded here. It has to be confirmed before
the comparison is reported, because an overlap would be test-set leakage.

**Observed in diagnostics before the runs** (ts58, sigma_v = 0.1, validation
LL and RMSE of the open-loop forecast):

| starting weights | no weight updates | after 1 epoch | after 2 epochs |
|---|---|---|---|
| `global_model.bin` (1 x 50) | -4.42 / 2.32 | -1.98 / 0.64 | -3.44 / 0.64 |
| seed_variability seed 1 (1 x 256) | -0.93 / 0.53 | -2.02 / 0.67 | -4.51 / 0.85 |

From `global_model.bin`, fine-tuning is what makes the network usable on this
series. From the 256-unit weights, it degrades a network that was already good,
so a zero-shot condition would likely beat `full_local`.
