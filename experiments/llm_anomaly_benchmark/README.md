# LLM anomaly-detection benchmark

Chronos-2 dropped into the recurrent-pattern slot of the upstream SKF anomaly
benchmark, so it can be scored in the same table as the local, global-finetune and
global-zeroshot LSTM conditions.

**Status: the H=1 pilot has run (see below); the horizon sweep has not.** The
sweep machinery is validated end to end on a reduced config.

The sweep is the point: every `H` in `llm_horizon_search_space` is tuned
independently (its own sigma_v grid and cdf search) and then evaluated, so each
horizon reports its best achievable operating point. Differences across `H`
therefore mix the horizon with its retuning -- that is the intended reading.

Design, and the full list of what is held identical to upstream versus what
necessarily differs: [`planning/README.md`](planning/README.md).

## Run it

```bash
# What it will cost, without running anything
python scripts/benchmark.py --dry_run True

# The prepared full sweep: 10 series x 8 H, bolt-tiny, sized for 48 cores
python scripts/benchmark.py --config_path config/full_bolt_tiny.yaml

# The single-series pilot
python scripts/benchmark.py --config_path config/pilot_ts58_bolt_tiny.yaml

# The full 10-series benchmark, then figures
python scripts/benchmark.py
python scripts/make_figures.py
```

### The global-LSTM condition

Upstream's `global_finetune` condition, fine-tuned per series from the global
model `saved_params/global_model.bin` (1 x 50, lookback 52), on the same series and
realizations as the LLM sweep. There is no `H`; the cases are (seed, series) and
land in `<run>/seed<s>/<series>/`. Design and differences:
[`planning/README.md`](planning/README.md#the-global-lstm-condition-global_finetune).

`results/full_local` is an earlier run of the same pipeline started from the
256-unit `seed_variability` weights instead (`config/full_local.yaml`). It is kept
under that name so it is not mistaken for the global model.

`plot_global_vs_llm.ipynb` compares a run with the LLM sweep at one `H`
(`GLOBAL_RUN`, `LLM_H`).

```bash
# Assert it sees exactly the LLM sweep's data and realizations
python scripts/check_same_inputs.py \
    --llm_run results/full_chronos2_small_zerovar_residual \
    --config_path config/full_global_finetune.yaml

# The 10 series, 10 cases x 4 CPUs (~35 min)
python scripts/benchmark.py --config_path config/full_global_finetune.yaml \
    --output_dir results/full_global_finetune
```

Each case also checkpoints its trained network (`trained_model.pkl`) before the
SKF search, which loads it in every trial. Extra `seeds` only change anything with
one weights file per seed (`{seed}` in `lstm_global_params`); with the single
`global_model.bin` they would repeat the same run.

**Tuning is checkpointed.** The SKF search is the expensive half (11.6 h of the
pilot's 17.5 h), so its result is written to `tuned_params.json` before the
evaluation starts. Re-running against the same output directory reuses it instead
of re-searching, and each magnitude is appended to `evaluation.partial.json` as it
finishes, so a late failure costs only the tail.

Everything is driven by [`config/benchmark.yaml`](config/benchmark.yaml), whose key
names mirror the upstream config so the two can be diffed. Values are the upstream
defaults unless a comment says otherwise.

## Before you start it: the cost

With `skf_objective_function: cdf` at `llm_horizon: 1` and upstream defaults:

| stage | filters | Chronos-2 calls | hours |
|---|---|---|---|
| sigma_v grid | 90 | 57,744 | 1.0 |
| SKF search | 50,500 | 32,400,800 | 581.4 |
| evaluation | 3,570 | 3,354,372 | 60.2 |
| **total** | **54,160** | **35,812,916** | **~643 (27 days)** |

**The `cdf` objective is what makes this expensive.** Every trial runs a whole
detection sweep -- one clean filter plus `2 x cdf_num_anomaly` realizations -- so
50 trials x 101 filters = 5,050 filters per series. That is cheap upstream, where a
recurrent-pattern step is an LSTM forward pass costing microseconds. Here each
step is a 63 ms Chronos-2 call, four orders of magnitude more, so the same
protocol costs 27 days instead of an afternoon.

`python scripts/benchmark.py --dry_run True` prints this for whatever config is in
place. Reduction paths, all measured with the same model:

| scenario | sigma_v | SKF search | eval | total | days |
|---|---|---|---|---|---|
| upstream defaults | 1.0 | 581.4 | 60.2 | **642.6 h** | 26.8 |
| `cdf_num_anomaly: 10` | 1.0 | 120.9 | 60.2 | **182.1 h** | 7.6 |
| + `num_optimization_trial: 25` | 1.0 | 60.4 | 60.2 | **121.7 h** | 5.1 |
| + `num_anomaly_realizations: 10` | 1.0 | 60.4 | 24.8 | **86.3 h** | 3.6 |
| + `llm_horizon: 13` | 0.1 | 6.0 | 2.5 | **8.6 h** | 0.4 |

`cdf_num_anomaly` is the first knob to reach for: it only affects the *tuning*
objective, not the reported results, and TPE copes with a noisier objective. The
last row is the only one that changes what is being measured, because
`llm_horizon: 1` is what makes this a like-for-like comparison with the LSTM.

## Four things to know before reading any results

1. **Upstream counts false alarms per time step, not per alarm episode.** One
   sustained alarm lasting 40 weeks counts as 40. Rates are therefore much larger
   than an episode-based count, and are **not comparable** to the companion
   experiment `llm_horizon_anomaly`, which counts up-crossings.
2. **Detection counts an alarm that was already running at the anomaly onset.**
   With a high false-alarm rate that inflates P(detect), so the detection and
   false-alarm columns must be read together.
3. **The objective is `cdf`**, so the threshold is genuinely tuned against
   detection rate and false-alarm rate, and `slope` is tuned alongside it -- the
   optimizer also chooses which anomaly magnitude to tune for, from
   `slope_search_space`. Note the contrast if you ever switch to `ll`: that
   objective does not depend on the threshold at all, so the threshold carried into
   evaluation would be whichever the best-likelihood trial happened to draw. A
   three-trial check showed `ll` scoring 0.98130 / 0.98137 / 0.98135 at thresholds
   0.44 / 0.19 / 0.33 -- indistinguishable.
4. **`SKF.objective`'s shape parameters are pinned in the config, not inherited.**
   The installed canari ships different defaults from upstream
   (`detection_rate_cdf_std` 0.5 vs 0.2, `anm_mag_cdf_median` 0.3 vs 0.2,
   `anm_mag_cdf_shape` 0.4 vs 0.6). They are passed explicitly so the search scores
   the same way upstream's does.

## Result so far: the H=1 pilot

`results/run_20260920_172931`, ts58, Chronos-Bolt-tiny, 80 min, 350 realizations:

| magnitude (sigma/yr) | 0.025 | 0.05 | 0.075 | 0.225 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|
| P(detect) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| False alarms/yr | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Zero detections **and** zero false alarms everywhere: the filter never crosses the
threshold at all. The tuned threshold was 0.11, so `Pr(abnormal)` never exceeds
0.11 anywhere in the record. The detector is inert at `H = 1`, not merely weak.

**The tuning that produced it was degenerate, and the result should be labelled
untuned.** All 50 cdf trials reported `detection_rate: 0.0` and
`false_alarm_rate: 0.0`, so the objective reduced to its third term:

```
j1 = norm.cdf(0.0, loc=0.5, scale=0.2)   -> constant
j2 = 1 - lognorm.cdf(0.0, ...)            -> constant
j3 = 1 - lognorm.cdf(slope, ...)          -> the only term that varies
```

The search therefore optimized `slope` alone, driving it to the grid minimum
(0.025); the winning trial's `threshold = 0.11` and switching parameters are
whatever that trial happened to sample. The cause is the magnitude range: upstream
tops out at 1.0 sigma/year, which is 3 sigma accumulated over the 156-week window,
while the companion experiments found Chronos at `H = 1` barely detects a *3
sigma/year* drift -- three times larger. Nothing in this space is detectable, so
there is nothing for the tuner to discriminate on.

The sigma_v grid hit its boundary too: validation CRPS fell monotonically across
the whole range and selected the smallest value, 0.02. The optimum is at or below
upstream's grid floor, so sigma_v is being clamped rather than found.

Both are expected to resolve at larger `H`, which is what the sweep will show.

## The prepared full sweep

`config/full_bolt_tiny.yaml` -- 10 series x 8 horizons = 80 cases, Chronos-Bolt-tiny,
100 SKF trials with 50 startup trials, `cdf_num_anomaly` left at upstream's 50.

Sized for a 48-core server with 4 left free:

| setting | value | why |
|---|---|---|
| `cpus` | 44 | 48 minus 4 kept free |
| `max_concurrent` | 11 | 4 CPUs per case |
| `ray_object_store_mb` | 256 | Ray's default is ~30% of system RAM, **per instance** |

**571 core-hours, ~16 h wall.** `max_concurrent: 11` rather than the 22 that pure
speed would choose: wall clock is nearly flat across that range (15.4 h at 22
against 15.9 h at 11) while each concurrent case runs its own local Ray instance,
so halving the count cuts peak RAM from roughly 29 GB to 22 GB. Rough memory
estimate is `concurrent x (0.6 GB Ray head + 0.35 GB per CPU)`; measure it early
rather than trusting it.

| cores | cases in parallel | CPUs each | wall |
|---|---|---|---|
| 16 | 11 | 1 | 54.1 h |
| 32 | 11 | 2 | 30.0 h |
| **44** | **11** | **4** | **15.9 h** |
| 64 | 11 | 5 | 12.9 h |

The SKF search is 542 of the 571 core-hours. Halving `num_optimization_trial` or
setting `cdf_num_anomaly: 10` (which only makes the tuning objective noisier and
never touches reported results) would cut the run to roughly 8 h or 4 h.

## Parallelism

The (H, series) cases are independent, so the driver runs them side by side and
gives each a slice of the cores for its own sigma_v pool, Ray trial concurrency
and evaluation pool. Ray is explicitly bounded to that slice -- left alone it sizes
itself from the whole machine, and concurrent cases would each claim every core.

Splitting matters more than it looks. **Every horizon runs the same number of
filters**; only the Chronos calls per filter scale as `1/H`. So `H = 1` costs far
more than `H = 208`, and running one case per core leaves that single case setting
the wall clock. The driver therefore picks the split that minimises the estimated
makespan rather than maximising case concurrency, capped by how many cores one
case can actually keep busy (its widest stage is the SKF search, so roughly
`num_optimization_trial`). Override with `max_concurrent`.

`python scripts/benchmark.py --dry_run True --cpus 64` prints core-hours and the
projected wall clock. With Chronos-Bolt-tiny:

| workload | core-hours | 8 cores | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| 1 series x 8 H | 29 | 4.4 h | 2.2 h | 1.1 h | 0.6 h | 0.3 h |
| 10 series x 8 H | 300 | 38.4 h | 19.6 h | 10.9 h | 5.8 h | 2.9 h |
| 10 series x 8 H, `cdf_num_anomaly: 10` | 85 | 10.9 h | 5.5 h | 3.1 h | 1.6 h | 0.8 h |

To shard across machines instead, restrict the horizons:

```bash
python scripts/benchmark.py --horizons_subset '[1]'        --output_dir out/sweep
python scripts/benchmark.py --horizons_subset '[3,6,13]'   --output_dir out/sweep
python scripts/benchmark.py --horizons_subset '[26,52,104,208]' --output_dir out/sweep
```

Each writes to `out/sweep/H{h}/{series}/`, so the shards merge into one tree and
`make_figures.py` reads it as a single run.

## Running on another machine

- `num_workers: null` resolves to `CPU count - 2` for a single-case run, and is
  set automatically per case when the driver parallelizes.
- The `--dry_run` core-hours come from single-process latency measured on an Apple
  M2 (Bolt-tiny 10.5 ms/call, Bolt-base 147.9 ms, Chronos-2 277.7 ms, SKF step
  1.6 ms). A Linux server will differ -- pass `--cpus N` to project the wall clock,
  and re-time one case before trusting the absolute numbers.
- Chronos-2 on the M2 was memory-bandwidth bound and barely scaled with processes.
  Bolt-tiny is 9M parameters and much lighter, so it should scale considerably
  better on a many-core Linux box; that is an expectation, not a measurement.
- `spawn` is used for all pools rather than Linux's default `fork`, which is
  unsafe alongside torch threads.
- Model weights download from Hugging Face on first use
  (`chronos-bolt-tiny` 33 MB, `chronos-bolt-base` 783 MB, `chronos-2` 456 MB), so
  the machine needs network access or a warm `~/.cache/huggingface`.
- Requires `chronos-forecasting`, `torch`, `ray`, `optuna`, `fire`, `pyyaml` and
  this repo's `canari` on the path.
- On Linux with CUDA PyTorch, put its bundled NCCL ahead of the system library
  before starting Python. Otherwise importing `canari` first in Ray workers can
  load an older system NCCL and make PyTorch fail with
  `undefined symbol: ncclCommResume`. From the repository root:

  ```bash
  conda activate canari
  nccl_lib=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/nccl/lib")')
  LD_LIBRARY_PATH="$nccl_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    python -u experiments/llm_anomaly_benchmark/scripts/benchmark.py \
    --config_path experiments/llm_anomaly_benchmark/config/full_bolt_tiny.yaml \
    --output_dir experiments/llm_anomaly_benchmark/results/sweep_bolt_tiny
  ```

- Ray writes trial state to `~/ray_results`; it can grow over a long sweep.
- Tuning is checkpointed per `(H, series)`, so an interrupted sweep resumes by
  re-running the same `--output_dir`.

## Layout

```
config/benchmark.yaml   every setting, mirroring the upstream config keys
scripts/common.py       data prep, model construction, Chronos-2 predictor,
                        and upstream's detection rules reimplemented
scripts/run_series.py   one (series, H): sigma_v grid -> SKF search -> evaluation
scripts/benchmark.py    sweeps H x series (seed x series for the LSTM),
                        aggregates, --dry_run cost plan
scripts/make_figures.py figures and LaTeX table from a finished run
scripts/check_same_inputs.py  asserts the LSTM config sees the LLM sweep's data
```

Results land in a timestamped `results/run_*/` with `benchmark_summary.json`, a
`summary.json` per series, and the derived CSV and `.tex` tables.

## Relationship to the companion experiments

- `llm_horizon_degradation` -- how forecast accuracy decays as `llm_horizon` grows.
- `llm_horizon_anomaly` -- how detection behaves as `llm_horizon` grows, with an
  episode-based false-alarm count and a chance-corrected detection probability.
- this one -- the upstream benchmark's protocol exactly, at a fixed `llm_horizon`,
  so the foundation model can be scored against the LSTM conditions.

The three share series selection, warmup standardization and the Chronos-2
predictor through `llm_horizon_degradation/scripts/common.py`, so they run on
identical data.
