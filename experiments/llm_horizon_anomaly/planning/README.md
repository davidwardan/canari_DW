# Anomaly detection with an LLM-SSM as the re-forecast interval `H` grows

## Research question

The companion experiment (`experiments/llm_horizon_degradation`) measured how
*forecasting* accuracy decays when Chronos-2 is queried once every `H` weeks
instead of every week. This one asks the operational question:

**How much detection power is lost as `H` grows from 1 to 208 weeks?**

Measured three ways: whether a drift anomaly is caught at all, how long it takes
to catch it, and how often the detector cries wolf on data with no anomaly.

## Hypothesis

**Revised after a pilot; the original prediction is kept below because it was
wrong in an informative way.**

*Original:* detection probability falls and time to detection rises with `H`,
because a stale forecast is a worse reference signal.

*Revised:* detection **improves** with `H`. The `Auxiliary` component is fed its
own posterior at every filtered step, so a slow drift enters Chronos-2's context
and the model simply forecasts the drift -- the residual that the switching filter
watches carries no anomaly at all. Only when `H` is large enough that Chronos-2
re-anchors a handful of times across the record does the drift survive in the
residual. False alarms should fall with `H` for the same reason: the level and
auxiliary split wanders more when the foundation model re-anchors every week.

Pilot evidence is in the Model section below: with a correctly parameterized
filter, a 2 sigma/year drift is caught 3 times out of 3 at `H = 13` and `H = 52`,
and 0-1 times out of 3 at `H = 1`.

At `H = 1` there is no operating point, for the structural reason given under
Model: the baseline invents ~1.8 sigma/year of drift there, as large as the
anomaly itself.

## Data

The same 10 weekly HQ-benchmark series as the forecasting experiment, loaded
through the same code (`llm_horizon_degradation/scripts/common.py`): the processed
`weekly_values.csv`, each series re-standardized on its own 52-week warmup, the
warmup handed to the `Auxiliary` component as context and never filtered.

## Model

```
y_t = level_t + trend_t (+ acceleration_t if abnormal) + auxiliary_t + white-noise_t
```

- **normal**: `LocalTrend` + `Auxiliary(H)` + `WhiteNoise` -- level and a slowly
  varying slope.
- **abnormal**: `LocalAcceleration` + `Auxiliary(H)` + `WhiteNoise` -- the slope is
  allowed to change.

This is canari's documented drift-detection pair. Two corrections were needed to
make it behave when the recurrent-pattern slot holds a frozen foundation model
rather than a trained LSTM.

**The trend is reset to zero after initialization.** `initialize_from_context`
seeds the trend state with the slope fitted to the warmup, and because the
auxiliary can represent any signal, no residual ever pushes that slope back: the
level integrates it for the rest of the record. On `ts43` the warmup slope is
+0.0149/week, and the level duly ran away to +12 over 896 *clean* steps while the
auxiliary went to -12 -- summing to the right observation, so forecasts looked
perfect while the decomposition was meaningless. On `ts53`, whose warmup slope
happens to be -0.00000/week, the identical model never drifted at all, which is
what identified the cause. Since the benchmark data is already detrended, "no
drift" is also the right prior for the normal regime. `scripts/check.py` guards
this as a regression check.

**The switching prior has to be much smaller than canari's default.** At
`norm_to_abnorm_prob = 1e-4` and above the filter fires 1-2 times a year on clean
data; 1e-6 to 1e-5 gives a real operating point. The grid was originally centred
on 1e-3 and found nothing, which is why the first tuning run returned a
configuration that never detected.

With both corrections, on held-out `ts43` with a 2 sigma/year drift:

| `H` | false alarms/yr | clean-data drift | detection |
|-----|-----------------|------------------|-----------|
| 52 | 0.058 | 0.22 sigma/yr | 3/3, median 28 weeks |
| 13 | 0.116 | 0.15 sigma/yr | 3/3, median 30 weeks |
| 1 | 0.232 | **1.82 sigma/yr** | 0-1/3 |

`H = 1` remains unusable, and for a structural reason worth stating: the level and
auxiliary split wanders there whatever the parameters, because Chronos-2
re-anchors every step, and at ~1.8 sigma/year the model invents drift as large as
the anomaly being injected. That is a property of coupling a live-context
foundation model to a switching baseline, not a tuning failure.

The `Auxiliary` instance is shared by all four transition models -- the SKF queries
it once per time step and hands the same forecast to each -- so the `H` block
semantics are exactly those of the forecasting experiment.

## Anomalies

One anomaly per realization, 10 realizations per series, injected with
`DataProcess.add_synthetic_anomaly`: a linear drift starting at a uniformly drawn
point in the middle third of the evaluation segment.

**Realizations are identical across `H`.** The onsets are drawn from a fixed seed
and depend only on the segment length, so every horizon is scored on exactly the
same 100 anomalies. The comparison across `H` is paired, and the differences
cannot come from a different draw.

The magnitude is calibrated (see below), not guessed: a magnitude that everything
detects, or that nothing detects, would show no `H` effect whatever the truth.

## Tuning, and why it cannot leak

Everything that has to be chosen -- the four SKF parameters and the anomaly
magnitude -- is chosen on **held-out series** (`ts43`, `ts69`, `ts53`), which are
eligible for the benchmark but are not among the 10 scored series. The 10
evaluation series are never filtered during tuning.

The original plan was to tune at `H = 1` and freeze. That is not viable: `H = 1`
has no operating point (see Hypothesis), so the selection rule could only return a
configuration that never detects, and freezing it would blind every other horizon.
Instead the configuration is chosen at the **lowest horizon that has a real
operating point**, found by scanning upward, and then frozen for every `H`:

1. **Screen** the 12-point grid over `norm_to_abnorm_prob`, the `LocalAcceleration`
   process noise and `std_transition_error`, at one cheap horizon;
   keep the 4 most promising. The grid ranges matter: `norm_to_abnorm_prob` must
   reach down to 1e-6, well below canari's 1e-4 default, and the abnormal model's
   acceleration noise sets sensitivity against false alarms.
2. **Scan** `H` upward through 1, 3, 6, 13, 26 with those finalists, stopping at
   the first horizon where a configuration both meets the false-alarm budget and
   beats its own chance level by at least 0.3.
3. **Magnitude**: with those parameters frozen, the grid value in
   {1, 2, 3, 4} sigma/year whose detection probability lands nearest 0.8.

The false-alarm budget is 0.2/year, about 3 false alarms across the ~17-year
record. It was relaxed from 0.1/year after no configuration met that at any
horizon tested; the change is recorded here rather than presented as
pre-registered. The alarm threshold stays at canari's default of 0.5: a threshold
sweep to 0.9999 showed it cannot separate signal from noise, because the false
alarms sit at `Pr ~ 1` as well.

The result is frozen in `data/tuned_params.json` and reused verbatim for every `H`.

## Metrics

Per horizon, pooled over the 10 series x 10 realizations:

- **Probability of detection**: the fraction of realizations where an alarm
  *starts* within **three years (156 weeks)** of the onset. An alarm that was
  already running when the anomaly began cannot be attributed to it and does not
  count; that matters here, because the pre-onset stretch is ~8 years and at a
  false-alarm rate of 0.185/year some 77% of realizations carry one.
- **Chance level and excess**: at false-alarm rate `r`, a three-year window shows
  an alarm with no anomaly present at all with probability `1 - exp(-3r)`. At
  0.185/year that is 43%, so a raw detection probability of 0.67 is far less
  impressive than it looks. Every horizon therefore reports the chance level
  implied by its own measured false-alarm rate, and the **excess over chance** is
  what is compared across `H` and what the tuning maximizes. An early version of
  this experiment selected on raw detection probability and chose a configuration
  whose detections were almost entirely its own false alarms.
- **Time to detection**: weeks from onset to the first qualifying alarm, among
  detected realizations. Median with interquartile range; censored at 156 weeks by
  construction, so it must be read together with the detection probability.
- **False-alarm rate**: measured by filtering **the same series with no anomaly
  injected**, counting threshold up-crossings (a sustained alarm is one alarm, not
  one per week) and dividing by the length of the segment in years.

`pre_onset_alarm_rate` and `alarm_active_at_onset_rate` are reported alongside so
the amount of contamination is visible rather than implicit.

## Reproducibility

`scripts/tune.py` then `scripts/run_experiment.py`, each writing a timestamped
directory under `results/` with `config.json` (settings, tuned values, package
versions, seed), per-realization records, and summaries. Nothing is overwritten.
The 880 filter runs of the sweep are independent and run across 6 processes.

## Expected outcome

Detection probability **rising** with `H`, time to detection falling, and false
alarms falling -- the inverse of the original hypothesis. The operational reading
would be that an LLM-SSM asked to re-forecast often adapts to a slow drift and
hides it, and that re-forecasting *less* often is what makes the drift visible.

The risk to watch is the floor: if detection probability is 0 for the first
several horizons, the informative part of the curve is compressed into the top
two or three. The magnitude calibration is what guards against that.

## Pipeline check

`scripts/check.py` asserts the things that would silently corrupt the sweep:
filtering the same data twice across a reset gives bit-identical probabilities
(no context leaks between realizations); the onsets are identical for every `H`
and always leave room for the 3-year window; the injected drift is exactly zero
before the onset and exactly linear after it; the scoring rules behave on
probability traces whose answer is obvious; and a large anomaly is detected while
a clean series is quiet. Run it before tuning.
