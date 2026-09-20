# LLM-SKF anomaly detection vs. the re-forecast interval `H`

How well does a Switching Kalman Filter detect a slow drift when Chronos-2 is
queried once every `H` weeks instead of every week?

Pre-registration (question, hypothesis, model, leakage argument, metric
definitions): [`planning/README.md`](planning/README.md).

## Run it

```bash
python scripts/check.py           # pipeline + regression checks
python scripts/tune.py            # choose the frozen configuration, ~1.5 h
python scripts/run_experiment.py  # 10 series x 8 horizons x 11 filters, ~3 h
python scripts/make_figures.py
```

## Headline

**Raw detection probability rises steeply with `H` -- and that is mostly an
illusion.** It climbs from 0.16 at `H = 1` to 0.93 at `H = 26`, which looks like a
clean inversion of the usual expectation. But the false-alarm rate rises with `H`
too, and once each horizon is compared against the detections its *own* false
alarms would produce by chance over a three-year window, the detector's real
contribution is small everywhere and is largest at the **longest** horizons.

| | H=1 | H=3 | H=6 | H=13 | H=26 | H=52 | H=104 | H=208 |
|---|---|---|---|---|---|---|---|---|
| Detection probability | 0.16 | 0.49 | 0.82 | 0.87 | **0.93** | 0.87 | 0.90 | 0.78 |
| Expected from false alarms | 0.15 | 0.31 | 0.66 | 0.64 | 0.78 | 0.70 | 0.55 | 0.42 |
| **Excess over chance** | 0.01 | 0.18 | 0.16 | 0.24 | 0.15 | 0.17 | **0.35** | **0.36** |
| Median time to detection [wk] | 26 | 22 | 20 | 22 | 19 | 20 | 23 | 26 |
| False alarms [1/yr] | 0.07 | 0.33 | 0.58 | 0.60 | **0.88** | 0.68 | 0.54 | 0.23 |

100 realizations per horizon (10 series x 10 anomalies), anomaly 3 sigma/year,
identical realizations at every `H`.

Three things follow.

- **`H = 1` cannot detect at all.** Excess over chance 0.01: the 16% of anomalies
  it flags is exactly what its own false alarms produce anyway. This is the same
  conclusion the pilot reached, now on 100 realizations.
- **The apparent peak at `H = 26` is the false-alarm peak.** Detection probability
  0.93 sounds excellent until you see 0.88 false alarms/year beside it, four times
  the budget the configuration was tuned to. On excess, `H = 26` (0.15) is *worse*
  than `H = 13` (0.24) and half of `H = 208` (0.36).
- **The genuine signal is at long horizons.** `H = 104` and `H = 208` are the only
  horizons where the detector clearly beats its own noise, and `H = 208` does it
  at the lowest false-alarm rate of any horizon above 1 (0.23/year).

Time to detection is flat at 19-26 weeks everywhere, so once an anomaly is caught,
`H` barely affects how fast.

## Figures

| file | content |
|------|---------|
| `figures/pod_vs_horizon.*` | detection probability with its chance level |
| `figures/excess_pod_vs_horizon.*` | the detector's contribution above chance |
| `figures/false_alarms_vs_horizon.*` | false alarms on the same series, no anomaly |
| `figures/time_to_detection_vs_horizon.*` | weeks from onset to alarm |

## Why detection improves with `H` at all

The `Auxiliary` component is fed its own posterior at every filtered step, so a
slow drift enters Chronos-2's context and the model forecasts the drift away --
the residual the switching filter watches carries no anomaly. At `H = 1` the
foundation model re-anchors weekly and erases the signal almost completely; at
`H = 208` it re-anchors about four times across the record, and the drift survives
in the residual long enough to be seen.

The same mechanism explains the false alarms. At small `H` the level and auxiliary
split wanders (~1.8 sigma/year of invented drift at `H = 1` against 0.002 at
`H = 52` on held-out `ts43`), and that wander is itself what trips the filter.

## The configuration does not transfer

The frozen configuration was tuned on held-out series to 0.080 false alarms/year.
On the 10 evaluation series it produces **0.23 to 0.88/year** depending on the
horizon -- 3 to 11 times the tuning value, and above the 0.2/year budget at every
horizon except 1 and 208. Per-series spread is larger still:

| series | false alarms/yr at `H = 26` | excess over chance, best `H` |
|--------|----------------------------|------------------------------|
| ts61 | 1.87 | 0.18 |
| ts57 | 1.60 | 0.50 |
| ts27 | 1.41 | 0.41 |
| ts67 | 0.35 | 0.76 |
| ts26 | 0.11 | 0.90 |

`ts61` is the clearest failure: detection probability 1.00 at six of eight
horizons, and excess of exactly 0.00, because it alarms roughly twice a year
regardless. `ts26` and `ts67` are genuinely detectable. A single frozen
configuration across heterogeneous series is the wrong operating model; the
false-alarm rate needs to be calibrated per series.

## What this says about the original question

The question was how much detection degrades as `H` grows. It does not degrade --
but neither does it straightforwardly improve. Raw detection probability rises
because the detector becomes more trigger-happy, not more discriminating, and only
at `H >= 104` does it earn its alarms. The operational reading is that an LLM-SSM
asked to re-forecast weekly is blind to slow drifts, and that the useful regime is
a re-forecast interval of one to four years, where the foundation model is stale
enough to leave the drift in the residual.

## Caveats

- Detection probability must always be read against the chance column. An earlier
  version of this experiment selected its configuration on raw detection
  probability and chose a setting whose detections were almost entirely its own
  false alarms.
- The anomaly is a single shape (linear drift) at a single magnitude (3 sigma/year,
  calibrated on held-out series). Nothing here speaks to steps, variance changes or
  smaller drifts.
- `H = 1` is outside the model's valid regime rather than merely a poor score: the
  baseline invents drift as large as the injected anomaly there. It is reported,
  not excluded, but its 0.16 should not be read as "detects 16% of anomalies".
- The alarm threshold is fixed at 0.5 and was not tuned; a threshold sweep during
  the pilot showed it cannot separate signal from noise at `H = 1`, because the
  false alarms sit at `Pr ~ 1` as well.
