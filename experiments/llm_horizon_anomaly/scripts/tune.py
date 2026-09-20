"""Choose the SKF configuration, on series that are never scored.

H=1 turned out to have no usable operating point: the auxiliary component's
growing context lets Chronos-2 learn a slow drift and forecast it away, so the
residual the switching filter watches carries no signal. The only settings that
fire at H=1 fire 1.6-5.5 times a year on clean data at any threshold, and every
quieter setting never detects. Tuning there would freeze a blind configuration
into every horizon.

So the configuration is chosen at the **lowest horizon that has a real operating
point**, in three stages, all on the held-out tuning series:

1. screen the whole parameter grid at one cheap horizon and keep the finalists;
2. scan horizons upward from H=1, stopping at the first one where a finalist
   both meets the false-alarm budget and detects at least half the anomalies;
3. with those parameters frozen, pick the anomaly magnitude whose detection
   probability lands nearest the target.

    python scripts/tune.py
"""

import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def grid_combinations():
    keys = list(common.PARAM_GRID)
    return [
        dict(zip(keys, values)) for values in itertools.product(*common.PARAM_GRID.values())
    ]


def evaluate(combos, horizon, slope_per_year, label):
    """Run every combo at one horizon and summarize per combo."""

    tasks = [
        common.make_task(
            series,
            horizon,
            params,
            slope_per_year / common.WEEKS_PER_YEAR,
            common.TUNE_REALIZATIONS,
            combo=index,
        )
        for index, params in combos
        for series in common.TUNE_SERIES
    ]
    print(f"{label}: {len(combos)} parameter sets x {len(common.TUNE_SERIES)} series "
          f"at H={horizon} ({len(tasks)} cases)")
    detections, false_alarms = common.run_all(tasks)
    table = common.summarize(detections, false_alarms, by="combo").reset_index()
    table["horizon"] = horizon
    lookup = dict(combos)
    for key in common.PARAM_GRID:
        table[key] = [lookup[index][key] for index in table["combo"]]
    return table


def viable(table):
    """Configurations that meet the false-alarm budget and actually detect."""

    return table[
        (table["false_alarms_per_year"] <= common.MAX_FALSE_ALARMS_PER_YEAR)
        & (table["excess_pod"] >= common.MIN_EXCESS_POD)
    ]


def screen(run_dir, combos):
    """Rank the whole grid cheaply, keep the most promising handful."""

    table = evaluate(combos, common.SCREEN_HORIZON, common.PROVISIONAL_SLOPE_PER_YEAR,
                     "screen")
    table.to_csv(run_dir / "screen.csv", index=False)
    print("\n" + table.sort_values("excess_pod", ascending=False).to_string(index=False)
          + "\n")

    ranked = table.sort_values(
        ["excess_pod", "false_alarms_per_year"], ascending=[False, True]
    )
    within_budget = ranked[
        ranked["false_alarms_per_year"] <= common.MAX_FALSE_ALARMS_PER_YEAR
    ].sort_values("excess_pod", ascending=False)
    chosen = pd.concat([within_budget, ranked]).drop_duplicates("combo")
    finalists = chosen.head(common.NUM_FINALISTS)["combo"].tolist()
    print(f"finalists: {finalists}\n")
    lookup = dict(combos)
    return [(index, lookup[index]) for index in finalists]


def scan(run_dir, finalists):
    """Walk horizons upward until one of them supports a real detector."""

    tables = []
    for horizon in common.SCAN_HORIZONS:
        table = evaluate(finalists, horizon, common.PROVISIONAL_SLOPE_PER_YEAR,
                         f"scan H={horizon}")
        tables.append(table)
        print("\n" + table.to_string(index=False) + "\n")
        candidates = viable(table)
        if len(candidates):
            best = candidates.sort_values(
                ["excess_pod", "median_weeks_to_detect"], ascending=[False, True]
            ).iloc[0]
            pd.concat(tables).to_csv(run_dir / "scan.csv", index=False)
            params = dict(finalists)[int(best["combo"])]
            print(f"lowest viable horizon: H={horizon}")
            print(f"  parameters {params}")
            print(f"  POD {best['pod']:.2f} (chance {best['chance_pod']:.2f}, "
                  f"excess {best['excess_pod']:.2f})  "
                  f"FA/yr {best['false_alarms_per_year']:.3f}  "
                  f"median TTD {best['median_weeks_to_detect']:.0f} weeks")
            return horizon, params
        print(f"H={horizon}: no configuration meets "
              f"FA<={common.MAX_FALSE_ALARMS_PER_YEAR}/yr with excess POD>="
              f"{common.MIN_EXCESS_POD}; going up.\n")

    pd.concat(tables).to_csv(run_dir / "scan.csv", index=False)
    raise SystemExit(
        f"No horizon in {common.SCAN_HORIZONS} supports a detector within budget. "
        "That is itself a result: report it rather than forcing a configuration."
    )


def tune_magnitude(run_dir, horizon, params):
    tasks = [
        common.make_task(
            series,
            horizon,
            params,
            slope_per_year / common.WEEKS_PER_YEAR,
            common.TUNE_REALIZATIONS,
            slope_per_year=slope_per_year,
        )
        for slope_per_year in common.MAGNITUDE_GRID_PER_YEAR
        for series in common.TUNE_SERIES
    ]
    print(f"\nmagnitude: {len(common.MAGNITUDE_GRID_PER_YEAR)} values at H={horizon} "
          f"({len(tasks)} cases)")
    detections, false_alarms = common.run_all(tasks)
    table = common.summarize(detections, false_alarms, by="slope_per_year").reset_index()
    table.to_csv(run_dir / "magnitude.csv", index=False)
    print("\n" + table.to_string(index=False))

    table["distance"] = (table["excess_pod"] - common.TARGET_POD).abs()
    chosen = table.sort_values("distance").iloc[0]
    print(f"\nchosen magnitude: {chosen['slope_per_year']:.2f} sigma/year "
          f"(excess POD {chosen['excess_pod']:.2f}, target {common.TARGET_POD})")
    return float(chosen["slope_per_year"])


def main():
    started = time.time()
    run_dir = common.EXP_DIR / "results" / datetime.now().strftime("tuning_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)

    combos = list(enumerate(grid_combinations()))
    finalists = screen(run_dir, combos)
    horizon, params = scan(run_dir, finalists)
    slope_per_year = tune_magnitude(run_dir, horizon, params)

    tuned = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "tuned_at_horizon": horizon,
        "scan_horizons": common.SCAN_HORIZONS,
        "screen_horizon": common.SCREEN_HORIZON,
        "tuning_series": common.TUNE_SERIES,
        "tuning_realizations": common.TUNE_REALIZATIONS,
        "max_false_alarms_per_year": common.MAX_FALSE_ALARMS_PER_YEAR,
        "min_excess_pod": common.MIN_EXCESS_POD,
        "target_pod": common.TARGET_POD,
        "alarm_threshold": common.ALARM_THRESHOLD,
        "params": params,
        "slope_per_year": slope_per_year,
        "slope_per_week": slope_per_year / common.WEEKS_PER_YEAR,
        "tuning_run": run_dir.name,
    }
    common.TUNED_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    common.TUNED_PARAMS_FILE.write_text(json.dumps(tuned, indent=2))
    (run_dir / "tuned_params.json").write_text(json.dumps(tuned, indent=2))
    print(f"\nwrote {common.TUNED_PARAMS_FILE}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
