"""Drive the LLM anomaly-detection benchmark across all series and aggregate.

Mirrors upstream `experiments/benchmark_anomaly_detection.py`, with the
local/global LSTM conditions replaced by a single `llm` condition: Chronos-2
frozen in the `Auxiliary` slot. Upstream sweeps seeds because the LSTM's random
initialization is a real source of spread; Chronos-2 is deterministic, so the
default is a single seed and the per-magnitude spread reported here is across
series rather than across seeds.

A config with `condition: global_finetune` runs upstream's global LSTM condition
on the same series and realizations instead. It has no horizon; its cases are
(seed, series), since upstream keeps one set of global weights per seed.

    python scripts/benchmark.py --dry_run True     # planned work and cost, runs nothing
    python scripts/benchmark.py                    # the full benchmark
    python scripts/benchmark.py --series '["ts67","ts26"]'
"""

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import run_series

# Measured SINGLE-PROCESS latency per Chronos call at context 600 on an Apple M2,
# and the single-process cost of one SKF step (four transition models plus the
# state collapse). These are sequential costs: the stage table below reports
# core-hours, and `estimate_wall_clock` divides by the parallel schedule. A Linux
# server will differ, so treat the absolute numbers as a starting calibration and
# the relative ones as sound.
SECONDS_PER_CALL = 0.2777
SECONDS_PER_CALL_BY_MODEL = {
    "amazon/chronos-bolt-tiny": 0.0105,
    "amazon/chronos-bolt-base": 0.1479,
    "amazon/chronos-2": 0.2777,
    # 54 vs 206 ms for chronos-2 on the Linux server, scaled to the M2 figure above
    "autogluon/chronos-2-small": 0.0728,
}
SECONDS_PER_SKF_STEP = 0.0016

# Global LSTM (1 x 256, lookback 52), single process on the 48-core Linux server,
# measured on ts58: one training step (filter with weight update, or the
# validation forecast) and one SKF step including the LSTM forward pass.
SECONDS_PER_LSTM_TRAIN_STEP = 0.0056
SECONDS_PER_LSTM_SKF_STEP = 0.0014
# Early stopping has patience 20, so a candidate trains for at least 21 epochs;
# ts58 stopped at exactly 21 at both sigma_v timed. A planning figure, not a bound.
PLANNED_LSTM_EPOCHS = 21


def seconds_per_call(config):
    return SECONDS_PER_CALL_BY_MODEL.get(config.get("model_id"), SECONDS_PER_CALL)


def horizons(config):
    """The horizons to sweep, or the single configured one."""

    return [int(h) for h in (config.get("llm_horizon_search_space")
                             or [config["llm_horizon"]])]


def plan(config, series_names):
    """Count the filter runs the benchmark will perform, and their cost."""

    if common.uses_lstm(config):
        return _plan_lstm(config, series_names)
    per_call = seconds_per_call(config)
    rows = []
    for horizon in horizons(config):
        rows.extend(_plan_one_horizon(config, series_names, horizon, per_call))
    return rows


def _plan_lstm(config, series_names):
    """The global-LSTM counterpart of `_plan_one_horizon`, one set of rows per seed.

    Training takes the place of the sigma_v scoring pass: every candidate filters
    the training split and forecasts the validation split once per epoch. The
    LSTM forward pass is part of the SKF step cost, so no Chronos calls appear.
    """

    num_magnitudes = len(common.evaluation_magnitudes(config))
    realizations = 2 * int(config["num_anomaly_realizations"])  # up and down
    rows = []
    for series in series_names:
        dataset = common.prepare_dataset(series, config)
        train_val_steps = len(dataset["train_val"]["y"])
        all_steps = len(dataset["all_data"]["y"])
        if str(config["skf_objective_function"]).lower() == "cdf":
            filters_per_trial, search_steps = 1 + 2 * int(config["cdf_num_anomaly"]), train_val_steps
        else:
            filters_per_trial, search_steps = 1, all_steps
        candidates = len(config["sigma_v_search_space"]) if config["optimize_sigma_v"] else 1

        stages = {
            "LSTM training": (candidates * PLANNED_LSTM_EPOCHS, train_val_steps,
                              SECONDS_PER_LSTM_TRAIN_STEP),
            "SKF search": (
                int(config["num_optimization_trial"]) * filters_per_trial
                if config["optimize_skf_parameters"]
                else 0,
                search_steps,
                SECONDS_PER_LSTM_SKF_STEP,
            ),
            "evaluation": (num_magnitudes * (1 + realizations), all_steps,
                           SECONDS_PER_LSTM_SKF_STEP),
        }
        for seed in config["seeds"]:
            for stage, (num_filters, steps, per_step) in stages.items():
                rows.append(
                    {
                        "series": series,
                        "horizon": 1,
                        "seed": int(seed),
                        "stage": stage,
                        "filters": num_filters,
                        "steps_each": steps,
                        "chronos_calls": 0,
                        "seconds": num_filters * steps * per_step,
                    }
                )
    return rows


def _plan_one_horizon(config, series_names, horizon, per_call):
    num_magnitudes = len(common.evaluation_magnitudes(config))
    realizations = 2 * int(config["num_anomaly_realizations"])  # up and down
    rows = []

    for series in series_names:
        dataset = common.prepare_dataset(series, config)
        train_val_steps = len(dataset["train_val"]["y"])
        all_steps = len(dataset["all_data"]["y"])

        # The `cdf` objective runs a whole detection sweep inside every trial:
        # one clean filter plus 2 x cdf_num_anomaly realizations, over train+val.
        # The `ll` objective is a single filter over all data per trial.
        if str(config["skf_objective_function"]).lower() == "cdf":
            filters_per_trial = 1 + 2 * int(config["cdf_num_anomaly"])
            search_steps = train_val_steps
        else:
            filters_per_trial = 1
            search_steps = all_steps

        stages = {
            "sigma_v grid": (
                len(config["sigma_v_search_space"]) if config["optimize_sigma_v"] else 0,
                train_val_steps,
            ),
            "SKF search": (
                int(config["num_optimization_trial"]) * filters_per_trial
                if config["optimize_skf_parameters"]
                else 0,
                search_steps,
            ),
            # one clean filter per magnitude, plus every realization
            "evaluation": (num_magnitudes * (1 + realizations), all_steps),
        }
        for stage, (num_filters, steps) in stages.items():
            calls = num_filters * steps / horizon
            seconds = num_filters * steps * SECONDS_PER_SKF_STEP + calls * per_call
            rows.append(
                {
                    "series": series,
                    "horizon": horizon,
                    "stage": stage,
                    "filters": num_filters,
                    "steps_each": steps,
                    "chronos_calls": int(calls),
                    "seconds": seconds,
                }
            )
    return rows


def estimate_wall_clock(rows, config, cpus=None):
    """Greedy longest-first schedule of the (H, series) cases over the cores.

    Case cost is dominated by H=1: every horizon runs the same *number* of
    filters, and only the Chronos calls per filter scale as 1/H, so the sequential
    cost of the H=1 case sets the floor unless it is given more than one core.
    """

    available = max(1, cpus or os.cpu_count() or 1)
    case_seconds = defaultdict(float)
    for row in rows:
        case_seconds[(row["horizon"], row["series"])] += row["seconds"]

    num_cases = len(case_seconds)
    requested = config.get("max_concurrent")
    concurrent = int(requested) if requested else available
    concurrent = max(1, min(concurrent, num_cases, available))
    budget = max(1, available // concurrent)

    # Inner parallelism is not free: the Chronos calls scale with cores, the SKF
    # bookkeeping much less. Assume 80% efficiency on the extra cores.
    speedup = 1 + (budget - 1) * 0.8
    slots = [0.0] * concurrent
    for seconds in sorted(case_seconds.values(), reverse=True):
        index = min(range(concurrent), key=lambda i: slots[i])
        slots[index] += seconds / speedup
    return {
        "cpus": available,
        "concurrent": concurrent,
        "budget": budget,
        "num_cases": num_cases,
        "longest_case_h": max(case_seconds.values()) / speedup / 3600,
        "wall_h": max(slots) / 3600,
        "serial_h": sum(case_seconds.values()) / 3600,
    }


def print_plan(rows, config, cpus=None):
    by_stage = defaultdict(lambda: {"filters": 0, "chronos_calls": 0, "seconds": 0.0})
    for row in rows:
        entry = by_stage[row["stage"]]
        entry["filters"] += row["filters"]
        entry["chronos_calls"] += row["chronos_calls"]
        entry["seconds"] += row["seconds"]

    lstm = common.uses_lstm(config)
    sweep = f"seeds={config['seeds']}" if lstm else f"H={horizons(config)}"
    model = config["condition"] if lstm else config.get("model_id")
    print(f"\nPlanned work over {sweep}, "
          f"objective={config['skf_objective_function']}, "
          f"model={model}, "
          f"{len({r['series'] for r in rows})} series\n")
    print(f"{'stage':<16}{'filters':>10}{'Chronos calls':>16}{'core-hours':>12}")
    print("-" * 52)
    total_seconds = 0.0
    for stage, entry in by_stage.items():
        print(f"{stage:<16}{entry['filters']:>10,}{entry['chronos_calls']:>16,}"
              f"{entry['seconds'] / 3600:>12.1f}")
        total_seconds += entry["seconds"]
    print("-" * 52)
    print(f"{'TOTAL':<16}{sum(e['filters'] for e in by_stage.values()):>10,}"
          f"{sum(e['chronos_calls'] for e in by_stage.values()):>16,}"
          f"{total_seconds / 3600:>12.1f}")
    if lstm:
        print(
            "\nCore-hours, from single-process cost measured on the Linux server "
            f"({SECONDS_PER_LSTM_TRAIN_STEP * 1000:.1f} ms per training step, "
            f"{SECONDS_PER_LSTM_SKF_STEP * 1000:.1f} ms per SKF step, "
            f"{PLANNED_LSTM_EPOCHS} epochs per sigma_v candidate)."
        )
    else:
        print(
            "\nCore-hours, from single-process latency "
            f"({seconds_per_call(config) * 1000:.1f} ms per call for "
            f"{config.get('model_id')}, {SECONDS_PER_SKF_STEP * 1000:.1f} ms per SKF "
            "step, measured on an Apple M2)."
        )
        print(
            "Cost scales as 1/llm_horizon for the Chronos calls; the SKF term does not."
        )
    if str(config["skf_objective_function"]).lower() == "cdf":
        print(
            f"The cdf objective filters {1 + 2 * int(config['cdf_num_anomaly'])} "
            "series per trial, so cdf_num_anomaly and num_optimization_trial "
            "multiply together and dominate everything else."
        )
    print(
        "Cheapest knobs: cdf_num_anomaly, num_optimization_trial, "
        f"num_anomaly_realizations, slope_search_space{'' if lstm else ', llm_horizon'}."
    )

    print(f"\n{'cores':>7}{'parallel cases':>16}{'cpus/case':>11}"
          f"{'longest case':>14}{'WALL CLOCK':>13}")
    print("-" * 61)
    planned = cpus or config.get("cpus") or os.cpu_count() or 8
    for candidate in sorted({8, 16, 32, 64, planned}):
        est = estimate_wall_clock(rows, config, candidate)
        marker = "  <- configured" if candidate == planned else ""
        print(f"{est['cpus']:>7}{est['concurrent']:>16}{est['budget']:>11}"
              f"{est['longest_case_h']:>13.1f}h{est['wall_h']:>12.1f}h{marker}")
    print(f"\nCases are independent, so this parallelizes at the "
          f"({'seed' if lstm else 'H'}, series) level; "
          f"{estimate_wall_clock(rows, config, cpus)['num_cases']} cases, "
          f"{estimate_wall_clock(rows, config, cpus)['serial_h']:.0f} core-hours total.")


def _run_case(args):
    """One (H, series) or (seed, series) case in its own process, with its own CPU budget."""

    series, case, config, run_dir = args
    if common.uses_lstm(config):
        return run_series.run(series, {**config, "seed": case}, Path(run_dir) / f"seed{case}" / series)
    case_config = {**config, "llm_horizon": case}
    return run_series.run(series, case_config, Path(run_dir) / f"H{case}" / series)


SPEEDUP_EFFICIENCY = 0.8  # extra cores help the Chronos calls, not the SKF bookkeeping


def _case_seconds(rows):
    seconds = defaultdict(float)
    for row in rows:
        seconds[(row["horizon"], row.get("seed"), row["series"])] += row["seconds"]
    return sorted(seconds.values(), reverse=True)


def _makespan(case_seconds, concurrent, budget, useful_budget=None):
    """Greedy longest-first packing of cases onto `concurrent` slots.

    A case cannot use unlimited cores: its widest stage is the SKF search, whose
    trials are the only thing there are many of. Cores beyond that are idle, so
    the modelled speedup is capped rather than growing without bound.
    """

    effective = min(budget, useful_budget or budget)
    speedup = 1 + (effective - 1) * SPEEDUP_EFFICIENCY
    slots = [0.0] * concurrent
    for seconds in case_seconds:
        index = min(range(concurrent), key=lambda i: slots[i])
        slots[index] += seconds / speedup
    return max(slots)


def useful_budget(config):
    """The most cores one case can keep busy, set by its widest parallel stage."""

    return max(
        int(config.get("num_optimization_trial", 50)) if config.get("optimize_skf_parameters") else 1,
        len(config.get("sigma_v_search_space", [])) if config.get("optimize_sigma_v") else 1,
        len(common.evaluation_magnitudes(config)),
    )


def choose_parallelism(rows, config, cpus=None):
    """Split the cores between concurrent cases and each case's own budget.

    Running as many cases as there are cores is usually wrong: the cases are very
    unbalanced -- every horizon runs the same number of filters and only the
    Chronos calls scale as 1/H, so H=1 costs far more than H=208 -- and giving
    each one a single core leaves the longest case setting the wall clock. Fewer,
    fatter cases finish sooner. `max_concurrent` overrides this if set.
    """

    available = max(1, cpus or config.get("cpus") or os.cpu_count() or 1)
    case_seconds = _case_seconds(rows)
    num_cases = len(case_seconds)

    requested = config.get("max_concurrent")
    if requested:
        concurrent = max(1, min(int(requested), num_cases, available))
        return concurrent, max(1, available // concurrent)

    best = None
    for concurrent in range(1, min(num_cases, available) + 1):
        budget = max(1, available // concurrent)
        span = _makespan(case_seconds, concurrent, budget, useful_budget(config))
        if best is None or span < best[0]:
            best = (span, concurrent, budget)
    return best[1], best[2]


def estimate_wall_clock(rows, config, cpus=None):
    available = max(1, cpus or config.get("cpus") or os.cpu_count() or 1)
    case_seconds = _case_seconds(rows)
    concurrent, budget = choose_parallelism(rows, config, available)
    return {
        "cpus": available,
        "concurrent": concurrent,
        "budget": budget,
        "num_cases": len(case_seconds),
        "longest_case_h": case_seconds[0]
        / (1 + (min(budget, useful_budget(config)) - 1) * SPEEDUP_EFFICIENCY)
        / 3600,
        "wall_h": _makespan(case_seconds, concurrent, budget, useful_budget(config)) / 3600,
        "serial_h": sum(case_seconds) / 3600,
    }


def aggregate(summaries):
    """Mean and spread of each metric per magnitude, across series (and seeds).

    The LSTM has no horizon, so its `llm_horizon` is None; with several seeds the
    spread is over every (seed, series) run.
    """

    groups = defaultdict(list)
    for summary in summaries:
        for row in summary["multi_realization_evaluation"]:
            key = (summary.get("condition", "llm"), summary.get("llm_horizon"),
                   row["anomaly_magnitude"])
            groups[key].append({**row, "series": summary["series"], "seed": summary.get("seed")})

    aggregates = []
    for (condition, horizon, magnitude), items in sorted(groups.items()):
        def stat(key, reducer):
            values = [r[key] for r in items if r[key] is not None and np.isfinite(r[key])]
            return float(reducer(values)) if values else None

        aggregates.append(
            {
                "condition": condition,
                "llm_horizon": horizon,
                "anomaly_magnitude": magnitude,
                "probability_of_detection": stat("probability_of_detection", np.mean),
                "probability_of_detection_std": stat("probability_of_detection", np.std),
                "false_alarm_rate_per_y_mean": stat("false_alarm_rate_per_y", np.mean),
                "false_alarm_rate_per_y_std": stat("false_alarm_rate_per_y", np.std),
                "time_to_detection_years_mean": stat("time_to_detection_years_mean", np.mean),
                "time_to_detection_years_std": stat("time_to_detection_years_mean", np.std),
                "total_realizations": sum(r["num_realizations"] for r in items),
                "num_series": len({r["series"] for r in items}),
                "num_seeds": len({r["seed"] for r in items}),
            }
        )
    return aggregates


def print_aggregates(aggregates):
    header = (
        f"{'H':>6}{'Magnitude':>10}{'P(detect)':>12}{'P(det) std':>12}"
        f"{'FA/yr mean':>12}{'FA/yr std':>12}{'TTD(yr)':>10}{'N(real)':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in aggregates:
        fmt = lambda v, d=3: "N/A" if v is None else f"{v:.{d}f}"
        horizon = "-" if row["llm_horizon"] is None else row["llm_horizon"]
        print(
            f"{horizon:>6}"
            f"{row['anomaly_magnitude']:>10.3f}"
            f"{fmt(row['probability_of_detection'], 2):>12}"
            f"{fmt(row['probability_of_detection_std']):>12}"
            f"{fmt(row['false_alarm_rate_per_y_mean']):>12}"
            f"{fmt(row['false_alarm_rate_per_y_std']):>12}"
            f"{fmt(row['time_to_detection_years_mean']):>10}"
            f"{row['total_realizations']:>10}"
        )


def main(config_path=None, series=None, dry_run=False, output_dir=None,
         horizons_subset=None, max_concurrent=None, cpus=None):
    """Run the benchmark.

    `horizons_subset` restricts the sweep, e.g. --horizons_subset '[1,3]', so the
    work can be sharded across machines or launched as separate processes.
    `max_concurrent` caps how many (H, series) cases run side by side.
    """

    config = common.load_config(config_path)
    series_names = list(series) if series else common.resolve_series(config)
    if horizons_subset:
        config = {**config, "llm_horizon_search_space": [int(h) for h in horizons_subset]}
    if max_concurrent:
        config = {**config, "max_concurrent": int(max_concurrent)}
    if cpus:
        config = {**config, "cpus": int(cpus)}

    if dry_run:
        print_plan(plan(config, series_names), config, cpus)
        return

    run_dir = Path(output_dir) if output_dir else (
        common.EXP_DIR / "results" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.json").write_text(json.dumps(config, indent=2))

    started = time.time()
    summaries = []
    lstm = common.uses_lstm(config)
    sweep = [int(s) for s in config["seeds"]] if lstm else horizons(config)
    label = "seed" if lstm else "H"
    cases = [(c, s) for c in sweep for s in series_names]
    concurrent, budget = choose_parallelism(plan(config, series_names), config)
    usable = config.get("cpus") or os.cpu_count()
    print(f"{len(cases)} cases ({len(sweep)} {'seeds' if lstm else 'horizons'} x "
          f"{len(series_names)} series), "
          f"up to {concurrent} in parallel, {budget} CPU(s) each, "
          f"using {usable} of {os.cpu_count()} cores.")

    if concurrent == 1:
        for index, (case, name) in enumerate(cases, 1):
            print(f"\n{'=' * 70}\n[{index}/{len(cases)}] {name}  {label}={case}\n"
                  f"{'=' * 70}", flush=True)
            try:
                summaries.append(_run_case((name, case, config, run_dir)))
            except Exception as exc:  # one bad case must not lose the rest
                print(f"FAILED {name} {label}={case}: {type(exc).__name__}: {exc}",
                      flush=True)
    else:
        case_config = {**config, "num_workers": budget}
        payload = [(name, case, case_config, str(run_dir)) for case, name in cases]
        with ProcessPoolExecutor(
            max_workers=concurrent, mp_context=get_context("spawn")
        ) as executor:
            futures = {executor.submit(_run_case, item): item for item in payload}
            for done, future in enumerate(as_completed(futures), 1):
                name, case, *_ = futures[future]
                try:
                    summaries.append(future.result())
                    print(f"[{done}/{len(cases)}] done {name} {label}={case}", flush=True)
                except Exception as exc:
                    print(f"[{done}/{len(cases)}] FAILED {name} {label}={case}: "
                          f"{type(exc).__name__}: {exc}", flush=True)

    aggregates = aggregate(summaries)
    (run_dir / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "condition": config.get("condition", "llm"),
                ("seeds" if lstm else "llm_horizon_search_space"): sweep,
                "series": series_names,
                "completed": [s["series"] for s in summaries],
                "aggregate_by_magnitude": aggregates,
                "per_series": summaries,
                "total_elapsed_seconds": round(time.time() - started, 1),
            },
            indent=2,
        )
    )
    print_aggregates(aggregates)
    print(f"\nSaved: {run_dir / 'benchmark_summary.json'}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
