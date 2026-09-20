"""Run the benchmark pipeline for one series.

Mirrors upstream `experiments/anomaly_detection_lstm.py::main` with the LSTM slot
filled by Chronos-2 through the `Auxiliary` component:

1. **sigma_v grid search** -- upstream retrains the LSTM at every candidate and
   scores the validation CRPS at the early-stopping epoch. The foundation model is
   frozen, so there is nothing to train: each candidate is scored by filtering
   train+validation once and measuring the same validation metric. The selection
   rule is upstream's -- the *smallest* sigma_v within `sigma_v_selection_delta` of
   the best score.
2. **SKF parameter search** -- the same Ray Tune space over
   `std_transition_error`, `norm_to_abnorm_prob`, `abnorm_to_norm_prob` and
   `threshold`, with the same objective.
3. **Multi-realization evaluation** -- the same magnitudes, the same up-and-down
   injection, the same three-year detection window, the same metrics.

    python scripts/run_series.py --series ts67
"""

import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

METRIC_KEYS = {
    "crps": ("validation_crps", "min"),
    "ll": ("validation_log_likelihood", "max"),
}


# ---- Stage 1: sigma_v grid search ---------------------------------------------
def _score_sigma_v(task):
    """Filter train+validation once at one sigma_v and score the validation split."""

    series, config, sigma_v = task
    common.set_model_id(config.get("model_id"))
    dataset = common.prepare_dataset(series, config)
    model, _ = common.build_model(common.predict_fn(), sigma_v, dataset, config)
    mu, std, _ = model.filter(data=dataset["train_val"])

    num_validation = len(dataset["validation_data"]["y"])
    observations = dataset["data_processor"].get_data("validation").flatten()
    metrics = common.validation_metrics(
        mu[-num_validation:], std[-num_validation:], observations,
        dataset["data_processor"],
    )
    return {"sigma_v": float(sigma_v), **metrics}


def search_sigma_v(series, config):
    metric_key, mode = METRIC_KEYS[str(config["early_stopping_metric"]).lower()]
    candidates = list(config["sigma_v_search_space"])
    print(f"---- sigma_v grid search over {candidates} (metric: {metric_key}) ----")

    tasks = [(series, config, sv) for sv in candidates]
    workers = max(1, min(int(config["num_workers"]), len(tasks)))
    if workers > 1:
        with mp.get_context("spawn").Pool(
            workers, initializer=common.init_worker, initargs=(config.get("model_id"),)
        ) as pool:
            results = pool.map(_score_sigma_v, tasks)
    else:
        results = [_score_sigma_v(task) for task in tasks]

    for row in results:
        print(f"  sigma_v={row['sigma_v']:.4f} -> {metric_key}={row[metric_key]:.6f}")

    scores = [row[metric_key] for row in results]
    best = max(scores) if mode == "max" else min(scores)
    delta = float(config.get("sigma_v_selection_delta", 0.0))
    tolerated = [
        row
        for row, score in zip(results, scores)
        if (score >= best - delta if mode == "max" else score <= best + delta)
    ]
    chosen = min(tolerated, key=lambda row: row["sigma_v"])
    print(f"---- Optimal sigma_v: {chosen['sigma_v']:.4f} "
          f"({metric_key}={chosen[metric_key]:.6f}) ----")
    return chosen, results


# ---- Stage 2: SKF parameter search ---------------------------------------------
def _skf_from_param(param, model_input):
    """Build an SKF at `param` and attach the objective Ray Tune will read."""

    series = model_input["series"]
    config = model_input["config"]
    resolved = {**model_input["default_param"], **param}

    common.set_model_id(config.get("model_id"))
    dataset = common.prepare_dataset(series, config)
    skf, auxiliary = common.build_skf(common.predict_fn(), resolved, dataset, config)

    if str(config["skf_objective_function"]).lower() == "cdf":
        magnitude = float(resolved["slope"])
        realizations = common.make_realizations(
            dataset["train_val"],
            magnitude,
            config["cdf_num_anomaly"],
            config["cdf_anomaly_start"],
            config["cdf_anomaly_end"],
        )
        detection_rate, num_false_alarms, _ = common.detect_synthetic_anomaly(
            skf,
            auxiliary,
            dataset["train_val"],
            realizations,
            threshold=float(resolved["threshold"]),
            max_timestep_to_detect=config["max_timestep_to_detect"],
        )
        years = common.years_spanned(
            dataset["data_processor"], dataset["data_processor"].validation_end
        )
        false_rate = num_false_alarms / years
        skf.metric_optim = skf.objective(
            detection_rate,
            false_rate,
            magnitude,
            detection_rate_cdf_mean=float(config["objective_detection_rate_cdf_mean"]),
            detection_rate_cdf_std=float(config["objective_detection_rate_cdf_std"]),
            false_rate_cdf_median=float(config["objective_false_rate_cdf_median"]),
            false_rate_cdf_shape=float(config["objective_false_rate_cdf_shape"]),
            anm_mag_cdf_median=float(config["objective_anm_mag_cdf_median"]),
            anm_mag_cdf_shape=float(config["objective_anm_mag_cdf_shape"]),
        )
        skf.print_metric = {
            "detection_rate": detection_rate,
            "yearly_false_rate": false_rate,
            "slope": magnitude,
        }
    else:
        skf.filter(data=dataset["all_data"])
        skf.metric_optim = -float(np.nanmean(skf.ll_history))
        skf.print_metric = {"neg_mean_log_likelihood": skf.metric_optim}

    common.reset(skf, auxiliary)
    return skf


def search_skf_parameters(series, config, default_param):
    import ray
    from canari import Optimizer
    from ray import tune

    # Ray sizes its trial concurrency from the whole machine unless told
    # otherwise. When several (H, series) cases run side by side, each would
    # claim every core, so bound this process to its share.
    budget = max(1, int(config["num_workers"]))
    if not ray.is_initialized():
        # Every concurrent case starts its own local Ray. Two defaults have to be
        # overridden or a many-core box falls over: Ray sizes trial concurrency
        # from the whole machine, and it reserves roughly 30% of system RAM for
        # its object store -- which, multiplied by the number of concurrent cases,
        # is enough to exhaust the server. Trials here exchange small dicts, so a
        # small store is ample.
        ray.init(
            num_cpus=budget,
            object_store_memory=int(config.get("ray_object_store_mb", 256)) * 1024**2,
            include_dashboard=False,
            log_to_driver=False,
            ignore_reinit_error=True,
            configure_logging=False,
        )

    space = {
        "std_transition_error": tune.loguniform(
            *[float(v) for v in config["std_transition_error_search_space"]]
        ),
        "norm_to_abnorm_prob": tune.loguniform(
            *[float(v) for v in config["norm_to_abnorm_prob_search_space"]]
        ),
        "abnorm_to_norm_prob": tune.quniform(
            *[float(v) for v in config["abnorm_to_norm_prob_search_space"]], 1e-2
        ),
        "threshold": tune.quniform(
            *[float(v) for v in config["threshold_search_space"]], 1e-2
        ),
    }
    objective = str(config["skf_objective_function"]).lower()
    if objective == "cdf":
        space["slope"] = tune.choice(list(config["slope_search_space"]))

    optimizer = Optimizer(
        model=_skf_from_param,
        param=space,
        model_input={"series": series, "config": config, "default_param": default_param},
        num_optimization_trial=int(config["num_optimization_trial"]),
        mode="max" if objective == "cdf" else "min",
        num_startup_trials=int(config["num_startup_trials"]),
    )
    optimizer.optimize()
    return {**default_param, **optimizer.get_best_param()}


# ---- Stage 3: multi-realization evaluation -------------------------------------
def _evaluate_magnitude(task):
    """Detection rate, false alarms and time to detection at one magnitude."""

    series, config, param, magnitude = task
    common.set_model_id(config.get("model_id"))
    dataset = common.prepare_dataset(series, config)
    processor = dataset["data_processor"]
    total_steps = len(dataset["all_data"]["y"])
    anomaly_start = processor.test_start / total_steps
    anomaly_end = (processor.test_end - config["max_timestep_to_detect"]) / total_steps

    realizations = common.make_realizations(
        dataset["all_data"],
        magnitude,
        config["num_anomaly_realizations"],
        anomaly_start,
        anomaly_end,
    )
    skf, auxiliary = common.build_skf(common.predict_fn(), param, dataset, config)
    detection_rate, num_false_alarms, (ttd_mean, ttd_std) = common.detect_synthetic_anomaly(
        skf,
        auxiliary,
        dataset["all_data"],
        realizations,
        threshold=float(param["threshold"]),
        max_timestep_to_detect=config["max_timestep_to_detect"],
    )
    years = common.years_spanned(processor, processor.test_end)
    return {
        "anomaly_magnitude": float(magnitude),
        "probability_of_detection": detection_rate,
        "false_alarm_rate_per_y": num_false_alarms / years,
        "num_false_alarms": num_false_alarms,
        "time_to_detection_years_mean": ttd_mean / common.WEEKS_PER_YEAR,
        "time_to_detection_years_std": ttd_std / common.WEEKS_PER_YEAR,
        "num_realizations": len(realizations),
    }


def evaluate_magnitudes(series, config, param, partial_path=None):
    magnitudes = list(config["slope_search_space"])
    tasks = [(series, config, param, mag) for mag in magnitudes]
    workers = max(1, min(int(config["num_workers"]), len(tasks)))

    def record(row):
        """Append as each magnitude lands, so a late failure loses only the tail."""

        if partial_path is not None:
            done = json.loads(partial_path.read_text()) if partial_path.exists() else []
            done.append(row)
            partial_path.write_text(json.dumps(done, indent=2))
        return row

    results = []
    if workers > 1:
        with mp.get_context("spawn").Pool(
            workers, initializer=common.init_worker, initargs=(config.get("model_id"),)
        ) as pool:
            for row in pool.imap_unordered(_evaluate_magnitude, tasks):
                results.append(record(row))
    else:
        for task in tasks:
            results.append(record(_evaluate_magnitude(task)))

    for row in sorted(results, key=lambda r: r["anomaly_magnitude"]):
        print(
            f"  mag={row['anomaly_magnitude']:.4f}:  "
            f"P(detect)={row['probability_of_detection']:.2f}  "
            f"FA/yr={row['false_alarm_rate_per_y']:.2f}  "
            f"TTD(yr)={row['time_to_detection_years_mean']:.3f}"
            f"±{row['time_to_detection_years_std']:.3f}"
        )
    return sorted(results, key=lambda r: r["anomaly_magnitude"])


# ---- Pipeline ------------------------------------------------------------------
def run(series, config, output_dir, resume=True):
    """Tune, then evaluate. Tuning is checkpointed because it is the expensive half.

    The SKF search can run for hours; writing its result before the evaluation
    starts means a failure downstream costs only the evaluation, and rerunning the
    same output directory picks up where it left off.
    """

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_path = output_dir / "tuned_params.json"

    if resume and tuning_path.exists():
        saved = json.loads(tuning_path.read_text())
        param = saved["model_parameters_used"]
        sigma_v_result = saved.get("optimal_validation_metrics")
        sigma_v_grid = saved.get("sigma_v_grid_search")
        print(f"Resuming: reusing tuned parameters from {tuning_path}")
    else:
        sigma_v_result, sigma_v_grid = None, None
        if config.get("optimize_sigma_v", False):
            sigma_v_result, sigma_v_grid = search_sigma_v(series, config)
            sigma_v = sigma_v_result["sigma_v"]
        else:
            sigma_v = float(config["sigma_v"])

        default_param = {
            "sigma_v": sigma_v,
            "std_transition_error": float(config["std_transition_error"]),
            "norm_to_abnorm_prob": float(config["norm_to_abnorm_prob"]),
            "abnorm_to_norm_prob": float(config["abnorm_to_norm_prob"]),
            "threshold": float(config["anomaly_detection_threshold"]),
            "slope": float(config.get("slope", 0.5)),
        }
        if config.get("optimize_skf_parameters", False):
            param = search_skf_parameters(series, config, default_param)
        else:
            param = default_param.copy()
        tuning_path.write_text(
            json.dumps(
                {
                    "series": series,
                    "model_parameters_used": param,
                    "optimal_validation_metrics": sigma_v_result,
                    "sigma_v_grid_search": sigma_v_grid,
                    "tuning_elapsed_seconds": round(time.time() - started, 1),
                },
                indent=2,
            )
        )
    print("Model parameters used:", param)

    results = evaluate_magnitudes(
        series, config, param, partial_path=output_dir / "evaluation.partial.json"
    )

    summary = {
        "series": series,
        "created": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 1),
        "llm_horizon": int(config["llm_horizon"]),
        "model_parameters_used": param,
        "optimal_validation_metrics": sigma_v_result,
        "sigma_v_grid_search": sigma_v_grid,
        "multi_realization_evaluation": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {output_dir / 'summary.json'}  ({summary['elapsed_seconds']:.0f}s)")
    return summary


def main(series=None, config_path=None, output_dir=None):
    config = common.load_config(config_path)
    series = series or common.resolve_series(config)[0]
    output_dir = Path(output_dir) if output_dir else (
        common.EXP_DIR / "results" / datetime.now().strftime("run_%Y%m%d_%H%M%S") / series
    )
    run(series, config, output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
