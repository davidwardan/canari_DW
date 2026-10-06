"""Assert that a global-LSTM config sees exactly the LLM sweep's data.

For every series: the same rows in every split, the same standardized `y`, the
same warmup, and the same synthetic realizations -- both the evaluation ones and
the ones the cdf objective tunes on. Also checks that the LSTM starts from the
pretrained means with freshly initialized variances.

    python scripts/check_same_inputs.py \
        --llm_run results/full_chronos2_small_zerovar_residual \
        --config_path config/full_global_finetune.yaml
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

# Keys that decide the data, the splits and the realizations.
SHARED_KEYS = [
    "values_file", "datetimes_file", "series", "global_warmup_lookback_len",
    "baseline_init_len", "validation_ratio", "test_ratio", "num_anomaly_realizations",
    "evaluation_magnitudes", "max_timestep_to_detect", "slope_search_space",
    "cdf_num_anomaly", "cdf_anomaly_start", "cdf_anomaly_end",
]
SPLITS = ["train_data", "validation_data", "test_data", "all_data", "train_val"]


def evaluation_realizations(dataset, magnitude, config):
    """As `run_series._evaluate_magnitude` draws them."""

    processor = dataset["data_processor"]
    total_steps = len(dataset["all_data"]["y"])
    return common.make_realizations(
        dataset["all_data"],
        magnitude,
        config["num_anomaly_realizations"],
        processor.test_start / total_steps,
        (processor.test_end - config["max_timestep_to_detect"]) / total_steps,
    )


def tuning_realizations(dataset, magnitude, config):
    """As `run_series._skf_from_param` draws them for the cdf objective."""

    return common.make_realizations(
        dataset["train_val"], magnitude, config["cdf_num_anomaly"],
        config["cdf_anomaly_start"], config["cdf_anomaly_end"],
    )


def assert_same_realizations(ours, theirs, label):
    assert len(ours) == len(theirs), label
    for a, b in zip(ours, theirs):
        assert a["anomaly_timestep"] == b["anomaly_timestep"], label
        np.testing.assert_array_equal(a["y"], b["y"], err_msg=label)


def check_lstm_start(dataset, config):
    """Pretrained means, initialization variances, as upstream's increase_output_variance."""

    from canari.component import LstmNetwork

    model = common.build_lstm_model(0.1, dataset, config)
    kwargs = dict(
        look_back_len=int(config["lstm_look_back_len"]),
        num_features=int(config["lstm_num_features"]),
        num_layer=int(config["lstm_num_layer"]),
        num_hidden_unit=int(config["num_hidden_unit"]),
        manual_seed=int(config["seed"]),
        smoother=False,
    )
    pretrained = LstmNetwork(load_lstm_net=common.global_lstm_params(config), **kwargs)
    pretrained = pretrained.initialize_lstm_network().state_dict()
    fresh = LstmNetwork(**kwargs).initialize_lstm_network().state_dict()
    for name, (mu_w, var_w, mu_b, var_b) in model.lstm_net.state_dict().items():
        np.testing.assert_array_equal(mu_w, pretrained[name][0])
        np.testing.assert_array_equal(mu_b, pretrained[name][2])
        np.testing.assert_array_equal(var_w, fresh[name][1])
        np.testing.assert_array_equal(var_b, fresh[name][3])
        assert not np.array_equal(var_w, pretrained[name][1]), name
    assert model.mu_states[model.get_states_index("trend")] == 0.0


def main(llm_run, config_path):
    llm_run = Path(llm_run)
    llm_config = json.loads((llm_run / "config_used.json").read_text())
    config = common.load_config(config_path)
    config.setdefault("seed", int(config["seeds"][0]))

    for key in SHARED_KEYS:
        assert config.get(key) == llm_config.get(key), f"{key}: {config.get(key)} != {llm_config.get(key)}"

    series_names = common.resolve_series(config)
    llm_series = sorted({p.parent.name for p in llm_run.glob("H*/*/summary.json")})
    assert sorted(series_names) == llm_series, (series_names, llm_series)

    for series in series_names:
        ours = common.prepare_dataset(series, config)
        theirs = common.prepare_dataset(series, llm_config)
        for split in SPLITS:
            np.testing.assert_array_equal(ours[split]["y"], theirs[split]["y"], err_msg=f"{series} {split}")
            assert ours[split]["x"].shape == (len(ours[split]["y"]), 1), f"{series} {split} x"
        np.testing.assert_array_equal(ours["warmup_context"], theirs["warmup_context"])
        for attr in ("train_start", "validation_start", "test_start", "test_end"):
            assert getattr(ours["data_processor"], attr) == getattr(theirs["data_processor"], attr)
        for magnitude in common.evaluation_magnitudes(config):
            assert_same_realizations(
                evaluation_realizations(ours, magnitude, config),
                evaluation_realizations(theirs, magnitude, llm_config),
                f"{series} evaluation {magnitude}",
            )
        for magnitude in config["slope_search_space"]:
            assert_same_realizations(
                tuning_realizations(ours, magnitude, config),
                tuning_realizations(theirs, magnitude, llm_config),
                f"{series} tuning {magnitude}",
            )
        print(f"{series}: same splits, y, warmup and realizations "
              f"({len(ours['all_data']['y'])} steps)")

    check_lstm_start(common.prepare_dataset(series_names[0], config), config)
    print("LSTM starts from pretrained means with initialization variances")
    print("OK")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
