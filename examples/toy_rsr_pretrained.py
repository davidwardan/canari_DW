"""Offline LSTM training with a validation split, then RSR on the test split.

Same trending series and same coupled model as `toy_rsr_trend.py`, but the LSTM is
trained the usual offline way first: several epochs over the train split, with the
validation split used only to select the epoch and stop early. Recurrent Smoothing and
Replay (RSR) then takes over on the test split and keeps updating the LSTM parameters
online -- one analytical update per observation -- while the `LocalTrend` baseline
carries the rising level.

`rsr_filter` re-filters everything before the test split with frozen parameters, so the
model reaches the first test step with the memory offline training left it and the test
observations are never used to select anything.

Run from the repository root:

    python examples/toy_rsr_pretrained.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytagi import metric

from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

NUM_STEPS = 24 * 40
PERIOD = 24
LEVEL = 2.0
SLOPE = 0.006
STD_NOISE = 0.1
SEED = 1
NUM_EPOCHS = 30
WINDOW_LEN = 24
TRAIN_SPLIT = 0.6
# pyTAGI sizes the SLSTM sequence buffer on the first forward pass and never grows it.
# Offline training fixes it at `3 * infer_len - 1 + len(train)`, and the frozen pass
# `rsr_filter` runs before the test split needs `test_start - window_len` of those
# slots. The validation split must therefore stay under `3 * infer_len - 1 + window_len`
# steps -- 95 here, against the 76 this split gives.
VALIDATION_SPLIT = 0.08
STATES_TO_PLOT = ["level", "trend", "lstm", "white noise"]


def generate_data():
    """Hourly series: a constant upward level, two daily harmonics, and white noise."""

    rng = np.random.default_rng(SEED)
    time = np.arange(NUM_STEPS)
    periodic = np.sin(2 * np.pi * time / PERIOD) + 0.5 * np.sin(
        4 * np.pi * time / PERIOD + np.pi / 4
    )
    values = LEVEL + SLOPE * time + periodic + rng.normal(0, STD_NOISE, NUM_STEPS)

    data = pd.DataFrame({"values": values})
    data.index = pd.date_range("2000-01-01", periods=NUM_STEPS, freq="h")
    data.index.name = "date_time"
    return data


def main():
    data_processor = DataProcess(
        data=generate_data(),
        time_covariates=["hour_of_day"],
        train_split=TRAIN_SPLIT,
        validation_split=VALIDATION_SPLIT,
        output_col=[0],
    )
    train_data, validation_data, test_data, all_data = data_processor.get_splits()

    model = Model(
        LocalTrend(std_error=1e-4),
        LstmNetwork(
            look_back_len=PERIOD,
            num_features=2,
            infer_len=PERIOD,
            num_hidden_unit=40,
            manual_seed=SEED,
        ),
        WhiteNoise(std_error=STD_NOISE),
    )
    # Three full days: a single day cannot separate the slope from the daily cycle.
    model.auto_initialize_baseline_states(train_data["y"][: PERIOD * 3])

    # Offline training: the train split updates the parameters, the validation split
    # only selects the epoch. `early_stopping` restores the best parameters on stop.
    validation_obs = validation_data["y"].flatten()
    for epoch in range(NUM_EPOCHS):
        validation_mean, validation_std, states = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
            data_processor=data_processor,
        )
        model.early_stopping(
            evaluate_metric=float(
                metric.log_likelihood(validation_mean, validation_obs, validation_std)
            ),
            current_epoch=epoch,
            max_epoch=NUM_EPOCHS,
            mode="max",
            patience=5,
        )
        if epoch == model.optimal_epoch:
            optimal_validation_mean = validation_mean.copy()
            optimal_validation_std = validation_std.copy()
        model.set_memory(states=states, time_step=0)
        if model.stop_training:
            break

    # Online phase: RSR carries the pretrained model to the test split with frozen
    # parameters, then resumes learning there. Each returned prediction is made before
    # its own test observation updates the LSTM and the baseline states.
    test_mean, test_std, test_states = model.rsr_filter(
        data=all_data,
        start=data_processor.test_start,
        window_len=WINDOW_LEN,
    )
    test_obs = test_data["y"].flatten()

    print(f"Optimal epoch               : {model.optimal_epoch}")
    print(f"Validation forecast log-lik : {model.early_stop_metric: 0.3f}")
    print(f"Online test MSE             : {metric.mse(test_mean, test_obs): 0.4f}")
    print(
        "Online test log-likelihood  : "
        f"{metric.log_likelihood(test_mean, test_obs, test_std): 0.3f}"
    )

    # Observations, validation forecast from the selected epoch, and online test
    # predictions.
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data(
        data_processor=data_processor,
        standardization=True,
        sub_plot=ax,
        train_label="train",
        validation_label="validation",
        test_label="test",
    )
    plot_prediction(
        data_processor=data_processor,
        mean_validation_pred=optimal_validation_mean,
        std_validation_pred=optimal_validation_std,
        sub_plot=ax,
        validation_label=["validation forecast", r"$\pm\sigma$"],
    )
    plot_prediction(
        data_processor=data_processor,
        mean_test_pred=test_mean,
        std_test_pred=test_std,
        sub_plot=ax,
        color="purple",
        test_label=["online prediction", r"$\pm\sigma$"],
    )
    ax.set_ylabel("y")
    ax.legend(loc=(0.0, 1.01), ncol=5, frameon=False)
    fig.tight_layout()

    # Hidden states estimated while learning online. RSR returns exactly one estimate
    # per test step, so `plot_states` places them on the test index on its own.
    plot_states(
        data_processor=data_processor,
        states=test_states,
        states_to_plot=STATES_TO_PLOT,
        states_type="posterior",
        standardization=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
