"""Online LSTM learning with RSR on a trending series, then forecasting.

The series is the stationary one of `toy_rsr_stationary.py` placed on a level that rises
at a constant rate. When forecasting, the LSTM only reads back its own outputs and the
hour-of-day covariate, both of which repeat every day, so on its own it can only predict
a daily pattern: it has no way to keep climbing past the data it has seen. The rise has
to come from the state-space model. A `LocalTrend` baseline carries it, which leaves the
LSTM with the recurrent pattern alone.

The LSTM starts untrained. Recurrent Smoothing and Replay (RSR) learns its parameters in
a single pass over the train split -- one analytical update per observation, no epochs
and no validation split to select them -- while the Kalman filter tracks the level and
the slope. The coupled model then forecasts the test split: the baseline extrapolates
the rise, the LSTM adds the daily shape.

Run from the repository root:

    python examples/toy_rsr_trend.py
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
    plot_with_uncertainty,
)
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

NUM_STEPS = 24 * 40
PERIOD = 24
LEVEL = 2.0
SLOPE = 0.006
STD_NOISE = 0.1
SEED = 1
WINDOW_LEN = 24
TRAIN_SPLIT = 0.85
STATES_TO_PLOT = ["level", "trend", "lstm", "white noise"]
STRATEGY =  "replay_smoothed"
    # "cached_smoothed",
    # "replay_filtered",
    # "replay_smoothed",


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
        validation_split=0.0,
        output_col=[0],
    )
    train_data, _, test_data, all_data = data_processor.get_splits()

    model = Model(
        # A small process noise lets the slope keep adjusting instead of being
        # frozen at whatever the first three days imply.
        LocalTrend(std_error=1e-4),
        LstmNetwork(
            look_back_len=PERIOD,
            num_features=2,
            infer_len=PERIOD,
            num_hidden_unit=20,
            manual_seed=SEED,
        ),
        WhiteNoise(std_error=STD_NOISE),
    )
    # Three full days: a single day cannot separate the slope from the daily cycle,
    # and a wrong initial slope sends the level and the LSTM off in opposite
    # directions, each cancelling the other.
    model.auto_initialize_baseline_states(train_data["y"][: PERIOD * 3])

    # Online learning: the first WINDOW_LEN steps fill the first RSR window, so the
    # returned predictions and states start at that index. Each one is made before its
    # own observation is used to update both the LSTM and the baseline states.
    online_start = WINDOW_LEN
    online_mean, online_std, states = model.rsr_filter(
        data=all_data,
        start=online_start,
        window_len=WINDOW_LEN,
        end=data_processor.train_end,
        strategy=STRATEGY,
    )
    online_obs = all_data["y"][online_start : data_processor.train_end].flatten()

    # Forecast: RSR leaves the model at the last training step. The level and the slope
    # are extrapolated by the transition matrix while the LSTM, with frozen parameters,
    # supplies the daily pattern on top of them.
    model.lstm_net.num_samples = len(test_data["y"])
    model.lstm_net.eval()
    test_mean, test_std, _ = model.forecast(data=test_data)
    test_obs = test_data["y"].flatten()

    print(f"Online train MSE            : {metric.mse(online_mean, online_obs): 0.4f}")
    print(f"Test forecast MSE           : {metric.mse(test_mean, test_obs): 0.4f}")
    print(
        "Online train log-likelihood : "
        f"{metric.log_likelihood(online_mean, online_obs, online_std): 0.3f}"
    )
    print(
        "Test forecast log-likelihood: "
        f"{metric.log_likelihood(test_mean, test_obs, test_std): 0.3f}"
    )

    online_time = data_processor.data.index[online_start : data_processor.train_end]
    fig, axes = plt.subplots(1 + len(STATES_TO_PLOT), 1, figsize=(10, 11), sharex=True)
    plot_data(
        data_processor=data_processor,
        standardization=True,
        sub_plot=axes[0],
        train_label="train",
        test_label="test",
    )
    plot_with_uncertainty(
        time=online_time,
        mu=online_mean,
        std=online_std,
        color="blue",
        label=["online prediction", r"$\pm\sigma$"],
        ax=axes[0],
    )
    plot_prediction(
        data_processor=data_processor,
        mean_test_pred=test_mean,
        std_test_pred=test_std,
        sub_plot=axes[0],
        color="purple",
        test_label=["forecast", r"$\pm\sigma$"],
    )
    axes[0].set_ylabel("y")
    axes[0].legend(loc=(0.0, 1.01), ncol=5, frameon=False)

    # Hidden states estimated while learning online
    for ax, state in zip(axes[1:], STATES_TO_PLOT):
        plot_with_uncertainty(
            time=online_time,
            mu=states.get_mean(state, "posterior"),
            std=states.get_std(state, "posterior"),
            color="blue",
            ax=ax,
        )
        ax.set_ylabel(state)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
