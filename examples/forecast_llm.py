"""
LLM-SSM forecasting: a pretrained time-series foundation model (Chronos-Bolt)
plugged into a state-space model through the `Auxiliary` component.

The foundation model replaces the Bayesian LSTM: it supplies the mean and
variance of the recurrent-pattern state, while the SSM keeps estimating the
baseline and the observation noise. Three knobs matter:

    - where filtering starts: everything before it is never filtered, it only
      serves as context for the foundation model (`DataProcess.split_at` plus
      `Model.initialize_from_context`);
    - the context, which then grows by one point at every filtered time step and
      always ends with the latest auxiliary posterior;
    - the horizon `H`, the number of steps the foundation model forecasts per
      call, i.e. how many steps the SSM filters before querying it again.

Requires: `pip install chronos-forecasting torch`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from chronos import BaseChronosPipeline

from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import Auxiliary, LocalTrend, WhiteNoise

HORIZON = 1  # foundation-model calls happen every HORIZON filtered steps
MODEL_ID = "amazon/chronos-bolt-tiny"  # -{mini,small,base} trade speed for accuracy

# Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.to_datetime(
    pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)[0]
)
df.index = time_series
df.index.name = "date_time"
df.columns = ["values"]
df = df.iloc[:300]  # keep the run tractable on CPU

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=0.8,
    validation_split=0.1,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Foundation model
pipeline = BaseChronosPipeline.from_pretrained(
    MODEL_ID, device_map="cpu", dtype=torch.float32
)


def llm_predict(context: np.ndarray, horizon: int):
    """Forecast `horizon` steps from `context`; return per-step mean and variance."""

    quantiles, mean = pipeline.predict_quantiles(
        inputs=torch.tensor(context, dtype=torch.float32),
        prediction_length=horizon,
        quantile_levels=[0.1587, 0.8413],  # -1 sigma, +1 sigma
    )
    # Mean for the location, and the +/-1 sigma quantile spread for the scale: the
    # two moments the Kalman update needs from the foundation model's forecast.
    mu = mean[0].cpu().numpy()  # (horizon,)
    q = quantiles[0].cpu().numpy()  # (horizon, 2)
    std = (q[:, 1] - q[:, 0]) / 2.0
    return mu, np.maximum(std**2, 1e-6)


# Split the series: the training portion is never filtered, it is handed to the
# foundation model as context. Filtering starts at the validation split.
filter_start = data_processor.validation_start
context_data, filter_data = DataProcess.split_at(all_data, filter_start)

# Model
sigma_v = 0.2
model = Model(
    LocalTrend(),
    Auxiliary(predict_fn=llm_predict, horizon=HORIZON),
    WhiteNoise(std_error=sigma_v),
)

# Initialize the baseline at the end of the context segment, and seed the
# foundation model's context with what the baseline does not explain.
model.initialize_from_context(context_data["y"])

# Filter the remaining segment. The foundation model is frozen, so a single pass
# is enough -- no training loop, unlike the LSTM component.
mu_preds, std_preds, states = model.filter(data=filter_data)

# Unstandardize the predictions, then score the validation and test splits
mu_preds = normalizer.unstandardize(
    mu_preds,
    data_processor.scale_const_mean[output_col],
    data_processor.scale_const_std[output_col],
)
std_preds = normalizer.unstandardize_std(
    std_preds, data_processor.scale_const_std[output_col]
)
num_validation = data_processor.test_start - filter_start
mu_validation_preds, std_validation_preds = (
    mu_preds[:num_validation],
    std_preds[:num_validation],
)
mu_test_preds, std_test_preds = mu_preds[num_validation:], std_preds[num_validation:]

validation_obs = data_processor.get_data("validation").flatten()
test_obs = data_processor.get_data("test").flatten()
print(f"Validation MSE      :{metric.mse(mu_validation_preds, validation_obs): 0.4f}")
print(f"Test MSE            :{metric.mse(mu_test_preds, test_obs): 0.4f}")
print(
    f"Test Log-Lik        :"
    f"{metric.log_likelihood(mu_test_preds, test_obs, std_test_preds): 0.2f}"
)

# Plot the predictions
fig, ax = plt.subplots(figsize=(10, 6))
plot_data(
    data_processor=data_processor,
    standardization=False,
    plot_column=output_col,
    validation_label="y",
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds,
    std_validation_pred=std_validation_preds,
    validation_label=[r"$\mu$", r"$\pm\sigma$"],
)
plot_prediction(
    data_processor=data_processor,
    mean_test_pred=mu_test_preds,
    std_test_pred=std_test_preds,
    test_label=[r"$\mu^{\prime}$", r"$\pm\sigma^{\prime}$"],
    color="purple",
)
plt.legend(loc=(0.1, 1.01), ncol=6, fontsize=12)
plt.tight_layout()

# Plot the hidden states
fig, axes = plot_states(
    data_processor=data_processor,
    states=states,
    states_type="posterior",
    standardization=True,
    color="b",
    legend_location="upper left",
    time_start_index=filter_start,  # states start where filtering started
)
fig.suptitle(f"LLM-SSM hidden states (H={HORIZON})", fontsize=10, y=1)
plt.tight_layout()
plt.show()
