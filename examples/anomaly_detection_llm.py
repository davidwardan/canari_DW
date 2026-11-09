"""
LLM-SKF anomaly detection: a Switching Kalman Filter whose recurrent-pattern
predictor is a pretrained time-series foundation model (Chronos-Bolt), plugged in
through the `Auxiliary` component.

The SKF switches between a normal model (`LocalTrend`) and an abnormal one
(`LocalAcceleration`); the foundation model explains the recurrent pattern in
both. It is queried once per HORIZON steps and the SKF keeps filtering in
between, appending each auxiliary posterior to its growing context.

Filtering does not start at the beginning of the series: the leading segment is
never filtered, it only serves as context for the foundation model
(`DataProcess.split_at` plus `SKF.initialize_from_context`).

Requires: `pip install chronos-forecasting torch`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from chronos import BaseChronosPipeline

from canari import DataProcess, Model, SKF, plot_skf_states
from canari.component import Auxiliary, LocalAcceleration, LocalTrend, WhiteNoise

HORIZON = 3  # foundation-model calls happen every HORIZON filtered steps
MODEL_ID = "amazon/chronos-bolt-tiny"  # -{mini,small,base} trade speed for accuracy

# Read data
data_file = "./data/toy_time_series/sine.csv"
df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.to_datetime(
    pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)[0]
)
df.index = time_series
df.index.name = "date_time"
df.columns = ["values"]

# Add a synthetic trend anomaly to the data
time_anomaly = 120
trend = np.zeros(len(df))
trend[time_anomaly:] = np.linspace(0, 1, num=len(df) - time_anomaly)
df = df.add(trend, axis=0)

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=0.3,
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

# Components. The same `Auxiliary` instance goes into both models: it owns the
# growing context, which has to stay unique across the SKF's transition models.
sigma_v = 1e-3
auxiliary = Auxiliary(predict_fn=llm_predict, horizon=HORIZON)
noise = WhiteNoise(std_error=sigma_v)

norm_model = Model(LocalTrend(), auxiliary, noise)
abnorm_model = Model(LocalAcceleration(), auxiliary, noise)

# SKF
skf = SKF(
    norm_model=norm_model,
    abnorm_model=abnorm_model,
    std_transition_error=1e-4,
    norm_to_abnorm_prob=1e-3,
)

# Initialize the baseline at the end of the context segment, and seed the
# foundation model's context with what the baseline does not explain.
skf.initialize_from_context(context_data["y"])

# Anomaly detection over the filtered segment
filter_marginal_abnorm_prob, states = skf.filter(data=filter_data)
smooth_marginal_abnorm_prob, states = skf.smoother()

# Plot the hidden states and the anomaly probability
fig, axes = plot_skf_states(
    data_processor=data_processor,
    states=states,
    states_type="posterior",
    model_prob=filter_marginal_abnorm_prob,
    color="b",
    legend_location="upper left",
    time_start_index=filter_start,  # states start where filtering started
)
for ax in axes:
    ax.axvline(df.index[time_anomaly], color="k", linestyle=":", linewidth=1)
fig.suptitle(f"LLM-SKF hidden states (H={HORIZON})", fontsize=10, y=1)
plt.tight_layout()
plt.show()
