"""
LLM-SKF anomaly detection on a real time series, with a synthetic anomaly injected
so that the detection can be scored against a known ground truth.

The series is `ts70` of the weekly HQ benchmark: 860 real weekly observations
(2009-2026), dominated by noise and short-range dependence rather than by a clean
seasonal pattern -- which is what makes it a harder, more realistic test than the
toy sine of `anomaly_detection_llm.py`.

A slow linear drift is added partway through, the kind of change that is invisible
step to step and only becomes clear over months. The leading part of the series is
never filtered: it only serves as context for the foundation model.

Requires: `pip install chronos-forecasting torch`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from chronos import BaseChronosPipeline

from canari import DataProcess, Model, SKF, plot_skf_states
from canari.component import Auxiliary, LocalAcceleration, LocalTrend, WhiteNoise

HORIZON = 104  # foundation-model calls happen every HORIZON filtered steps
MODEL_ID = "amazon/chronos-bolt-base"  # -{mini,small,base} trade speed for accuracy
SERIES = "ts70"

# Read data
values = pd.read_csv("./data/hq_benchmark_data/weekly/weekly_values.csv")[SERIES]
time_series = pd.to_datetime(
    pd.read_csv("./data/hq_benchmark_data/weekly/weekly_datetimes.csv")[SERIES]
)
df = pd.DataFrame({"values": values.to_numpy()}, index=time_series)
df.index.name = "date_time"
df = df[df.index.notna()]

# Add a synthetic drift anomaly to the data: a slope change of ANOMALY_SLOPE per
# week, i.e. a drift that only becomes visible over the following months.
anomaly_index = int(len(df) * 0.7)
ANOMALY_SLOPE = 1.0 / 52  # ~1 standard deviation per year
drift = np.zeros(len(df))
drift[anomaly_index:] = ANOMALY_SLOPE * np.arange(len(df) - anomaly_index)
df["values"] = df["values"].to_numpy() + drift
anomaly_time = df.index[anomaly_index]

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=0.4,
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
sigma_v = 0.2
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

# Score the detection against the known onset
onset = anomaly_index - filter_start
detected = np.flatnonzero(filter_marginal_abnorm_prob > 0.5)
print(f"Series              : {SERIES}, {len(df)} weekly points")
print(f"Context / filtered  : {filter_start} / {len(filter_data['y'])} steps")
print(f"Anomaly onset       : {anomaly_time.date()} (index {anomaly_index})")
print(f"Max Pr before onset :{filter_marginal_abnorm_prob[:onset].max(): 0.4f}")
print(f"Max Pr after onset  :{filter_marginal_abnorm_prob[onset:].max(): 0.4f}")
if detected.size:
    first = detected[0] + filter_start
    print(f"First Pr>0.5        : index {first} ({df.index[first].date()}), "
          f"{first - anomaly_index} steps after onset")
else:
    print("First Pr>0.5        : never")

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
    ax.axvline(anomaly_time, color="k", linestyle=":", linewidth=1)
fig.suptitle(f"LLM-SKF on {SERIES} with injected drift (H={HORIZON})", fontsize=10, y=1)
plt.tight_layout()
plt.show()
