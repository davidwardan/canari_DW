"""
Constrain the LSTM's long-term mean and drift with EwmaConstraint.

The LSTM is trained offline on a trending periodic series, with the validation split
selecting the epoch. From the same memory, the test split is then filtered twice:
with frozen LSTM parameters, and while the LSTM keeps learning online. The EWMA
constraint applies in all three cases. Its two memories, the fast and slow EWMAs of
the LSTM output, are shown with the other hidden states for the online run.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytagi import metric

from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import EwmaConstraint, LocalTrend, LstmNetwork, WhiteNoise

PERIOD = 12
NUM_STEPS = 20 * PERIOD
NUM_EPOCHS = 200
SEED = 11

# Data: a slow linear baseline, a 12-step cycle, and white noise
rng = np.random.default_rng(SEED)
time = np.arange(NUM_STEPS)
values = (
    0.003 * time
    + np.sin(2 * np.pi * time / PERIOD)
    + rng.normal(0, 0.1, NUM_STEPS)
)
df = pd.DataFrame(
    {"values": values},
    index=pd.date_range("2025-01-01", periods=NUM_STEPS, freq="D", name="date_time"),
)
data_processor = DataProcess(
    data=df,
    train_split=0.5,
    validation_split=0.2,
    output_col=[0],
)
train_data, validation_data, test_data, _ = data_processor.get_splits()

# Model: the EWMAs of the LSTM output must stay near zero, with a near-zero drift
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=PERIOD,
        infer_len=PERIOD,
        num_hidden_unit=16,
        manual_seed=SEED,
    ),
    WhiteNoise(std_error=0.1),
    EwmaConstraint(
        source="lstm",
        fast_span=PERIOD,
        slow_span=4 * PERIOD,
        tau_mean=0.1,
        tau_slope=0.01,
    ),
)
model.auto_initialize_baseline_states(train_data["y"][: 2 * PERIOD])

# Offline training: the constraint enters every training update
validation_obs = validation_data["y"].flatten()
for epoch in range(NUM_EPOCHS):
    mu_validation_preds, std_validation_preds, _ = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
        white_noise_decay=False,
    )
    model.early_stopping(
        evaluate_metric=metric.mse(mu_validation_preds, validation_obs),
        current_epoch=epoch,
        max_epoch=NUM_EPOCHS,
    )
    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
    if model.stop_training:
        break

print(f"Optimal epoch      : {model.optimal_epoch}")
print(f"Validation MSE     : {model.early_stop_metric: 0.4f}")

# Test: filter from the memory at the end of validation, frozen then online
memory = model.get_memory(time_step=data_processor.test_start - 1)
test_obs = test_data["y"].flatten()
results = {}
for label, train_lstm in (("frozen", False), ("online", True)):
    model.set_memory(memory=memory)
    results[label] = model.filter(test_data, train_lstm=train_lstm)
    mu_test_preds, std_test_preds, _ = results[label]
    mse = metric.mse(mu_test_preds, test_obs)
    log_lik = metric.log_likelihood(mu_test_preds, test_obs, std_test_preds)
    print(f"Test MSE, {label:6} : {mse: 0.4f}   log-lik: {log_lik: 0.3f}")

# Plot: data and predictions, then the online run's hidden states
fig, ax = plt.subplots(figsize=(10, 4))
plot_data(data_processor=data_processor, standardization=True, sub_plot=ax)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds_optim,
    std_validation_pred=std_validation_preds_optim,
    sub_plot=ax,
    validation_label=["validation forecast", r"$\pm\sigma$"],
)
for label, color in (("frozen", "tab:orange"), ("online", "purple")):
    plot_prediction(
        data_processor=data_processor,
        mean_test_pred=results[label][0],
        std_test_pred=results[label][1],
        sub_plot=ax,
        color=color,
        test_label=[f"{label} prediction", r"$\pm\sigma$"],
    )
ax.legend(loc=(0.0, 1.01), ncol=4, frameon=False)
fig.tight_layout()

plot_states(
    data_processor=data_processor,
    states=results["online"][2],
    states_to_plot=["level", "lstm", "lstm ewma fast", "lstm ewma slow"],
    standardization=True,
    time_start_index=data_processor.test_start,
)
plt.show()
