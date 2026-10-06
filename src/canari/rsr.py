"""
Recurrent Smoothing and Replay (RSR) for online LSTM learning.

RSR keeps the LSTM of a coupled LSTM/SSM model learning while the series is filtered.
It walks overlapping windows of `window_len + 1` observations: the LSTM parameters are
updated analytically at every step of a window, the window is then smoothed backwards,
and the next window restarts one step later from the smoothed recurrent memory. Every
time step therefore receives its own one-step-ahead prediction, made before its
observation is used.

Four strategies differ in how the state-space model is handled while the LSTM replays
the older steps of a window. They all perform the same number of LSTM updates and all
smooth the LSTM's own cell and hidden states:

- `"cached_filtered"`: the older steps train the LSTM against the baseline prior
  recorded when each step first arrived. The state-space model is never replayed; it
  advances chronologically once per window and is not smoothed.
- `"cached_smoothed"`: same, but the baseline is the smoothed estimate computed after
  the preceding window.
- `"replay_filtered"`: the whole window is re-filtered, starting from the causal
  posterior saved just before the window.
- `"replay_smoothed"`: the whole window is re-filtered, starting from the preceding
  window's smoothed boundary.

The cached baselines are held fixed during a replay, and a smoothed boundary already
carries information from the overlap. These are algorithm variants, not exact Bayesian
inference over the window.
"""

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from canari import common
from canari.data_struct import StatesHistory

STRATEGIES = (
    "cached_filtered",
    "cached_smoothed",
    "replay_filtered",
    "replay_smoothed",
)

_STATE_FIELDS = (
    "mu_prior",
    "var_prior",
    "mu_posterior",
    "var_posterior",
    "mu_smooth",
    "var_smooth",
    "cov_states",
)


class _LookBackBuffer:
    """Keep smoothed LSTM outputs that later overlapping windows read back."""

    def __init__(self, look_back_len: int, num_steps: int):
        self.look_back_len = look_back_len
        self.pad = look_back_len - 1
        self.mu = np.zeros(num_steps + self.pad, dtype=np.float32)
        self.var = np.ones(num_steps + self.pad, dtype=np.float32)

    def seed(self, mu: np.ndarray, var: np.ndarray) -> None:
        num_values = min(self.pad, len(mu))
        if num_values:
            self.mu[self.pad - num_values : self.pad] = mu[-num_values:]
            self.var[self.pad - num_values : self.pad] = var[-num_values:]

    def store(self, time_step: int, mu: np.ndarray, var: np.ndarray) -> None:
        self.mu[time_step + self.pad] = mu[0]
        self.var[time_step + self.pad] = var[0]

    def get(self, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
        stop = time_step + self.pad
        start = stop - self.look_back_len
        return self.mu[start:stop], self.var[start:stop]


def _slice_data(
    data: Dict[str, np.ndarray], start: int, end: int
) -> Dict[str, np.ndarray]:
    return {"x": data["x"][start:end], "y": data["y"][start:end]}


def _validate_inputs(model, data, start: int, window_len: int, end: int, strategy: str):
    if model.lstm_net is None:
        raise ValueError("rsr_filter requires an LstmNetwork component")
    if not model.lstm_net.smooth:
        raise ValueError("rsr_filter requires LstmNetwork(smoother=True)")
    if strategy not in STRATEGIES:
        raise ValueError(f"Incorrect strategy: choose from {STRATEGIES}")
    if window_len < 1:
        raise ValueError("window_len must be at least 1")
    if start < window_len:
        raise ValueError("start must be greater than or equal to window_len")
    if not start < end <= len(data["y"]):
        raise ValueError("expected start < end <= len(data['y'])")


def _new_states_history(states_name: List[str]) -> StatesHistory:
    history = StatesHistory()
    history.initialize(states_name)
    return history


def _extend_states(
    target: StatesHistory,
    source: StatesHistory,
    start: int,
    stop: Optional[int] = None,
) -> None:
    """Append the entries `start:stop` of one hidden-state history to another."""

    for name in _STATE_FIELDS:
        getattr(target, name).extend(
            value.copy() for value in getattr(source, name)[start:stop]
        )


def _baseline_moments(
    model, mu_states: np.ndarray, var_states: np.ndarray
) -> Tuple[float, float]:
    """
    Observation moments contributed by everything except the LSTM.

    The white noise state is left out of `mu_states` and `var_states` and re-added at
    its prior variance: a replayed observation always carries fresh measurement noise,
    never the noise realization already inferred for that time step.
    """

    observation_matrix = model.observation_matrix.copy()
    observation_matrix[0, model.get_states_index("lstm")] = 0
    noise_var = 0.0
    if "white noise" in model.states_name:
        noise_index = model.get_states_index("white noise")
        observation_matrix[0, noise_index] = 0
        noise_var = model.process_noise_matrix[noise_index, noise_index]

    mu_baseline = observation_matrix @ mu_states
    var_baseline = observation_matrix @ var_states @ observation_matrix.T
    return float(mu_baseline.item()), float(var_baseline.item() + noise_var)


def _cached_update(model, x, y, mu_baseline: float, var_baseline: float) -> None:
    """
    Train the LSTM on one replayed step without touching the state-space model.

    The baseline stands in for the rest of the observation model, so the update is the
    same analytical message `Model.filter` sends, computed from a frozen baseline
    instead of from a re-run Kalman step.
    """

    mu_input, var_input = common.prepare_lstm_input(model.lstm_output_history, x)
    mu_lstm, var_lstm = model.lstm_net.forward(mu_x=mu_input, var_x=var_input)
    mu_lstm = float(np.asarray(mu_lstm).item())
    var_lstm = float(np.asarray(var_lstm).item())

    var_obs = var_lstm + var_baseline
    delta_mu = (float(np.asarray(y).item()) - mu_baseline - mu_lstm) / var_obs
    delta_var = -1 / var_obs
    if np.isnan(delta_mu):
        # A missing observation carries no information, as in `Model.filter`.
        delta_mu, delta_var = 0.0, 0.0
    model.lstm_net.update_param(np.float32([delta_mu]), np.float32(delta_var))
    model.lstm_output_history.update(
        np.float32(mu_lstm + var_lstm * delta_mu),
        np.float32(var_lstm + var_lstm**2 * delta_var),
    )


def _smooth_ssm_window(model) -> None:
    for time_step in reversed(range(len(model.states.mu_smooth) - 1)):
        model.rts_smoother(time_step)


def _warm_up_model(model, data, end: int, window_len: int):
    """Carry the model to the first window with frozen LSTM parameters."""

    if end == 0:
        return None

    # The SLSTM sizes its internal sequence buffers the first time it runs, and a later
    # increase is not honoured. The warm-up must therefore reserve room for the longest
    # sequence the online loop will feed it, otherwise the first window overruns them.
    model.lstm_net.num_samples = max(end, window_len + 1)
    model.lstm_net.eval()
    _, _, states = model.filter(_slice_data(data, 0, end), train_lstm=False)
    seed = (
        states.get_mean("lstm", "posterior"),
        states.get_std("lstm", "posterior") ** 2,
    )
    model.lstm_net.smoother()
    model.lstm_net.set_lstm_states(model.lstm_net.get_lstm_states(end - 1))
    return seed


def rsr_filter(
    model,
    data: Dict[str, np.ndarray],
    start: int,
    window_len: int,
    strategy: str = "replay_smoothed",
    end: Optional[int] = None,
    include_warmup: bool = False,
) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
    """Run RSR online LSTM learning and return one prediction per step.

    By default the returned sequence starts at ``start``.  When
    ``include_warmup`` is true, return the first window's states and predictions
    as well, so a run started at ``start=window_len`` has a complete history from
    its first filtered observation.
    """

    end = len(data["y"]) if end is None else end
    _validate_inputs(model, data, start, window_len, end, strategy)

    first_window = start - window_len
    seed = _warm_up_model(model, data, first_window, window_len)
    num_steps = end - start
    buffer = _LookBackBuffer(model.lstm_net.lstm_look_back_len, num_steps + window_len)
    if seed is not None:
        buffer.seed(*seed)

    # Hidden states at their first arrival, from `first_window` onwards. Cached
    # strategies read their baselines here, `replay_filtered` its window boundary.
    chronological = _new_states_history(model.states_name)
    smooth_baselines: Dict[int, Tuple[float, float]] = {}
    means, stds = [], []

    model.lstm_net.num_samples = window_len + 1
    for step in range(num_steps):
        window_start = first_window + step
        newest = window_start + window_len
        model.lstm_net.train()

        if step > 0 and strategy.startswith("cached"):
            # The state-space model stays at the preceding chronological posterior; only
            # the LSTM revisits the older steps of the window.
            model.set_states(
                chronological.mu_posterior[-1], chronological.var_posterior[-1]
            )
            for time_step in range(window_start, newest):
                index = time_step - first_window
                if strategy == "cached_filtered":
                    baseline = _baseline_moments(
                        model,
                        chronological.mu_prior[index],
                        chronological.var_prior[index],
                    )
                else:
                    baseline = smooth_baselines[index]
                _cached_update(
                    model, data["x"][time_step], data["y"][time_step], *baseline
                )
            window = _slice_data(data, newest, newest + 1)
        else:
            if step > 0 and strategy == "replay_filtered":
                model.set_states(
                    chronological.mu_posterior[step - 1],
                    chronological.var_posterior[step - 1],
                )
            window = _slice_data(data, window_start, newest + 1)

        mean, std, states = model.filter(window, train_lstm=True)
        if include_warmup and step == 0:
            means.extend(mean)
            stds.extend(std)
        else:
            means.append(mean[-1])
            stds.append(std[-1])
        _extend_states(chronological, states, 0 if step == 0 else -1)

        filtered_lstm_states = copy.deepcopy(model.lstm_net.get_lstm_states())
        filtered_states = (model.mu_states.copy(), model.var_states.copy())

        if strategy == "cached_smoothed":
            # The window was never re-filtered, so smooth its chronological estimates.
            model.states = _new_states_history(model.states_name)
            _extend_states(model.states, chronological, step, step + window_len + 1)
        if strategy != "cached_filtered":
            _smooth_ssm_window(model)
        if strategy == "cached_smoothed":
            smooth_baselines = {
                step + index: _baseline_moments(
                    model,
                    model.states.mu_smooth[index],
                    model.states.var_smooth[index],
                )
                for index in range(window_len + 1)
            }

        smooth_mu, smooth_var = model.lstm_net.smoother()
        buffer.store(step, np.asarray(smooth_mu), np.asarray(smooth_var))

        if step < num_steps - 1:
            if strategy == "replay_smoothed":
                model.set_states(model.states.mu_smooth[0], model.states.var_smooth[0])
            look_back_mu, look_back_var = buffer.get(step + 1)
            model.lstm_output_history.set(look_back_mu, look_back_var)
            model.lstm_net.set_lstm_states(model.lstm_net.get_lstm_states(0))
        else:
            model.set_states(*filtered_states)
            model.lstm_net.set_lstm_states(filtered_lstm_states)

    states_history = _new_states_history(model.states_name)
    _extend_states(states_history, chronological, 0 if include_warmup else window_len)
    return np.asarray(means).flatten(), np.asarray(stds).flatten(), states_history
