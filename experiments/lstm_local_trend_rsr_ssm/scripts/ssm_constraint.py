"""Auxiliary local-trend SSM on the LSTM output, with zero level and slope pseudo-observations.

The auxiliary model treats the LSTM output as a local trend plus a white residual that
absorbs the periodic pattern,

    R_t = L_t + E_t,        E_t ~ N(0, std_residual^2),
    Z_t = A Z_{t-1} + W_t,  Z_t = [L_t, S_t],  W_t ~ N(0, Q),

with CANARI's LocalTrend matrices A and Q. Its steady-state Kalman filter is run on the
LSTM output, and the filtered level and slope are kept as two joint-state memories,

    M_t = F M_{t-1} + k R_t,  F = (I - k h') A,

which CANARI's EwmaConstraint machinery couples, conditions, smooths and replays
unchanged. The pseudo-observation 0 = M_t + Delta_t uses the auxiliary filter's
steady-state posterior covariance P as the covariance of Delta_t. This module does
not patch CANARI.
"""

from typing import Optional

import numpy as np
from scipy.linalg import solve_discrete_are

from canari.component import EwmaConstraint
from canari.component.base_component import BaseComponent

TRANSITION = np.array([[1.0, 1.0], [0.0, 1.0]])
OBSERVATION = np.array([[1.0, 0.0]])
# CANARI LocalTrend process noise divided by std_error^2
NOISE_SHAPE = np.array([[0.25, 0.5], [0.5, 1.0]])


def steady_state(std_error: float, std_residual: float):
    """Steady-state Kalman gain k, posterior covariance P, and memory transition F."""

    prior_var = solve_discrete_are(
        TRANSITION.T,
        OBSERVATION.T,
        std_error**2 * NOISE_SHAPE,
        np.array([[std_residual**2]]),
    )
    gain = prior_var @ OBSERVATION.T / (
        OBSERVATION @ prior_var @ OBSERVATION.T + std_residual**2
    )
    residual_map = np.eye(2) - gain @ OBSERVATION
    posterior_var = residual_map @ prior_var
    return gain, (posterior_var + posterior_var.T) / 2, residual_map @ TRANSITION


def step_response_time(std_ratio: float) -> int:
    """First step at which the filtered level of a unit step reaches 1 - 1/e."""

    gain, _, memory_transition = steady_state(std_ratio, 1.0)
    memory = np.zeros((2, 1))
    for step in range(1, 10**6):
        memory = memory_transition @ memory + gain
        if memory[0, 0] >= 1 - np.exp(-1):
            return step
    raise ValueError("Step response too slow")


def std_error_for_response_time(std_residual: float, steps: int) -> float:
    """Process-noise std giving a level step response of `steps` (gains depend on the ratio)."""

    low, high = -12.0, 0.0  # log10 of std_error / std_residual
    for _ in range(60):
        middle = (low + high) / 2
        if step_response_time(10**middle) > steps:
            low = middle
        else:
            high = middle
    return std_residual * 10**high


def detrended_std(values: np.ndarray) -> float:
    """Standard deviation around the least-squares line through the finite values."""

    values = np.asarray(values, dtype=float).ravel()
    time = np.arange(len(values))
    finite = np.isfinite(values)
    coefficients = np.polyfit(time[finite], values[finite], 1)
    return float(np.std(values[finite] - np.polyval(coefficients, time[finite])))


class LocalTrendConstraint(EwmaConstraint):
    """Filtered level and slope of `source`, softly pseudo-observed as zero.

    Args:
        source: Name of the scalar state to constrain, e.g. ``"lstm"``.
        std_error: Process-noise std of the auxiliary local trend (LocalTrend form).
        std_residual: Std of the auxiliary residual that absorbs the periodic part.
        name: Unique component name; defaults to ``"{source} aux"``.

    The memories start at known zero, which is the auxiliary filter's steady state
    with a zero prior mean. With a deterministic source they are the plain Kalman
    estimates; a stochastic source is pulled towards a zero level and slope.
    """

    def __init__(
        self,
        source: str = "lstm",
        std_error: float = 1e-5,
        std_residual: float = 0.7,
        name: Optional[str] = None,
    ):
        if not (np.isfinite([std_error, std_residual]).all()
                and std_error > 0 and std_residual > 0):
            raise ValueError("Expected positive finite std_error and std_residual")
        self.source = source
        self.std_error = std_error
        self.std_residual = std_residual
        self.name = name if name is not None else f"{source} aux"
        self.gain, self.posterior_var, self.memory_transition = steady_state(
            std_error, std_residual
        )
        BaseComponent.__init__(self)

    def initialize_states_name(self):
        self._states_name = [f"{self.name} level", f"{self.name} slope"]

    def initialize_transition_matrix(self):
        self._transition_matrix = self.memory_transition

    def coupling_matrix(self, states_name):
        """Map uncoupled predictions to memories containing the current source."""
        matrix = np.eye(len(states_name))
        indices = [states_name.index(name) for name in self.states_name]
        matrix[indices, states_name.index(self.source)] = self.gain.ravel()
        return matrix

    def constraint_observation(self, states_name):
        """Zero-target rows selecting the memories, with the auxiliary posterior covariance."""
        rows = np.zeros((2, len(states_name)))
        rows[[0, 1], [states_name.index(name) for name in self.states_name]] = 1
        return rows, self.posterior_var
