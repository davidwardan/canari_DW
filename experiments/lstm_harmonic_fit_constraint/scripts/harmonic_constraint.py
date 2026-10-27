"""Exponentially weighted joint fit of a line and seasonal harmonics to the LSTM output.

At time t the fit minimizes sum_{i<=t} rho^(t-i) (r_i - theta' f(i - t))^2 with
f(j) = [1, j, cos(w j), sin(w j), ..., cos(K w j), sin(K w j)], w = 2 pi / period.
theta = [alpha, beta, a_1, b_1, ...]: alpha is the line's level at t and beta its slope;
the harmonic coefficients absorb the seasonal wave. In steady state the fit is the
recursion theta_t = A theta_{t-1} + h r_t, so theta is kept as 2 + 2K joint-state
memories that CANARI's EwmaConstraint machinery couples, conditions, smooths and replays
unchanged. Only alpha and beta are pseudo-observed as zero. See ../planning/maths.md.
"""

from typing import Optional

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from canari.component import EwmaConstraint
from canari.component.base_component import BaseComponent


def basis(lags, period: float, num_harmonics: int) -> np.ndarray:
    """Columns f(j) for each lag j relative to the fit's time origin."""

    lags = np.atleast_1d(np.asarray(lags, dtype=float))
    rows = [np.ones_like(lags), lags]
    for k in range(1, num_harmonics + 1):
        rows += [np.cos(2 * np.pi * k * lags / period), np.sin(2 * np.pi * k * lags / period)]
    return np.array(rows)


def shift_matrix(period: float, num_harmonics: int) -> np.ndarray:
    """L with f(j + 1) = L f(j)."""

    blocks = [np.array([[1.0, 0.0], [1.0, 1.0]])]
    for k in range(1, num_harmonics + 1):
        c, s = np.cos(2 * np.pi * k / period), np.sin(2 * np.pi * k / period)
        blocks.append(np.array([[c, -s], [s, c]]))
    size = 2 * len(blocks)
    matrix = np.zeros((size, size))
    for index, block in enumerate(blocks):
        matrix[2 * index:2 * index + 2, 2 * index:2 * index + 2] = block
    return matrix


def steady_state(period: float, num_harmonics: int, rho: float):
    """Gain h and memory transition A of the steady-state discounted fit."""

    shift_inverse = np.linalg.inv(shift_matrix(period, num_harmonics))
    origin = basis(0.0, period, num_harmonics)
    gram = solve_discrete_lyapunov(np.sqrt(rho) * shift_inverse, origin @ origin.T)
    gain = np.linalg.solve(gram, origin)
    transition = (np.eye(len(origin)) - gain @ origin.T) @ shift_matrix(period, num_harmonics).T
    return gain, transition


def discounted_fit(values, period: float, num_harmonics: int, rho: float) -> np.ndarray:
    """Exact weighted fit at the last time step of a finite sequence."""

    values = np.asarray(values, dtype=float).ravel()
    lags = np.arange(len(values)) - (len(values) - 1)
    design = basis(lags, period, num_harmonics)
    weights = rho ** (-lags)
    return np.linalg.solve((design * weights) @ design.T, (design * weights) @ values)


class HarmonicTrendConstraint(EwmaConstraint):
    """Discounted line-plus-harmonics fit of `source`; level and slope softly set to zero.

    Args:
        source: Name of the scalar state to constrain, e.g. ``"lstm"``.
        period: Seasonal period in time steps.
        num_harmonics: Number K of harmonics of the period that the fit absorbs.
        span: Discount e-folding duration, rho = exp(-1 / span).
        tau_level: Std of the zero-level pseudo-observation, in signal units.
        tau_slope: Std of the zero-slope pseudo-observation, in signal/time units.
        name: Unique component name; defaults to ``"{source} harmonic"``.

    The memories start at known zero; set them to `discounted_fit` of the source's
    history to avoid the start-up transient.
    """

    def __init__(
        self,
        source: str = "lstm",
        period: float = 52.0,
        num_harmonics: int = 1,
        span: float = 208.0,
        tau_level: float = 0.05,
        tau_slope: float = 0.0005,
        name: Optional[str] = None,
    ):
        if not (np.isfinite([period, span, tau_level, tau_slope]).all()
                and period > 2 * num_harmonics and span > 0
                and tau_level > 0 and tau_slope > 0 and num_harmonics >= 0):
            raise ValueError("Expected period > 2K, K >= 0 and positive span and tolerances")
        self.source = source
        self.period = period
        self.num_harmonics = int(num_harmonics)
        self.span = span
        self.rho = float(np.exp(-1 / span))
        self.tau_level = tau_level
        self.tau_slope = tau_slope
        self.name = name if name is not None else f"{source} harmonic"
        self.gain, self.memory_transition = steady_state(period, self.num_harmonics, self.rho)
        BaseComponent.__init__(self)

    def initialize_num_states(self):
        self._num_states = 2 + 2 * self.num_harmonics

    def initialize_states_name(self):
        names = [f"{self.name} level", f"{self.name} slope"]
        for k in range(1, self.num_harmonics + 1):
            names += [f"{self.name} cos{k}", f"{self.name} sin{k}"]
        self._states_name = names

    def initialize_transition_matrix(self):
        self._transition_matrix = self.memory_transition

    def initialize_observation_matrix(self):
        self._observation_matrix = np.zeros((1, self._num_states))

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = np.zeros((self._num_states, self._num_states))

    def initialize_mu_states(self):
        self._mu_states = np.zeros((self._num_states, 1))

    def initialize_var_states(self):
        self._var_states = np.zeros((self._num_states, 1))

    def coupling_matrix(self, states_name):
        """Map uncoupled predictions to memories containing the current source."""
        matrix = np.eye(len(states_name))
        indices = [states_name.index(name) for name in self.states_name]
        matrix[indices, states_name.index(self.source)] = self.gain.ravel()
        return matrix

    def constraint_observation(self, states_name):
        """Zero-target rows selecting the fitted level and slope."""
        rows = np.zeros((2, len(states_name)))
        rows[[0, 1], [states_name.index(name) for name in self.states_name[:2]]] = 1
        return rows, np.diag([self.tau_level**2, self.tau_slope**2])
