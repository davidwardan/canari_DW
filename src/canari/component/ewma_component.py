from typing import List, Optional, Tuple
import numpy as np
from canari.component.base_component import BaseComponent


class EwmaConstraint(BaseComponent):
    """
    `EwmaConstraint` class, inheriting from Canari's `BaseComponent`.
    It softly constrains the long-term mean and drift of another hidden state, the
    `source`, while leaving its short-term variations free. The source can be any
    scalar hidden state, e.g., the output of
    :class:`~canari.component.lstm_component.LstmNetwork`, or the state of a linear
    component.

    The component tracks a fast and a slow exponentially weighted moving average (EWMA)
    of the source. At each time step, the EWMAs absorb the source's prediction, and the
    prior is conditioned on two zero-valued pseudo-observations: the slow EWMA, i.e.,
    the long-term mean, and the scaled difference between both EWMAs, i.e., the
    drift.
    The EWMAs do not contribute to the observation, and start from known zero values.

    Args:
        source (Optional[str]): Name of the hidden state to constrain.
            Defaults to "lstm".
        fast_span (Optional[float]): E-folding duration of the fast EWMA (not a pandas
            span). Defaults to 12.
        slow_span (Optional[float]): E-folding duration of the slow EWMA, larger than
            `fast_span`. Defaults to 48.
        tau_mean (Optional[float]): Standard deviation of the zero long-term mean
            pseudo-observation, in signal units. None disables it. Defaults to 0.1.
        tau_slope (Optional[float]): Standard deviation of the zero drift
            pseudo-observation, in signal units per unit time. None disables it.
            Defaults to 0.01.
        dt (Optional[float]): Time between consecutive steps, in span units.
            Defaults to 1.

    Examples:
        >>> from canari.component import EwmaConstraint
        >>> ewma = EwmaConstraint(source="lstm", fast_span=12, slow_span=48)
        >>> ewma.states_name
        ['lstm ewma fast', 'lstm ewma slow']
    """

    def __init__(
        self,
        source: Optional[str] = "lstm",
        fast_span: Optional[float] = 12.0,
        slow_span: Optional[float] = 48.0,
        tau_mean: Optional[float] = 0.1,
        tau_slope: Optional[float] = 0.01,
        dt: Optional[float] = 1.0,
    ):
        if not isinstance(source, str) or not source.strip():
            raise ValueError("EwmaConstraint's source must be a nonempty state name.")
        if not (
            np.all(np.isfinite([fast_span, slow_span, dt]))
            and 0 < fast_span < slow_span
            and dt > 0
        ):
            raise ValueError(
                "EwmaConstraint expects finite 0 < fast_span < slow_span and dt > 0."
            )
        for tau in (tau_mean, tau_slope):
            if tau is None:
                continue
            with np.errstate(over="ignore", under="ignore"):
                variance = np.float64(tau) ** 2
            valid = np.isfinite(tau) and tau > 0 and np.isfinite(variance)
            if not (valid and variance > 0):
                raise ValueError(
                    "EwmaConstraint's tolerances must be None or positive, with a "
                    "finite nonzero square."
                )
        self.source = source
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.tau_mean = tau_mean
        self.tau_slope = tau_slope
        self.dt = dt
        durations = np.array([fast_span, slow_span], dtype=float)
        with np.errstate(
            over="ignore", divide="ignore", invalid="ignore", under="ignore"
        ):
            self.rhos = np.exp(-dt / durations)
            self.weights = -np.expm1(-dt / durations)
            # Delay of each EWMA behind a steady ramp, used to recover its slope.
            self.lags = dt * self.rhos / self.weights
        if not (np.all(np.isfinite(self.lags)) and self.lags[1] > self.lags[0]):
            raise ValueError("EwmaConstraint's spans and dt must give distinct lags.")
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "ewma"

    def initialize_num_states(self):
        self._num_states = 2

    def initialize_states_name(self):
        self._states_name = [f"{self.source} ewma fast", f"{self.source} ewma slow"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.diag(self.rhos)

    def initialize_observation_matrix(self):
        self._observation_matrix = np.zeros((1, self._num_states))

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = np.zeros((self._num_states, self._num_states))

    def initialize_mu_states(self):
        self._mu_states = np.zeros((self._num_states, 1))

    def initialize_var_states(self):
        self._var_states = np.zeros((self._num_states, 1))

    def constraint_observation(
        self, num_states: int, memory_indices: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Observation matrix and noise variances of the zero-valued pseudo-observations.

        Args:
            num_states (int): Number of hidden states in the model.
            memory_indices (List[int]): Model indices of the fast and slow EWMAs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: One row per enabled pseudo-observation, and
            the corresponding noise variances.
        """

        fast, slow = memory_indices
        rows, variances = [], []
        if self.tau_mean is not None:
            row = np.zeros(num_states)
            row[slow] = 1
            rows.append(row)
            variances.append(self.tau_mean**2)
        if self.tau_slope is not None:
            row = np.zeros(num_states)
            row[[fast, slow]] = np.array([1, -1]) / (self.lags[1] - self.lags[0])
            rows.append(row)
            variances.append(self.tau_slope**2)
        return np.reshape(rows, (-1, num_states)), np.array(variances)
