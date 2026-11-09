from typing import Callable, Optional, Sequence, Tuple
import numpy as np
from canari.component.base_component import BaseComponent


PredictFn = Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray]]


class Auxiliary(BaseComponent):
    """
    `Auxiliary` class, inheriting from Canari's `BaseComponent`.
    Plugs an external forecaster -- typically a pretrained time-series foundation
    model (Chronos, TimesFM, Moirai, ...) -- into the state-space model, filling
    the same slot as :class:`~canari.component.lstm_component.LstmNetwork`.

    The component owns a **growing context**: the posterior means of the auxiliary
    hidden state, one appended per filtered time step. The external model therefore
    sees a context that lengthens as the filter advances, always ending with the
    most recently assimilated value.

    Predictions are issued in blocks of :attr:`horizon` steps. At the block origin,
    ``predict_fn`` is called once with the current context and returns `H`
    means/variances; the state-space model then keeps filtering through those `H`
    steps -- appending posteriors to the context -- before the next call. With
    ``horizon=1`` the external model is queried at every time step.

    Args:
        predict_fn (Callable): External forecaster with signature
            ``(context, horizon) -> (mu, var)``, where ``context`` is a 1-D array of
            past values and ``mu``/``var`` are 1-D arrays of length ``horizon``.
        horizon (Optional[int]): Number of steps forecast per call, i.e. the number
            of time steps the state-space model filters before `predict_fn` is
            called again. Defaults to 1.
        context (Optional[Sequence[float]]): Initial context used to warm up the
            external model, e.g. the training observations. Defaults to empty.
        max_context_len (Optional[int]): Cap on the context length; only the most
            recent values are kept. Defaults to None (context grows unbounded).
        std_error (Optional[float]): Process noise std in the SSM. Defaults to 0.0.
        mu_states (Optional[list[float]]): Initial mean of the hidden state.
        var_states (Optional[list[float]]): Initial variance of the hidden state.

    Examples:
        >>> import numpy as np
        >>> from canari.component import Auxiliary
        >>> def predict_fn(context, horizon):
        ...     return np.zeros(horizon), np.ones(horizon)
        >>> aux = Auxiliary(predict_fn=predict_fn, horizon=4, context=[1.0, 2.0, 3.0])
        >>> aux.states_name
        ['auxiliary']
    """

    def __init__(
        self,
        predict_fn: PredictFn,
        horizon: Optional[int] = 1,
        context: Optional[Sequence[float]] = None,
        max_context_len: Optional[int] = None,
        std_error: Optional[float] = 0.0,
        mu_states: Optional[list] = None,
        var_states: Optional[list] = None,
    ):
        if horizon < 1:
            raise ValueError("Auxiliary's horizon must be at least 1.")
        self.predict_fn = predict_fn
        self.horizon = horizon
        self.max_context_len = max_context_len
        self.std_error = std_error
        self._initial_context = np.asarray(
            [] if context is None else context, dtype=float
        ).flatten()
        self._mu_states = mu_states
        self._var_states = var_states
        self.reset()
        super().__init__()

    def reset(self):
        """
        Reset the context to the one provided at construction and discard the
        current prediction block, so that the next :meth:`predict` re-queries
        `predict_fn`.
        """

        self.context = self._initial_context.tolist()
        self._block_mu = None
        self._block_var = None
        self._offset = 0

    def set_context(self, context: Sequence[float]):
        """
        Replace the context, and make it the one :meth:`reset` restores.

        Args:
            context (Sequence[float]): Values the external forecaster starts from.
        """

        self._initial_context = np.asarray(context, dtype=float).flatten()
        self.reset()

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the one-step-ahead mean and variance from the external forecaster.

        Calls `predict_fn` only when the current block of :attr:`horizon`
        predictions is exhausted; otherwise serves the next value in that block.

        Returns:
            Tuple[np.ndarray, np.ndarray]: mean and variance, each of size 1.
        """

        if self._block_mu is None or self._offset == self.horizon:
            context = np.asarray(self.context, dtype=float)
            if self.max_context_len is not None:
                context = context[-self.max_context_len :]
            mu, var = self.predict_fn(context, self.horizon)
            self._block_mu = np.asarray(mu, dtype=float).flatten()
            self._block_var = np.asarray(var, dtype=float).flatten()
            if len(self._block_mu) != self.horizon or len(self._block_var) != self.horizon:
                raise ValueError(
                    f"predict_fn must return {self.horizon} means and variances, "
                    f"got {len(self._block_mu)} and {len(self._block_var)}."
                )
            self._offset = 0

        mu = self._block_mu[self._offset : self._offset + 1]
        var = self._block_var[self._offset : self._offset + 1]
        self._offset += 1
        return mu, var

    def update_context(self, value: float):
        """
        Append the newly assimilated auxiliary posterior mean to the context.

        Args:
            value (float): Posterior mean of the auxiliary hidden state.
        """

        self.context.append(float(np.ravel(value)[0]))

    def initialize_component_name(self):
        self._component_name = "auxiliary"

    def initialize_num_states(self):
        self._num_states = 1

    def initialize_states_name(self):
        self._states_name = ["auxiliary"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[0]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = np.array([[self.std_error**2]])

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self._num_states, 1))
        elif len(self._mu_states) == self._num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
        else:
            raise ValueError(
                "Incorrect mu_states dimension for the auxiliary component."
            )

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self._num_states, 1))
        elif len(self._var_states) == self._num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(
                "Incorrect var_states dimension for the auxiliary component."
            )
