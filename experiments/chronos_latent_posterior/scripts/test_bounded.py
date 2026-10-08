"""Control the declared amplitude prior independently of Chronos inference."""

from types import SimpleNamespace

import numpy as np

from run_bounded import BoundedTransition
from run_experiment import Transition


def test_bound_handles_extreme_transition_outputs(monkeypatch):
    values = np.array([[-1e15, -.0001, 0., .0001, 1e15]])
    monkeypatch.setattr(Transition, "__call__", lambda *args, **kwargs: values)
    transition = BoundedTransition(SimpleNamespace(quantiles=[.5]), bound=3.)
    bounded = transition(np.zeros((1, 32)))
    assert np.isfinite(bounded).all() and np.max(abs(bounded)) <= 3
    np.testing.assert_allclose(bounded[0, 1:4], values[0, 1:4], atol=1e-12)
    assert bounded[0, 0] == -3 and bounded[0, -1] == 3
