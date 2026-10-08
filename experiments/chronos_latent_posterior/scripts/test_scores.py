"""Probability-score and variance-identity controls for saved forecasts."""

import numpy as np
import pytest

from run_experiment import mixture_scores, record


def test_gaussian_mixture_scores_reduce_to_single_gaussian():
    from decomposition import gaussian_metrics
    for y in (-1., 0., 1.):
        nll, crps = mixture_scores(y, np.full(4, .2), .09)
        expected = gaussian_metrics(y, .2, .09)
        assert nll == pytest.approx(expected["nll"])
        assert crps == pytest.approx(expected["crps"])


def test_record_preserves_nonnegative_identity_and_zero_denominator():
    identity = {"case": "toy", "seed": 0, "origin": 10, "lead": 1}
    result = record(identity, "joint", np.array([-1., 1.]), .25, .1, .2)
    assert result["total"] == pytest.approx(1.25)
    assert result["reconstruction_error"] == 0
    zero = record(identity, "plugin", [0.], .25, .1, .2)
    assert zero["epistemic"] == 0 and not zero["state_scored"]
    assert np.isnan(zero["predictable_coverage90"])
