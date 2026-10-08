"""Controls for the zero-Chronos-variance variant."""

import unittest

import numpy as np
import torch

import run_zero_variance
from core import advance, build_model


class FakePipeline:
    """Deterministic stand-in for a Chronos pipeline with a non-zero spread."""

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        means = [torch.full((prediction_length,), float(x[-1]) + 0.5) for x in inputs]
        bounds = [torch.stack((m - 1, m + 1), dim=-1) for m in means]
        return bounds, means


class ZeroVarianceTests(unittest.TestCase):
    def test_wrapper_zeroes_variance_only(self):
        contexts = [np.arange(10.0), np.ones(20)]
        original = run_zero_variance.chronos_predict_batch(FakePipeline(), contexts, 4)
        zeroed = run_zero_variance.predict_batch_zero_variance(FakePipeline(), contexts, 4)
        for (mu, variance, q), (mu0, variance0, q0) in zip(original, zeroed):
            np.testing.assert_array_equal(mu0, mu)
            np.testing.assert_array_equal(q0, q)
            self.assertTrue(np.all(variance > 0))
            np.testing.assert_array_equal(variance0, 0)

    def test_reference_runner_uses_zero_variance(self):
        self.assertIs(run_zero_variance.reference.predict_batch, run_zero_variance.predict_batch_zero_variance)

    def test_zero_variance_filter(self):
        rng = np.random.default_rng(0)
        y = np.sin(np.arange(120) / 4) + 0.01 * np.arange(120) + 0.1 * rng.standard_normal(120)
        served = []

        def predict(context, horizon):
            mu = 0.3 * np.cos(len(context) + np.arange(horizon))
            served.extend(mu)
            return mu, np.zeros(horizon)

        model = build_model(y[:52], predict, 13)
        rows = [advance(model, value) for value in y[52:]]
        self.assertEqual(len(served), 13 * 6)
        np.testing.assert_array_equal([r["aux_variance"] for r in rows], 0)
        np.testing.assert_allclose([r["var"] for r in rows],
                                   [r["trend_variance"] + r["white_variance"] for r in rows], rtol=1e-12)
        # Zero gain: the posterior, and thus the appended context, is the served mean.
        np.testing.assert_allclose(model.aux_component.context[52:], served[:len(rows)], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
