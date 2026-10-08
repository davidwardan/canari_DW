"""Analytical controls for the acceleration experiment."""

import unittest

import numpy as np
import pandas as pd

import core


def constant_predictor(calls, variance=0.25):
    def predict(context, horizon):
        calls.append(np.asarray(context).copy())
        return np.zeros(horizon), np.full(horizon, variance)

    return predict


class CoreTests(unittest.TestCase):
    def test_quadratic_scaling_and_future_independence(self):
        values = np.sin(np.arange(150) / 4) + np.arange(150) / 100
        frame = pd.DataFrame({"values": values}, index=pd.date_range("2000-01-01", periods=150, freq="7D"))
        first, scale = core.prepare_series(frame)
        changed = frame.copy()
        changed.iloc[52:, 0] += 1000
        second, second_scale = core.prepare_series(changed)
        self.assertEqual(scale, second_scale)
        pd.testing.assert_frame_equal(first.iloc[:52], second.iloc[:52])
        np.testing.assert_allclose(first.y - first.residual, first.synthetic_acceleration_trend)
        np.testing.assert_allclose(first.synthetic_slope, core.ACCELERATION_PER_WEEK2 * np.arange(150))

    def test_ols_full_covariance_and_residual_context(self):
        time = np.arange(52) - 51
        y = 2 + 0.1 * time + 0.5 * 0.02 * time**2 + 0.2 * np.sin(np.arange(52))
        y[7] = np.nan
        model = core.build_model(y, constant_predictor([]), 13)
        valid = np.isfinite(y)
        design = np.column_stack((np.ones(52), time, 0.5 * time**2))
        beta = np.linalg.lstsq(design[valid], y[valid], rcond=None)[0]
        residual = y - design @ beta
        sigma2 = max(np.sum(residual[valid] ** 2) / (valid.sum() - 3), core.SIGMA_V**2)
        expected = sigma2 * np.linalg.inv(design[valid].T @ design[valid])
        np.testing.assert_allclose(model.mu_states[:3, 0], beta)
        np.testing.assert_allclose(model.var_states[:3, :3], expected)
        np.testing.assert_allclose(model.aux_component.context, residual, equal_nan=True)
        np.testing.assert_allclose(model.process_noise_matrix[:3, :3], 0)

    def test_known_quadratic_and_refresh_cadence(self):
        acceleration = 0.02
        t = np.arange(152, dtype=float)
        y = 2 + 0.1 * t + 0.5 * acceleration * t**2
        for horizon in core.HORIZONS:
            with self.subTest(horizon=horizon):
                calls = []
                model = core.build_model(y[:52], constant_predictor(calls), horizon)
                rows = [core.advance(model, value) for value in y[52:]]
                np.testing.assert_allclose([r["mu"] for r in rows], y[52:], atol=1e-11)
                np.testing.assert_allclose([r["slope_posterior"] for r in rows], 0.1 + acceleration * t[52:], atol=1e-11)
                np.testing.assert_allclose([r["acceleration_posterior"] for r in rows], acceleration, atol=1e-11)
                expected = 52 + horizon * np.arange(int(np.ceil(100 / horizon)))
                self.assertEqual([len(context) for context in calls], expected.tolist())
                for context in calls:
                    np.testing.assert_allclose(context, 0, atol=1e-11)

    def test_missing_observation_preserves_prior_covariance(self):
        model = core.build_model(np.sin(np.arange(52) / 4), constant_predictor([]), 13)
        row = core.advance(model, np.nan)
        self.assertEqual(row["level_prior"], row["level_posterior"])
        self.assertEqual(row["slope_prior"], row["slope_posterior"])
        self.assertEqual(row["acceleration_prior"], row["acceleration_posterior"])
        np.testing.assert_allclose(model.var_states, model.var_states_prior)

    def test_manual_updates_match_library_filter(self):
        y = np.sin(np.arange(100) / 4) + 0.01 * np.arange(100) + 0.0001 * np.arange(100)**2
        manual = core.build_model(y[:52], constant_predictor([]), 13)
        filtered = core.build_model(y[:52], constant_predictor([]), 13)
        rows = [core.advance(manual, value) for value in y[52:]]
        mu, std, _ = filtered.filter({"y": y[52:].reshape(-1, 1), "x": np.empty((48, 0)), "time": np.arange(52, 100)})
        np.testing.assert_allclose([r["mu"] for r in rows], mu)
        np.testing.assert_allclose([r["var"] for r in rows], std**2)
        np.testing.assert_allclose(manual.mu_states, filtered.mu_states)
        np.testing.assert_allclose(manual.var_states, filtered.var_states)


if __name__ == "__main__":
    unittest.main()
