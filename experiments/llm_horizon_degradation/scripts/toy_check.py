"""Pipeline check on series whose behaviour is known in advance.

Run this before `run_experiment.py`. It exercises the same `run_filter` path the
sweep uses, on (a) a constant predictor where the Kalman recursion has a closed
form, (b) an AR(1) process whose 1-step and long-horizon errors are analytic, and
(c) a noiseless annual sinusoid, which a competent foundation model should
extrapolate almost as well at H=208 as at H=1.

Scores are reported as a ratio to climatology, i.e. to predicting the warmup mean
on the evaluation segment. Standardization uses the warmup only, so the
climatological RMSE is near but not exactly 1 and has to be measured, not assumed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

NUM_STEPS = 300


def as_frame(values):
    index = pd.date_range("2000-01-03", periods=len(values), freq="7D")
    frame = pd.DataFrame({"values": values}, index=index)
    frame.index.name = "date_time"
    return frame


def constant_predictor(mu, var, calls):
    def predict_fn(context, horizon):
        calls.append(len(context))
        return np.full(horizon, mu), np.full(horizon, var)

    return predict_fn


def skill(predictions):
    """RMSE, and RMSE as a fraction of the climatological RMSE on the same steps."""

    scores = common.score(
        predictions["mu"].to_numpy(),
        predictions["std"].to_numpy(),
        predictions["y"].to_numpy(),
    )
    observed = predictions["y"].to_numpy()
    climatology = float(np.sqrt(np.nanmean(observed**2)))
    return scores["rmse"], scores["rmse"] / climatology


def check_block_structure():
    """`predict_fn` is called once per block, on a context that grows every step."""

    frame = as_frame(np.sin(np.arange(NUM_STEPS) * 2 * np.pi / 52))
    num_filtered = NUM_STEPS - common.WARMUP

    for horizon in common.HORIZONS:
        calls = []
        common.run_filter(frame, horizon, constant_predictor(0.0, 1.0, calls))
        expected_calls = int(np.ceil(num_filtered / horizon))
        assert len(calls) == expected_calls, (horizon, len(calls), expected_calls)
        expected_lengths = common.WARMUP + horizon * np.arange(expected_calls)
        assert calls == expected_lengths.tolist(), (horizon, calls[:5])
    print(f"block structure   : OK for H in {common.HORIZONS}")


def check_kalman_closed_form():
    """With a constant auxiliary forecast, the filter output is analytic."""

    mu_aux, var_aux = 0.4, 0.25
    frame = as_frame(np.random.default_rng(common.SEED).normal(size=NUM_STEPS))
    predictions = common.run_filter(frame, 13, constant_predictor(mu_aux, var_aux, []))

    np.testing.assert_allclose(predictions["mu"], mu_aux, atol=1e-10)
    np.testing.assert_allclose(
        predictions["std"], np.sqrt(var_aux + common.SIGMA_V**2), atol=1e-10
    )
    print("kalman closed form: OK")


def check_ar1(predict_fn):
    """Chronos-2 must beat climatology at H=1 and fall back to it at H=208."""

    phi = 0.7
    rng = np.random.default_rng(common.SEED)
    noise = rng.normal(scale=np.sqrt(1 - phi**2), size=NUM_STEPS)
    values = np.zeros(NUM_STEPS)
    for t in range(1, NUM_STEPS):
        values[t] = phi * values[t - 1] + noise[t]

    short = skill(common.run_filter(as_frame(values), 1, predict_fn))
    long = skill(common.run_filter(as_frame(values), 208, predict_fn))
    print(
        f"AR(1) phi={phi}     : RMSE/clim H=1 {short[1]:.3f} "
        f"(analytic {np.sqrt(1 - phi**2):.3f}), H=208 {long[1]:.3f} (expected ~1)"
    )
    assert short[0] < long[0], "H=1 must beat H=208 on a decaying process"
    assert short[1] < 0.85, "H=1 must clearly beat climatology"
    assert 0.8 < long[1] < 1.3, "H=208 must fall back to roughly climatology"


def check_sinusoid(predict_fn):
    """A deterministic annual cycle must survive a 208-week block."""

    rng = np.random.default_rng(common.SEED)
    values = np.sin(np.arange(NUM_STEPS) * 2 * np.pi / 52) + rng.normal(
        scale=0.05, size=NUM_STEPS
    )

    short = skill(common.run_filter(as_frame(values), 1, predict_fn))
    long = skill(common.run_filter(as_frame(values), 208, predict_fn))
    print(
        f"annual sinusoid   : RMSE/clim H=1 {short[1]:.3f}, H=208 {long[1]:.3f} "
        "(expected well below 1 at both)"
    )
    assert long[1] < 0.5, "a pure seasonal signal must stay predictable at H=208"


if __name__ == "__main__":
    check_block_structure()
    check_kalman_closed_form()
    chronos_predict = common.make_predict_fn(common.load_chronos())
    check_ar1(chronos_predict)
    check_sinusoid(chronos_predict)
    print("all pipeline checks passed")
