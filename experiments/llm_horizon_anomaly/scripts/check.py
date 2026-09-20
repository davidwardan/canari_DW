"""Pipeline checks for the anomaly-detection sweep.

Run before `tune.py`. Verifies the three things that would silently corrupt the
whole sweep: that a realization leaves no trace in the next one, that the
realizations really are identical across horizons, and that the injected anomaly
has the shape and magnitude it is supposed to have. Then checks detection
behaviour on cases whose answer is known.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

PARAMS = {
    "std_transition_error": 1e-4,
    "norm_to_abnorm_prob": 1e-5,
    "local_trend_std": 1e-4,
    "local_accel_std": 1e-3,
}
SERIES = "ts43"  # a tuning series, never scored


def check_reset_is_clean(predict_fn):
    """Two identical runs separated by a reset must agree bit for bit."""

    context_data, filter_data = common.split_series(SERIES)
    skf, auxiliary = common.build_skf(predict_fn, 13, context_data["y"], PARAMS)

    first, _ = skf.filter(data=filter_data)
    assert len(auxiliary.context) == common.WARMUP + len(filter_data["y"])
    common.reset(skf, auxiliary)
    assert len(auxiliary.context) == common.WARMUP

    second, _ = skf.filter(data=filter_data)
    np.testing.assert_array_equal(first, second)
    print("reset is clean     : OK (identical probabilities across runs)")


def check_realizations_match_across_horizons():
    """The anomaly onsets must not depend on H, or the comparison is unpaired."""

    _, filter_data = common.split_series(SERIES)
    slope = 1.0 / common.WEEKS_PER_YEAR
    reference = [
        realization["anomaly_timestep"]
        for realization in common.make_realizations(
            filter_data, slope, common.NUM_REALIZATIONS
        )
    ]
    for _ in common.HORIZONS:  # nothing about H enters the draw; prove it stays put
        again = [
            realization["anomaly_timestep"]
            for realization in common.make_realizations(
                filter_data, slope, common.NUM_REALIZATIONS
            )
        ]
        assert again == reference
    low, high = common.ANOMALY_WINDOW
    assert all(
        int(np.ceil(len(filter_data["y"]) * low))
        <= onset
        < int(np.ceil(len(filter_data["y"]) * high))
        for onset in reference
    )
    assert max(reference) + common.MAX_WEEKS_TO_DETECT <= len(filter_data["y"])
    print(f"realizations       : OK (onsets {sorted(reference)}, identical for every H)")


def check_anomaly_shape():
    """The injection must be zero before the onset and linear after it."""

    _, filter_data = common.split_series(SERIES)
    slope = 0.75 / common.WEEKS_PER_YEAR
    realization = common.make_realizations(filter_data, slope, 1)[0]

    injected = realization["y"].flatten() - filter_data["y"].flatten()
    onset = realization["anomaly_timestep"]
    np.testing.assert_allclose(injected[:onset], 0.0, atol=1e-12)
    after = injected[onset:]
    np.testing.assert_allclose(np.diff(after), slope, atol=1e-9)
    print(f"anomaly shape      : OK (slope {slope * common.WEEKS_PER_YEAR:.2f}/year)")


def check_detection_metrics():
    """The scoring rules, on probability traces whose answer is obvious."""

    onset = 100
    late = np.zeros(400)
    late[onset + 10 :] = 0.9
    scored = common.score_realization(late, onset)
    assert scored["detected"] and scored["weeks_to_detect"] == 10
    assert not scored["pre_onset_alarm"]

    too_late = np.zeros(400)
    too_late[onset + common.MAX_WEEKS_TO_DETECT :] = 0.9
    assert not common.score_realization(too_late, onset)["detected"]

    # An alarm already running at the onset cannot be credited to the anomaly.
    carried_over = np.zeros(400)
    carried_over[onset - 20 :] = 0.9
    scored = common.score_realization(carried_over, onset)
    assert scored["pre_onset_alarm"] and scored["alarm_active_at_onset"]
    assert not scored["detected"], "a carried-over alarm must not count as a detection"

    # A pre-onset alarm that has cleared does not block a later real detection.
    cleared = np.zeros(400)
    cleared[50:60] = 0.9
    cleared[onset + 5 :] = 0.9
    scored = common.score_realization(cleared, onset)
    assert scored["pre_onset_alarm"] and not scored["alarm_active_at_onset"]
    assert scored["detected"] and scored["weeks_to_detect"] == 5

    sustained = np.zeros(520)
    sustained[100:200] = 0.9  # one alarm, not a hundred
    sustained[300:310] = 0.9
    clean = common.score_clean(sustained)
    assert clean["num_false_alarms"] == 2, clean
    np.testing.assert_allclose(clean["false_alarms_per_year"], 2 / 10)

    # Detection probability must be read against what the false alarms alone give.
    detections = pd.DataFrame(
        [{"horizon": 1, "detected": True, "weeks_to_detect": 10,
          "pre_onset_alarm": False, "alarm_active_at_onset": False}]
    )
    false_alarms = pd.DataFrame(
        [{"horizon": 1, "false_alarms_per_year": 0.2, "any_false_alarm": True}]
    )
    summary = common.summarize(detections, false_alarms)
    expected_chance = 1 - np.exp(-0.2 * common.MAX_WEEKS_TO_DETECT / common.WEEKS_PER_YEAR)
    np.testing.assert_allclose(summary["chance_pod"].iloc[0], expected_chance)
    np.testing.assert_allclose(summary["excess_pod"].iloc[0], 1.0 - expected_chance)
    print(f"detection metrics  : OK (chance level at 0.2 FA/yr = {expected_chance:.2f})")


def check_level_stays_put(predict_fn):
    """Regression check for the unidentifiable-baseline bug.

    `initialize_from_context` seeds the trend with the slope fitted to the warmup,
    and nothing ever pushes it back, because the auxiliary absorbs every residual
    that would. The level integrates that slope for the whole record: on `ts43`
    (warmup slope +0.0149/week) it ran away to +12 over 896 clean steps while the
    auxiliary went to -12, summing to the right observation so forecasts looked
    fine. `build_skf` resets the trend to zero to prevent it.

    The criterion is about *rate*: on clean data the baseline must not invent drift
    faster than a quarter of the anomaly magnitude we inject, or the model's own
    wander masquerades as the signal.

    H=1 is reported but not asserted. There the decomposition wanders badly whatever
    the parameters (~1.8 sigma/year), because Chronos-2 re-anchors every step; that
    is a property of the architecture, it is why H=1 has no operating point, and it
    is reported as a finding rather than treated as a bug to be hidden.
    """

    budget = 0.25 * common.PROVISIONAL_SLOPE_PER_YEAR  # sigma/year
    context_data, filter_data = common.split_series(SERIES)

    def drift_rate(horizon):
        skf, auxiliary = common.build_skf(predict_fn, horizon, context_data["y"], PARAMS)
        _, states = skf.filter(data=filter_data)
        level = states.get_mean(
            states_name="level", states_type="posterior", standardization=True
        )
        common.reset(skf, auxiliary)
        return abs(level[-1] - level[0]) / len(level) * common.WEEKS_PER_YEAR

    for horizon in (13, 52):
        rate = drift_rate(horizon)
        assert rate < budget, (
            f"level drifts {rate:.2f} sigma/year on clean data at H={horizon}, "
            f"budget {budget:.2f}; the baseline is running away again"
        )
        print(f"level stays put    : OK (H={horizon}, drift {rate:.3f} sigma/yr "
              f"< {budget:.2f})")
    print(f"level at H=1       : drift {drift_rate(1):.3f} sigma/yr "
          f"(reported, not asserted -- see docstring)")


def check_known_cases(predict_fn):
    """A large anomaly must be caught; the clean series should stay quiet."""

    context_data, filter_data = common.split_series(SERIES)
    skf, auxiliary = common.build_skf(predict_fn, 13, context_data["y"], PARAMS)

    prob, _ = skf.filter(data=filter_data)
    clean = common.score_clean(prob)
    common.reset(skf, auxiliary)

    big = common.make_realizations(filter_data, 4.0 / common.WEEKS_PER_YEAR, 1)[0]
    prob, _ = skf.filter(data=big)
    scored = common.score_realization(prob, big["anomaly_timestep"])
    print(
        f"known cases        : clean {clean['num_false_alarms']} false alarms "
        f"({clean['false_alarms_per_year']:.2f}/yr); 4 sigma/year anomaly "
        f"{'detected in ' + str(scored['weeks_to_detect']) + ' weeks' if scored['detected'] else 'MISSED'}"
    )
    assert scored["detected"], "a 4 sigma/year drift must be detected"


if __name__ == "__main__":
    check_realizations_match_across_horizons()
    check_anomaly_shape()
    check_detection_metrics()
    predict_fn = common.base.make_predict_fn(common.base.load_chronos())
    check_reset_is_clean(predict_fn)
    check_level_stays_put(predict_fn)
    check_known_cases(predict_fn)
    print("all pipeline checks passed")
