import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from canari import DataProcess, Model, SKF
from canari.component import Auxiliary, LocalAcceleration, LocalTrend, WhiteNoise

HORIZON = 5


def make_data(num_time_steps=120):
    return make_data_processor(num_time_steps).get_splits()


def make_data_processor(num_time_steps=120):
    time = pd.date_range("2020-01-01", periods=num_time_steps, freq="h")
    values = np.sin(np.arange(num_time_steps) * 2 * np.pi / 24)
    data = pd.DataFrame({"values": values}, index=time)
    data.index.name = "date_time"
    return DataProcess(
        data=data, train_split=0.5, validation_split=0.2, output_col=[0]
    )


class RecordingPredictor:
    """Constant predictor that logs the context it is given at each call."""

    def __init__(self):
        self.context_lengths = []
        self.contexts = []

    def __call__(self, context, horizon):
        self.context_lengths.append(len(context))
        self.contexts.append(np.asarray(context).copy())
        return np.zeros(horizon), np.full(horizon, 0.1)


def test_predict_fn_called_once_per_horizon():
    predictor = RecordingPredictor()
    auxiliary = Auxiliary(predict_fn=predictor, horizon=HORIZON, context=[0.0] * 24)
    train_data, _, _, all_data = make_data()
    model = Model(LocalTrend(), auxiliary, WhiteNoise(std_error=0.1))
    model.auto_initialize_baseline_states(train_data["y"][0:24])

    num_time_steps = len(all_data["y"])
    model.filter(data=all_data)

    assert len(predictor.context_lengths) == int(np.ceil(num_time_steps / HORIZON))


def test_context_grows_with_auxiliary_posterior():
    predictor = RecordingPredictor()
    auxiliary = Auxiliary(predict_fn=predictor, horizon=HORIZON, context=[0.0] * 24)
    train_data, _, _, all_data = make_data()
    model = Model(LocalTrend(), auxiliary, WhiteNoise(std_error=0.1))
    model.auto_initialize_baseline_states(train_data["y"][0:24])

    _, _, states = model.filter(data=all_data)

    # One point is appended per filtered time step, so the context grows by
    # exactly `horizon` between two consecutive calls.
    growth = np.diff(predictor.context_lengths)
    npt.assert_array_equal(growth, np.full(len(growth), HORIZON))
    assert len(auxiliary.context) == 24 + len(all_data["y"])

    # The appended values are the auxiliary posteriors.
    auxiliary_posterior = states.get_mean(
        states_name="auxiliary", states_type="posterior"
    )
    npt.assert_allclose(auxiliary.context[24:], auxiliary_posterior, rtol=1e-10)


def test_reset_restores_initial_context():
    predictor = RecordingPredictor()
    auxiliary = Auxiliary(predict_fn=predictor, horizon=HORIZON, context=[0.0] * 24)
    train_data, _, _, all_data = make_data()
    model = Model(LocalTrend(), auxiliary, WhiteNoise(std_error=0.1))
    model.auto_initialize_baseline_states(train_data["y"][0:24])

    model.filter(data=all_data)
    auxiliary.reset()
    assert len(auxiliary.context) == 24


def test_max_context_len_caps_the_context():
    predictor = RecordingPredictor()
    auxiliary = Auxiliary(
        predict_fn=predictor, horizon=HORIZON, context=[0.0] * 24, max_context_len=30
    )
    train_data, _, _, all_data = make_data()
    model = Model(LocalTrend(), auxiliary, WhiteNoise(std_error=0.1))
    model.auto_initialize_baseline_states(train_data["y"][0:24])

    model.filter(data=all_data)

    assert max(predictor.context_lengths) == 30


def test_skf_shares_one_auxiliary_across_transition_models():
    predictor = RecordingPredictor()
    auxiliary = Auxiliary(predict_fn=predictor, horizon=HORIZON, context=[0.0] * 24)
    train_data, _, _, all_data = make_data()
    noise = WhiteNoise(std_error=0.1)
    skf = SKF(
        norm_model=Model(LocalTrend(), auxiliary, noise),
        abnorm_model=Model(LocalAcceleration(), auxiliary, noise),
        std_transition_error=1e-4,
        norm_to_abnorm_prob=1e-4,
    )
    skf.auto_initialize_baseline_states(train_data["y"][0:24])

    # All four transition models must drive the same context, so that the
    # external predictor is queried once per horizon rather than once per model.
    assert all(model.aux_component is auxiliary for model in skf.model.values())

    num_time_steps = len(all_data["y"])
    skf.filter(data=all_data)

    assert len(predictor.context_lengths) == int(np.ceil(num_time_steps / HORIZON))
    assert len(auxiliary.context) == 24 + num_time_steps


def test_auxiliary_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        Auxiliary(predict_fn=lambda context, horizon: (0, 0), horizon=0)

    auxiliary = Auxiliary(
        predict_fn=lambda context, horizon: (np.zeros(2), np.zeros(2)), horizon=3
    )
    with pytest.raises(ValueError):
        auxiliary.predict()


def test_split_at_by_index_and_by_time():
    _, _, _, all_data = make_data()

    context_data, filter_data = DataProcess.split_at(all_data, 40)
    assert len(context_data["y"]) == 40
    assert len(filter_data["y"]) == len(all_data["y"]) - 40
    assert filter_data["start_date"] == all_data["time"][40]
    npt.assert_array_equal(
        np.concatenate([context_data["y"], filter_data["y"]]), all_data["y"]
    )

    by_time_context, by_time_filter = DataProcess.split_at(
        all_data, all_data["time"][40]
    )
    npt.assert_array_equal(by_time_context["y"], context_data["y"])
    npt.assert_array_equal(by_time_filter["y"], filter_data["y"])


@pytest.mark.parametrize("split_point", [0, 120, 500])
def test_split_at_rejects_out_of_range(split_point):
    _, _, _, all_data = make_data()
    with pytest.raises(ValueError):
        DataProcess.split_at(all_data, split_point)


def test_initialize_from_context_seeds_context_and_baseline():
    _, _, _, all_data = make_data()
    context_data, _ = DataProcess.split_at(all_data, 60)

    predictor = RecordingPredictor()
    auxiliary = Auxiliary(predict_fn=predictor, horizon=HORIZON)
    model = Model(LocalTrend(), auxiliary, WhiteNoise(std_error=0.1))
    model.initialize_from_context(context_data["y"])

    # The context holds the segment with its linear trend removed, which is the
    # quantity the component appends once filtering starts.
    trend, slope, _, _ = DataProcess.decompose_data(context_data["y"].flatten())
    npt.assert_allclose(auxiliary.context, context_data["y"].flatten() - trend)

    # The baseline is placed at the *end* of the context, i.e. where filtering resumes.
    npt.assert_allclose(model.mu_states[model.get_states_index("level")], trend[-1])
    npt.assert_allclose(model.mu_states[model.get_states_index("trend")], slope)


def test_filtering_starts_after_the_context_segment():
    _, _, _, all_data = make_data()
    context_data, filter_data = DataProcess.split_at(all_data, 60)

    predictor = RecordingPredictor()
    auxiliary = Auxiliary(predict_fn=predictor, horizon=HORIZON)
    model = Model(LocalTrend(), auxiliary, WhiteNoise(std_error=0.1))
    model.initialize_from_context(context_data["y"])
    mu_preds, _, states = model.filter(data=filter_data)

    num_filtered = len(filter_data["y"])
    assert len(mu_preds) == num_filtered
    assert len(states.mu_posterior) == num_filtered

    # The context segment is never filtered: it costs no call and is only context.
    assert len(predictor.context_lengths) == int(np.ceil(num_filtered / HORIZON))
    assert predictor.context_lengths[0] == 60
    assert len(auxiliary.context) == 60 + num_filtered


def test_skf_filtering_starts_after_the_context_segment():
    _, _, _, all_data = make_data()
    context_data, filter_data = DataProcess.split_at(all_data, 60)

    predictor = RecordingPredictor()
    auxiliary = Auxiliary(predict_fn=predictor, horizon=HORIZON)
    noise = WhiteNoise(std_error=0.1)
    skf = SKF(
        norm_model=Model(LocalTrend(), auxiliary, noise),
        abnorm_model=Model(LocalAcceleration(), auxiliary, noise),
        std_transition_error=1e-4,
        norm_to_abnorm_prob=1e-4,
    )
    skf.initialize_from_context(context_data["y"])
    filter_prob, states = skf.filter(data=filter_data)

    num_filtered = len(filter_data["y"])
    assert len(filter_prob) == num_filtered
    assert len(states.mu_posterior) == num_filtered
    assert predictor.context_lengths[0] == 60
    assert len(auxiliary.context) == 60 + num_filtered
