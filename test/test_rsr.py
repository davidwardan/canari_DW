import numpy as np
import pytest

from canari import Model
from canari.component import LocalTrend, LstmNetwork, WhiteNoise
from canari.rsr import STRATEGIES


def make_data(num_steps=20):
    return {
        "x": np.empty((num_steps, 0), dtype=np.float32),
        "y": np.sin(np.arange(num_steps, dtype=np.float32)).reshape(-1, 1),
    }


def parameters_changed(before, after):
    return any(
        not np.array_equal(np.asarray(old), np.asarray(new))
        for layer in before
        for old, new in zip(before[layer], after[layer])
    )


def make_lstm_model():
    return Model(
        LocalTrend(),
        LstmNetwork(
            look_back_len=1,
            num_features=1,
            num_hidden_unit=4,
            manual_seed=1,
        ),
        WhiteNoise(std_error=0.1),
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_rsr_filter_updates_parameters(strategy):
    model = make_lstm_model()
    parameters_before = model.lstm_net.state_dict()

    mean, std, states = model.rsr_filter(
        make_data(), start=10, end=13, window_len=3, strategy=strategy
    )

    assert len(mean) == len(std) == len(states.mu_prior) == 3
    assert np.all(np.isfinite(mean))
    assert np.all(std > 0)
    assert parameters_changed(parameters_before, model.lstm_net.state_dict())


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_rsr_filter_handles_missing_observation(strategy):
    # The missing step is replayed by every later window of the run.
    data = make_data()
    data["y"][9] = np.nan
    model = make_lstm_model()

    mean, std, _ = model.rsr_filter(
        data, start=10, end=13, window_len=3, strategy=strategy
    )

    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))
    assert all(
        np.all(np.isfinite(np.asarray(values)))
        for layer in model.lstm_net.state_dict().values()
        for values in layer
    )


def test_rsr_filter_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Incorrect strategy"):
        make_lstm_model().rsr_filter(
            make_data(), start=10, end=13, window_len=3, strategy="smoothed"
        )
