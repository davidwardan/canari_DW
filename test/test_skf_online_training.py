import copy

import numpy as np
import pytest

from canari import Model, SKF
from canari.component import LocalAcceleration, LocalTrend, LstmNetwork, WhiteNoise


def make_data(num_steps=20):
    return {
        "x": np.empty((num_steps, 0), dtype=np.float32),
        "y": np.sin(np.arange(num_steps, dtype=np.float32)).reshape(-1, 1),
        "time": np.arange(num_steps),
    }


def make_skf():
    lstm = LstmNetwork(look_back_len=3, num_features=1, num_hidden_unit=4, manual_seed=1)
    return SKF(
        norm_model=Model(LocalTrend(), lstm, WhiteNoise(std_error=0.1)),
        abnorm_model=Model(LocalAcceleration(), lstm, WhiteNoise(std_error=0.1)),
        std_transition_error=1e-4,
        norm_to_abnorm_prob=1e-4,
    )


def parameters_changed(before, after):
    return any(
        not np.array_equal(np.asarray(old), np.asarray(new))
        for layer in before
        for old, new in zip(before[layer], after[layer])
    )


@pytest.mark.parametrize("train_lstm", [False, True])
def test_skf_filter_updates_lstm_only_when_training(train_lstm):
    skf = make_skf()
    before = copy.deepcopy(skf.lstm_net.state_dict())

    skf.filter(make_data(), train_lstm=train_lstm)

    assert parameters_changed(before, skf.lstm_net.state_dict()) == train_lstm


@pytest.mark.parametrize("reset_memory", [False, True])
def test_skf_filter_reset_memory(reset_memory):
    skf = make_skf()
    model = skf.model["norm_norm"]

    skf.filter(make_data(), reset_memory=reset_memory)

    kept_final_states = np.array_equal(model.mu_states, model.states.mu_posterior[-1])
    assert kept_final_states != reset_memory
