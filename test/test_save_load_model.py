import numpy as np
import numpy.testing as npt
import pytest
from canari import Model
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

lstm_look_back_len = 10


def compare_model_dict(model_1_dict, model_2_dict):
    """Test save/load for model.py"""

    for component_1, component_2 in zip(
        model_1_dict["components"].values(), model_2_dict["components"].values()
    ):
        npt.assert_allclose(component_1.mu_states, component_2.mu_states)
        npt.assert_allclose(component_1.var_states, component_2.var_states)
        npt.assert_allclose(
            component_1.transition_matrix, component_2.transition_matrix
        )
        npt.assert_allclose(
            component_1.process_noise_matrix, component_2.process_noise_matrix
        )
        npt.assert_allclose(
            component_1.observation_matrix, component_2.observation_matrix
        )
        assert component_1.states_name == component_2.states_name

    npt.assert_allclose(
        model_1_dict["memory"]["mu_states"], model_2_dict["memory"]["mu_states"]
    )
    npt.assert_allclose(
        model_1_dict["memory"]["var_states"], model_2_dict["memory"]["var_states"]
    )

    if "lstm_network_params" in model_1_dict and "lstm_network_params" in model_2_dict:
        assert (
            model_1_dict["lstm_network_params"] == model_2_dict["lstm_network_params"]
        )


def compare_lstm_dict(model_1_dict, model_2_dict):
    """Test lstm initialization with different seeds"""
    assert (
        not model_1_dict["lstm_network_params"] == model_2_dict["lstm_network_params"]
    )


@pytest.mark.parametrize("smoother", [False, True], ids=["LSTM", "SLSTM"])
def test_model_save_load(smoother):
    """Test save/load for model.py"""

    lstm_network_1 = LstmNetwork(
        look_back_len=lstm_look_back_len,
        num_features=2,
        num_layer=1,
        infer_len=24,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
        smoother=smoother,
    )
    lstm_network_2 = LstmNetwork(
        look_back_len=lstm_look_back_len,
        num_features=2,
        num_layer=1,
        infer_len=24,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=2,
        smoother=smoother,
    )

    model1 = Model(LocalTrend(), lstm_network_1, WhiteNoise())
    model2 = Model(LocalTrend(), lstm_network_2, WhiteNoise())

    model1_dict = model1.get_dict()
    model1_loaded = Model.load_dict(model1_dict)
    model1_loaded_dict = model1_loaded.get_dict()
    compare_model_dict(model1_dict, model1_loaded_dict)

    model2_dict = model2.get_dict()
    compare_lstm_dict(model1_dict, model2_dict)


@pytest.mark.parametrize("smoother", [False, True], ids=["LSTM", "SLSTM"])
def test_loaded_model_resumes_filtered_memory(smoother):
    """A model saved mid-series forecasts exactly like the original."""

    num_steps = 12
    data = {
        "x": np.empty((num_steps, 0), dtype=np.float32),
        "y": np.sin(np.arange(num_steps, dtype=np.float32)).reshape(-1, 1),
        "time": np.arange(num_steps),
    }
    history = {key: values[:8] for key, values in data.items()}
    future = {key: values[8:] for key, values in data.items()}
    model = Model(
        LocalTrend(),
        LstmNetwork(
            look_back_len=3,
            num_features=1,
            num_hidden_unit=4,
            manual_seed=1,
            smoother=smoother,
        ),
        WhiteNoise(std_error=0.1),
    )
    model.lstm_net.num_samples = 20
    model.lstm_net.eval()
    model.filter(history, train_lstm=False)
    if smoother:
        model.lstm_net.set_lstm_states(model.lstm_net.get_lstm_states(7))

    loaded = Model.load_dict(model.get_dict())
    loaded.initialize_states_history()
    loaded.lstm_net.eval()

    npt.assert_array_equal(loaded.forecast(future)[0], model.forecast(future)[0])
