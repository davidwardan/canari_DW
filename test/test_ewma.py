"""EWMA coupling, Gaussian inference, and workflow tests for EwmaConstraint."""

import copy
import importlib.util
import inspect
import pickle

import numpy as np
import pandas as pd
import pytest

import canari.component
from canari import DataProcess, Model, SKF
from canari.component import (
    Autoregression,
    EwmaConstraint,
    LocalAcceleration,
    LocalLevel,
    LocalTrend,
    LstmNetwork,
    WhiteNoise,
)

# Combinations with other branches' features run only where those features exist.
HAS_RSR = importlib.util.find_spec("canari.rsr") is not None
HAS_AUXILIARY = hasattr(canari.component, "Auxiliary")
HAS_SKF_ONLINE_TRAINING = "train_lstm" in inspect.signature(SKF.filter).parameters


def data(n=16):
    return {
        "x": np.empty((n, 0), dtype=np.float32),
        "y": (0.2 + np.sin(np.arange(n) * 0.5)).astype(np.float32)[:, None],
        "time": np.arange(n),
    }


def make_model(constrained=True, smoother=True, **kwargs):
    components = [
        LocalTrend(std_error=0.03, mu_states=[0.1, 0.005], var_states=[0.2, 0.01]),
        LstmNetwork(num_hidden_unit=4, manual_seed=11, smoother=smoother),
        WhiteNoise(std_error=0.1),
    ]
    if constrained:
        components.append(EwmaConstraint(fast_span=4, slow_span=16, **kwargs))
    return Model(*components)


def memory_indices(model, constraint):
    return [model.get_states_index(name) for name in constraint.states_name]


def fix_lstm_output(net, mu, var):
    """Replace the LSTM forward by fixed output moments."""
    net.forward = lambda **_: (np.array([mu]), np.array([var]))


def record_lstm_messages(net):
    """Record the learning messages sent to the LSTM instead of applying them."""
    messages = []
    net.set_delta_z = lambda dm, dv: messages.append((dm.item(), dv.item()))
    net.backward = lambda: None
    net.step = lambda: None
    return messages


def condition_reference(mu, var, matrix, observed, noise):
    """Independent Schur-complement reference (the implementation uses Joseph form)."""
    cross = var @ matrix.T
    gain = np.linalg.solve(matrix @ cross + noise, cross.T).T
    return mu + gain @ (observed - matrix @ mu), var - gain @ cross.T


def reference_transition(model):
    """Build the joint transition directly from the EWMA recurrence."""
    transition = model.transition_matrix.copy()
    source = model.get_states_index("lstm")
    injection = np.zeros((model.num_states, 1))
    injection[source] = 1
    for constraint in model.ewma_constraints:
        for index, rho in zip(memory_indices(model, constraint), constraint.rhos):
            transition[index, :] = 0
            transition[index, index] = rho
            injection[index] = 1 - rho
    noise = model.process_noise_matrix.copy()
    noise[source, source] = 0
    return transition, injection, noise


def test_joint_prediction_matches_independent_linear_transform():
    model = make_model(tau_mean=None, tau_slope=None)
    rng = np.random.default_rng(903)
    n = model.num_states
    mean = rng.normal(size=(n, 1))
    a = rng.normal(size=(n, n))
    covariance = a @ a.T + np.eye(n)
    model.set_states(mean, covariance)
    transition, injection, noise = reference_transition(model)
    raw_mean, raw_var = 1.3, 0.8
    expected_mean = transition @ mean + injection * raw_mean
    expected_cov = (
        transition @ covariance @ transition.T
        + noise
        + raw_var * injection @ injection.T
    )

    _, _, actual_mean, actual_cov = model.forward(
        mu_lstm_pred=np.array([raw_mean]), var_lstm_pred=np.array([raw_var])
    )
    np.testing.assert_allclose(actual_mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(actual_cov, expected_cov, atol=1e-12)

    model.initialize_states_history()
    model._set_posterior_states(actual_mean, actual_cov)
    model.save_states_history()
    np.testing.assert_allclose(
        model.states.cov_states[-1], covariance @ transition.T, atol=1e-12
    )


TWO_OBSERVATIONS = (0.8, -0.1)


def assert_two_step_smoother_matches(model, transition, mu1, cov1, mu2, cov2):
    """Filter and smooth two steps, then compare with their full joint Gaussian."""
    n = model.num_states
    constraint = model.ewma_constraints[0]
    c, r = constraint.constraint_observation(n, memory_indices(model, constraint))
    one_h = np.vstack([c, model.observation_matrix])
    k = len(one_h)
    h = np.zeros((2 * k, 2 * n))
    h[:k, :n] = h[k:, n:] = one_h
    observation_noise = np.zeros((2 * k, 2 * k))
    observation_noise[: k - 1, : k - 1] = np.diag(r)
    observation_noise[k : 2 * k - 1, k : 2 * k - 1] = np.diag(r)
    observed = np.zeros((2 * k, 1))
    observed[[k - 1, 2 * k - 1], 0] = TWO_OBSERVATIONS
    cross = cov1 @ transition.T
    expected_mu, expected_cov = condition_reference(
        np.vstack([mu1, mu2]),
        np.block([[cov1, cross], [cross.T, cov2]]),
        h,
        observed,
        observation_noise,
    )
    model.filter(
        {
            "x": np.empty((2, 0)),
            "y": np.array(TWO_OBSERVATIONS)[:, None],
            "time": np.arange(2),
        },
        train_lstm=False,
    )
    model.rts_smoother(0)
    for step in range(2):
        sl = slice(step * n, (step + 1) * n)
        np.testing.assert_allclose(
            model.states.mu_smooth[step], expected_mu[sl], atol=1e-10
        )
        np.testing.assert_allclose(
            model.states.var_smooth[step], expected_cov[sl, sl], atol=1e-10
        )
        assert np.linalg.eigvalsh(model.states.var_smooth[step]).min() > -1e-12


def test_two_step_smoother_matches_full_joint_gaussian():
    model = make_model(tau_mean=0.04, tau_slope=0.01)
    transition, injection, noise = reference_transition(model)
    raw = [(0.4, 0.8), (-0.2, 0.6)]
    mu1 = transition @ model.mu_states + injection * raw[0][0]
    cov1 = (
        transition @ model.var_states @ transition.T
        + noise
        + raw[0][1] * injection @ injection.T
    )
    mu2 = transition @ mu1 + injection * raw[1][0]
    cov2 = (
        transition @ cov1 @ transition.T + noise + raw[1][1] * injection @ injection.T
    )
    outputs = iter(raw)
    model.lstm_net.forward = lambda **_: tuple(np.array([v]) for v in next(outputs))
    assert_two_step_smoother_matches(model, transition, mu1, cov1, mu2, cov2)


def test_two_step_smoother_with_linear_source_matches_full_joint_gaussian():
    # The source has its own dynamics: l_t = l_{t-1} + eta_t, so the EWMAs
    # m_t = rho m_{t-1} + (1 - rho) l_t depend on l_{t-1} and on eta_t.
    std_level = 0.3
    model = Model(
        LocalLevel(std_error=std_level, mu_states=[0.5], var_states=[0.4]),
        WhiteNoise(std_error=0.1),
        EwmaConstraint(
            source="level", fast_span=4, slow_span=16, tau_mean=0.04, tau_slope=0.01
        ),
    )
    level = model.get_states_index("level")
    transition = model.transition_matrix.copy()
    level_noise = np.zeros((model.num_states, 1))
    level_noise[level] = 1
    for index, rho in zip(
        memory_indices(model, model.ewma_constraints[0]),
        model.ewma_constraints[0].rhos,
    ):
        transition[index, level] = 1 - rho
        level_noise[index] = 1 - rho
    noise = model.process_noise_matrix.copy()
    noise[level, level] = 0
    noise += std_level**2 * level_noise @ level_noise.T
    mu1 = transition @ model.mu_states
    cov1 = transition @ model.var_states @ transition.T + noise
    mu2 = transition @ mu1
    cov2 = transition @ cov1 @ transition.T + noise
    assert_two_step_smoother_matches(model, transition, mu1, cov1, mu2, cov2)


@pytest.mark.parametrize("observed", [0.9, np.nan])
def test_learning_messages_include_constraint_and_real_observation(observed):
    model = make_model(tau_mean=0.025, tau_slope=0.005)
    raw_mu, raw_var = 0.7, 0.6
    fix_lstm_output(model.lstm_net, raw_mu, raw_var)
    messages = record_lstm_messages(model.lstm_net)
    model.filter({"x": np.empty((1, 0)), "y": np.array([[observed]]), "time": [0]})
    dm, dv = messages[0]
    index = model.get_states_index("lstm")
    assert raw_mu + raw_var * dm == pytest.approx(model.mu_states[index, 0], abs=2e-7)
    assert raw_var + raw_var**2 * dv == pytest.approx(
        model.var_states[index, index], abs=2e-7
    )
    assert abs(dm) > 1e-5 and dv < 0


def test_disabled_constraints_preserve_predictions_and_learning():
    original = make_model(constrained=False)
    augmented = make_model(tau_mean=None, tau_slope=None)
    results = []
    for model in (original, augmented):
        model.lstm_net.num_samples = 20
        model.lstm_net.train()
        results.append(model.filter(data())[:2])
    np.testing.assert_allclose(results[0], results[1], rtol=2e-5, atol=2e-6)
    for layer in original.lstm_net.state_dict():
        for expected, actual in zip(
            original.lstm_net.state_dict()[layer],
            augmented.lstm_net.state_dict()[layer],
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_frozen_weights_keep_constraints_active():
    model = make_model(smoother=False)
    before = copy.deepcopy(model.lstm_net.state_dict())
    mean, std, _ = model.filter(data(), train_lstm=False)
    assert before == model.lstm_net.state_dict()
    assert np.all(np.isfinite(mean)) and np.all(std > 0)
    assert np.any(model.mu_states[model._ewma_indices] != 0)
    assert not np.allclose(model._mu_states_transition, model.mu_states_prior)


def test_forecast_applies_constraints():
    model = make_model(smoother=False)
    model.filter(data(), train_lstm=False)
    _, _, states = model.forecast(data(4))
    prior = np.hstack(states.mu_prior[-4:])
    posterior = np.hstack(states.mu_posterior[-4:])
    assert not np.allclose(prior, posterior)


def test_deterministic_source_and_memories_are_not_conditioned():
    model = make_model()
    model.forward(mu_lstm_pred=np.array([2.0]), var_lstm_pred=np.array([0.0]))
    model.backward(0.7)
    index = model.get_states_index("lstm")
    assert model.mu_states_posterior[index, 0] == 2
    assert model.var_states_posterior[index, index] == 0


def test_linear_source_and_multiple_constraints_are_order_independent():
    level = LocalLevel(std_error=0.1, mu_states=[1], var_states=[0.2])
    ar = Autoregression(phi=0.8, std_error=0.2)
    constraints = [
        EwmaConstraint(source="level"),
        EwmaConstraint(source="autoregression"),
    ]
    first = Model(level, ar, WhiteNoise(), *constraints)
    second = Model(constraints[1], ar, constraints[0], WhiteNoise(), level)
    for model in (first, second):
        model.filter(data(), train_lstm=False)
        model.smoother()
    for state in first.states_name:
        np.testing.assert_allclose(
            first.states.get_mean(state, "smooth"),
            second.states.get_mean(state, "smooth"),
            atol=1e-9,
        )


def test_offline_training_resets_ewma_at_epoch_start():
    num_steps = 60
    series = pd.DataFrame(
        {"values": 0.2 + np.sin(np.arange(num_steps) * 0.5)},
        index=pd.date_range("2000-01-01", periods=num_steps, freq="D"),
    )
    train, validation, _, _ = DataProcess(
        data=series, train_split=0.8, validation_split=0.2, output_col=[0]
    ).get_splits()
    model = make_model()
    before = copy.deepcopy(model.lstm_net.state_dict())
    for _ in range(2):
        mean, std, _ = model.lstm_train(train, validation, white_noise_decay=False)
        assert np.all(np.isfinite(mean)) and np.all(std > 0)
        np.testing.assert_array_equal(model.mu_states[model._ewma_indices], 0)
        np.testing.assert_array_equal(model.var_states[model._ewma_indices, :], 0)
        np.testing.assert_array_equal(model.var_states[:, model._ewma_indices], 0)
    assert before != model.lstm_net.state_dict()


def test_serialization_preserves_constraint_and_next_prediction():
    model = make_model()
    model.lstm_net.num_samples = 20
    model.lstm_net.eval()
    model.filter(data(8), train_lstm=False)
    model.lstm_net.set_lstm_states(model.lstm_net.get_lstm_states(7))
    restored = Model.load_dict(pickle.loads(pickle.dumps(model.get_dict())))
    np.testing.assert_allclose(restored.mu_states, model.mu_states)
    np.testing.assert_allclose(restored.var_states, model.var_states)
    assert restored.ewma_constraints[0].source == "lstm"
    restored.initialize_states_history()
    restored.lstm_net.num_samples = 20
    restored.lstm_net.eval()
    expected = model.forecast(data(3))
    actual = restored.forecast(data(3))
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=1e-6, atol=1e-7)


def test_slope_calibration_for_steady_ramp():
    constraint = EwmaConstraint(source="level", fast_span=4, slow_span=16, dt=0.5)
    model = Model(LocalLevel(), constraint)
    slope, intercept = 0.03, 0.2
    memory = intercept - slope * constraint.lags
    for t in range(1, 15):
        value = intercept + slope * constraint.dt * t
        memory = constraint.rhos * memory + constraint.weights * value
    mean = np.array([[value], *[[v] for v in memory]])
    matrix, _ = constraint.constraint_observation(
        model.num_states, memory_indices(model, constraint)
    )
    assert (matrix @ mean)[1, 0] == pytest.approx(slope)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fast_span": 0},
        {"fast_span": 48},
        {"slow_span": np.inf},
        {"dt": 0},
        {"dt": np.nan},
        {"tau_mean": 0},
        {"tau_slope": -1},
        {"tau_slope": np.nan},
        {"tau_mean": 1e200},
        {"tau_mean": 1e-200},
        {"source": ""},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        EwmaConstraint(**kwargs)


def test_missing_duplicate_and_recursive_source_rejected():
    with pytest.raises(ValueError, match="Unknown EwmaConstraint source"):
        Model(LocalLevel(), EwmaConstraint())
    with pytest.raises(ValueError, match="distinct source"):
        Model(LocalLevel(), EwmaConstraint("level"), EwmaConstraint("level"))
    with pytest.raises(ValueError, match="another EWMA"):
        Model(LocalLevel(), EwmaConstraint("level"), EwmaConstraint("level ewma fast"))


def abnormal_model(*constraints):
    return Model(
        LocalAcceleration(),
        LstmNetwork(num_hidden_unit=4, manual_seed=11),
        WhiteNoise(std_error=0.1),
        *constraints,
    )


def test_switching_filter_requires_matching_constraints():
    with pytest.raises(ValueError, match="matching EwmaConstraint"):
        SKF(make_model(), abnormal_model())
    with pytest.raises(ValueError, match="matching EwmaConstraint"):
        SKF(make_model(constrained=False), abnormal_model(EwmaConstraint()))


def test_switching_filter_reindexes_constraints_after_alignment():
    skf = SKF(make_model(), abnormal_model(EwmaConstraint(fast_span=4, slow_span=16)))
    for model in skf.model.values():
        constraint = model.ewma_constraints[0]
        fast, slow = memory_indices(model, constraint)
        source = model.get_states_index("lstm")
        np.testing.assert_allclose(
            model._ewma_coupling[[fast, slow], source], constraint.weights
        )
    probability, states = skf.filter(data())
    skf.smoother()
    assert np.all(np.isfinite(probability))
    assert np.any(np.hstack(states.mu_posterior)[skf.model["norm_norm"]._ewma_indices])


@pytest.mark.skipif(not HAS_RSR, reason="needs online LSTM learning (canari.rsr)")
@pytest.mark.parametrize(
    "strategy",
    ["cached_filtered", "cached_smoothed", "replay_filtered", "replay_smoothed"],
)
def test_rsr_keeps_constraints_active_and_remains_causal(strategy):
    model = make_model()
    mean, std, states = model.rsr_filter(
        data(10), start=3, window_len=3, end=7, strategy=strategy, include_warmup=True
    )
    assert len(mean) == len(std) == len(states.mu_prior) == 7
    assert np.all(np.isfinite(mean)) and np.all(std > 0)
    assert np.any(np.hstack(states.mu_posterior)[model._ewma_indices])
    changed = data(10)
    changed["y"][6:] += 100
    causal = make_model().rsr_filter(
        changed, start=3, window_len=3, end=7, strategy=strategy, include_warmup=True
    )
    np.testing.assert_array_equal(mean, causal[0])


@pytest.mark.skipif(not HAS_RSR, reason="needs online LSTM learning (canari.rsr)")
@pytest.mark.parametrize("strategy", ["replay_filtered", "replay_smoothed"])
def test_rsr_replays_windows_from_joint_memories(strategy):
    model = make_model()
    starts, smoothed = [], []
    original_filter, original_smoother = model.filter, model.rts_smoother

    def record_filter(window, train_lstm=True):
        starts.append((model.mu_states.copy(), model.var_states.copy()))
        return original_filter(window, train_lstm=train_lstm)

    def record_smoother(step, *args, **kwargs):
        original_smoother(step, *args, **kwargs)
        if step == 0:
            smoothed.append(
                (model.states.mu_smooth[0].copy(), model.states.var_smooth[0].copy())
            )

    model.filter, model.rts_smoother = record_filter, record_smoother
    _, _, states = model.rsr_filter(
        data(10), start=3, window_len=3, end=7, strategy=strategy, include_warmup=True
    )
    boundaries = (
        list(zip(states.mu_posterior, states.var_posterior))
        if strategy == "replay_filtered"
        else smoothed
    )
    for actual, expected in zip(starts[1:], boundaries):
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-12)


@pytest.mark.skipif(
    not HAS_SKF_ONLINE_TRAINING, reason="needs online LSTM training in SKF.filter"
)
def test_switching_filter_trains_from_raw_prediction():
    skf = SKF(make_model(), abnormal_model(EwmaConstraint(fast_span=4, slow_span=16)))
    raw_mu, raw_var = 0.7, 0.6
    fix_lstm_output(skf.lstm_net, raw_mu, raw_var)
    messages = record_lstm_messages(skf.lstm_net)
    skf.filter(data(1), train_lstm=True, reset_memory=False)
    index = skf.model["norm_norm"].get_states_index("lstm")
    dm, dv = messages[0]
    assert raw_mu + raw_var * dm == pytest.approx(
        skf.mu_states_posterior[index, 0], abs=2e-7
    )
    assert raw_var + raw_var**2 * dv == pytest.approx(
        skf.var_states_posterior[index, index], abs=2e-7
    )


@pytest.mark.skipif(not HAS_AUXILIARY, reason="needs the Auxiliary component")
def test_auxiliary_forecaster_output_is_constrained():
    def predict_fn(context, horizon):
        return np.full(horizon, 1.0), np.full(horizon, 0.5)

    model = Model(
        canari.component.Auxiliary(predict_fn=predict_fn),
        WhiteNoise(std_error=0.1),
        EwmaConstraint(source="auxiliary", fast_span=4, slow_span=16, tau_mean=0.05),
    )
    missing = data(8)
    missing["y"][:] = np.nan
    _, _, states = model.filter(missing)
    np.testing.assert_allclose(states.get_mean("auxiliary", "prior"), 1.0)
    posterior = states.get_mean("auxiliary", "posterior")
    assert np.all(posterior < 1.0)
    np.testing.assert_allclose(model.aux_component.context[-8:], posterior)
