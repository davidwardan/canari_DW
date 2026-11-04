EWMA constraints for predictor outputs
======================================

``EwmaConstraint`` limits the sustained mean and drift of another hidden state while
leaving its short-term variations free. It targets a scalar **state name**, so the same
constraint works with the LSTM output, with the states of linear components, and with
any predictor that fills the model's predictor slot.

.. code-block:: python

   from canari import Model
   from canari.component import EwmaConstraint, LocalTrend, LstmNetwork, WhiteNoise

   model = Model(
       LocalTrend(),
       LstmNetwork(num_hidden_unit=16),
       WhiteNoise(std_error=0.1),
       EwmaConstraint(
           source="lstm",
           fast_span=12,
           slow_span=48,
           tau_mean=0.1,
           tau_slope=0.01,
       ),
   )

These parameters illustrate the API; choose durations and tolerances for your signal
and sampling frequency. Spans are exponential e-folding durations, not pandas EWMA
spans. ``dt`` defaults to one sample. ``tau_mean`` is in signal units; ``tau_slope`` is
in signal units per unit time. Smaller tolerances impose stronger constraints. Set
either tolerance to ``None`` to disable that pseudo-observation, or both to track the
EWMAs without constraining the source. Each source takes at most one constraint, and
an EWMA cannot itself be a source.

Training, filtering and forecasting
-----------------------------------

The component follows the ordinary ``Model`` workflows:

.. code-block:: python

   # Offline training: the constraint enters every training update, and each epoch
   # restarts from known zero EWMAs.
   for epoch in range(10):
       mean, std, states = model.lstm_train(train_data, validation_data)

   # Filtering with frozen parameters still applies the constraint.
   mean, std, states = model.filter(test_data, train_lstm=False)

   # Online parameter updates.
   mean, std, states = model.filter(new_data, train_lstm=True)

``forecast`` applies the constraint too, so multi-step forecasts keep the source's
long-term mean and drift near zero. ``smoother`` propagates the constraint evidence
backward. ``set_memory(time_step=0)`` resets the EWMAs to their known zero values; at
any other time step they are restored with the other hidden states.

The EWMAs are named after their source, e.g. ``lstm ewma fast`` and ``lstm ewma
slow``, and are available through ``StatesHistory``. Posterior and smoothed estimates
include the constraint. The prior history holds each step's prediction **before** the
constraint, which the smoother needs; for the LSTM, it is the network's raw output.

For a switching Kalman filter, include matching ``EwmaConstraint`` components in the
normal and abnormal models. Their indices are rebuilt when SKF aligns the state spaces.

Model configuration and joint hidden states, EWMAs included, use the standard
``get_dict``/``load_dict`` serialization.

Other sources
-------------

A linear component needs no special integration, e.g.
``EwmaConstraint(source="level")``. A predictor whose output fills the model's
predictor slot, through ``forward(mu_lstm_pred=..., var_lstm_pred=...)``, is
constrained the same way as the LSTM: its fresh prediction enters the EWMAs, and the
constraint conditions it before the observation does.

Mathematical definition
-----------------------

For source :math:`r_t`, the EWMAs obey

.. math::

   m_{j,t}=\rho_j m_{j,t-1}+(1-\rho_j)r_t,
   \qquad \rho_j=\exp(-\Delta t / s_j).

With :math:`\ell_j=\Delta t\,\rho_j/(1-\rho_j)`, the delay of each EWMA behind a steady
ramp, the two pseudo-observations are

.. math::

   0=m_{\mathrm{slow},t}+\epsilon_m,
   \qquad
   0=\frac{m_{\mathrm{fast},t}-m_{\mathrm{slow},t}}
            {\ell_{\mathrm{slow}}-\ell_{\mathrm{fast}}}+\epsilon_d,

where the independent noise standard deviations are ``tau_mean`` and ``tau_slope``. The
second one recovers the slope of a steady ramp after start-up transients. These are
Gaussian soft constraints, not hard bounds.

At each time step, the source's prediction is added to the EWMAs after the transition
step, and the joint prior is conditioned on both pseudo-observations, using Joseph
covariance updates, before the real observation. Smoothing uses the predictions before
these pseudo-observations, and the LSTM's learning messages compare the final
posterior with its **raw** output, so the network learns from both the constraint and
the observation. A deterministic source with deterministic EWMAs cannot be altered by
Gaussian conditioning.

See ``examples/toy_ewma.py`` for a runnable example: offline training, then filtering
the test split with frozen parameters and with online updates.
