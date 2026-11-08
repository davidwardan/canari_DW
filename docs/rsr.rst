Online LSTM learning with RSR
=============================

Recurrent Smoothing and Replay (RSR) keeps a model's LSTM learning while the series is
filtered. ``Model.rsr_filter`` walks overlapping windows of ``window_len + 1``
observations: the LSTM parameters are updated at every step of a window, the window is
smoothed backwards, and the next window restarts one step later from the smoothed
recurrent memory. Every time step receives its own one-step-ahead prediction, made
before its observation is used.

.. code-block:: python

   from canari import Model
   from canari.component import LocalTrend, LstmNetwork, WhiteNoise

   model = Model(
       LocalTrend(),
       LstmNetwork(look_back_len=24, num_features=2, infer_len=24, num_hidden_unit=40),
       WhiteNoise(std_error=0.1),
   )

   mean, std, states = model.rsr_filter(data=all_data, start=24, window_len=24)

The LSTM must use ``smoother=True`` (the default), and ``start`` must be at least
``window_len``. Everything before ``start - window_len`` is first filtered with frozen
LSTM parameters. The returned predictions and states cover ``data[start:end]``; with
``include_warmup=True`` they also cover the first window.

Strategies
----------

The ``strategy`` argument sets how the state-space model is handled while the LSTM
replays the older steps of a window:

- ``"replay_smoothed"`` (default): re-filter the whole window, starting from the
  preceding window's smoothed boundary.
- ``"replay_filtered"``: re-filter the whole window, starting from the causal posterior
  saved just before it.
- ``"cached_smoothed"``: train the LSTM on the older steps against the smoothed
  baseline from the preceding window; the state-space model advances once per window.
- ``"cached_filtered"``: same, against the baseline prior recorded when each step first
  arrived.

All four perform the same number of LSTM updates and smooth the LSTM's own cell and
hidden states. They are algorithm variants, not exact Bayesian inference over the
window.

After offline training
----------------------

``rsr_filter`` can continue from a network trained offline with ``lstm_train``, e.g.
with ``start`` at the beginning of the test split. pyTAGI sizes the SLSTM's sequence
buffer on its first forward pass and does not grow it afterwards; offline training
fixes it at ``infer_len + len(train)`` steps, and the frozen pass before the first
window needs ``start - window_len`` of them. With ``start`` at the test split, the
validation split must therefore be shorter than ``infer_len + window_len`` steps.

Resuming from a saved model
---------------------------

``get_dict``/``load_dict`` keep the LSTM's recurrent states, so a restored model
continues where it stopped. Calling ``rsr_filter`` with ``start=window_len`` skips the
frozen pass and starts from the model's current states and LSTM look-back.

Examples
--------

- ``examples/toy_rsr_stationary.py``: an untrained LSTM learns a stationary series in
  one pass, then forecasts the test split with frozen parameters.
- ``examples/toy_rsr_trend.py``: the same on a rising level; a ``LocalTrend`` carries
  the rise so the LSTM learns only the recurrent pattern.
- ``examples/toy_rsr_pretrained.py``: offline training with a validation split, then
  RSR on the test split.
