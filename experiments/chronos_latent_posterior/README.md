# Chronos latent-posterior uncertainty experiment

A staged experiment with a visual walkthrough of Bayesian uncertainty
propagation. Chronos stays frozen; all code is outside `src/`.

- [Exploratory bounded-model report](results/run_20261007T181022_448332Z_bounded/report.md)
- [Numbered figure guide](results/run_20261007T181022_448332Z_bounded/figure_guide.md)
- [Printable figure guide](results/run_20261007T181022_448332Z_bounded/guide.pdf)
- [Original unbounded experiment and failure](results/run_20261007T175118_194212Z/report.md)
- [Protocol and mathematical assumptions](planning/README.md)
- [Inference verification](results/run_20261007T175118_194212Z/verification.json)

The main mechanism test generates and filters latent series using the same
Chronos median transition. A fully adapted particle filter propagates correlated
latent contexts; one-step uncertainty splits into state uncertainty and known
future process/measurement noise. Recursive multi-step simulations diagnose
the finite-sampling correction needed for that decomposition.

The original unbounded recursion failed in one of five episodes. An explicitly
exploratory follow-up uses the bounded mean `3*tanh(Chronos median/3)` in both
truth and inference. This changes the model and supplies an amplitude prior;
it does not recover a unique split from the original intervals. See the
[follow-up protocol](planning/bounded_followup.md). Both runs are preserved.
The `plot_states`-style view includes prior observation forecasts and filtered
posterior signal, transition mean, process shock and measurement-noise rows.

Separate controls verify exact Gaussian joint-history inference, pass an AR
posterior through Chronos, fit noise using training observations, and compute
a three-term Bayesian decomposition under a discrete Q/R prior. The AR-filtered
forecast is explicitly a hybrid conditional model. Neither stage uniquely
decomposes Chronos's native quantile distribution or infers its weight posterior.

Run from the repository root in the existing environment:

```sh
/opt/miniconda3/envs/pycanari/bin/python -m pytest experiments/chronos_latent_posterior/scripts -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/miniconda3/envs/pycanari/bin/python experiments/chronos_latent_posterior/scripts/run_experiment.py --smoke
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/miniconda3/envs/pycanari/bin/python experiments/chronos_latent_posterior/scripts/run_experiment.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/miniconda3/envs/pycanari/bin/python experiments/chronos_latent_posterior/scripts/run_bounded.py <original-completed-run>
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/miniconda3/envs/pycanari/bin/python experiments/chronos_latent_posterior/scripts/collect_states_example.py <completed-run>
MPLBACKEND=Agg /opt/miniconda3/envs/pycanari/bin/python experiments/chronos_latent_posterior/scripts/make_report.py <new-run-directory>
/Users/davidwardan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 experiments/chronos_latent_posterior/scripts/make_pdf_guide.py <new-run-directory>
```

The PDF-guide script uses the bundled document Python runtime reported by
`load_workspace_dependencies`, with reportlab and pypdf. It preserves the vector
figure PDFs and places one figure and explanatory caption on each page.

Every inference run has a new timestamped results directory. Reporting creates
a matching figure directory with PDF, PGF and PNG exports. Raw scores, examples,
posterior members, true synthetic series, particle diagnostics, noise fits,
parameter-grid weights, seed choices and source hashes are saved. Previous
experiments are never overwritten. Run reporting once per completed run.
