# Chronos-2 small switching posterior

This experiment extends the bounded Chronos latent-history particle filter with a
two-state normal/abnormal Markov regime. The abnormal state contributes an explicit
residual shift, so the predictive variance separates into known aleatoric noise,
within-regime latent-state uncertainty, and between-regime uncertainty.

The completed five-seed run is documented in:

- [report](results/run_20261007T203104_713548Z/report.md)
- [planning and model specification](planning/README.md)
- [figures](figures/run_20261007T203104_713548Z/)

It uses the cached Chronos-2 small revision
`ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a`, 32-step contexts, 64 primary
particles, a 128-particle seed-0 repeat, and five independent 96-step synthetic
episodes. The abnormal interval is steps 24--55 and the declared residual shift is
`+0.75`.

Run the numerical tests with:

```bash
/opt/miniconda3/envs/pycanari/bin/python -m pytest -q \
  experiments/chronos_switching_posterior/scripts/test_switching.py
```

The runner is offline and reproducible when the cached checkpoint is available:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/opt/miniconda3/envs/pycanari/bin/python \
  experiments/chronos_switching_posterior/scripts/run_experiment.py
```

Figures and the Markdown report are generated from saved run outputs with:

```bash
/opt/miniconda3/envs/pycanari/bin/python \
  experiments/chronos_switching_posterior/scripts/make_report.py \
  experiments/chronos_switching_posterior/results/run_20261007T203104_713548Z
```
