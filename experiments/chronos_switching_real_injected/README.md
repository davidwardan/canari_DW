# Chronos-2 small on real data with an injected anomaly

This experiment tests the switching posterior on real weekly benchmark data rather
than a synthetic generator. It uses the raw `ts67` and `ts66` series, the original
train-only standardization, and train-only (Q/R) noise estimates. A +2.5
standardized-unit level residual is injected for 12 held-out test weeks; a paired
clean run uses the unchanged observations.

The completed run is documented in:

- [report](results/run_20261007T204754_651289Z/report.md)
- [planning and leakage controls](planning/README.md)
- [figures](figures/run_20261007T204754_651289Z/)

Run the focused tests with:

```bash
/opt/miniconda3/envs/pycanari/bin/python -m pytest -q \
  experiments/chronos_switching_real_injected/scripts/test_real_injected.py
```

Re-run offline with the cached Chronos-2 small checkpoint:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/opt/miniconda3/envs/pycanari/bin/python \
  experiments/chronos_switching_real_injected/scripts/run_experiment.py
```

Generate figures and the report from saved outputs:

```bash
/opt/miniconda3/envs/pycanari/bin/python \
  experiments/chronos_switching_real_injected/scripts/make_report.py \
  experiments/chronos_switching_real_injected/results/run_20261007T204754_651289Z
```
