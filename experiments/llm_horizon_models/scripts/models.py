"""The model list, plus access to the unchanged pipeline of `llm_horizon_degradation`."""

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
BASE_SCRIPTS = EXP_DIR.parent / "llm_horizon_degradation" / "scripts"
sys.path.insert(0, str(BASE_SCRIPTS))
import common  # noqa: E402  -- data, SSM, metrics and plot style, shared verbatim

# (Hugging Face id, short label, color, linestyle)
MODELS = [
    ("amazon/chronos-2", "Chronos-2", "tab:blue", "-"),
    ("autogluon/chronos-2-small", "Chronos-2 small", "tab:blue", "--"),
    ("amazon/chronos-bolt-tiny", "Bolt tiny", "#fdbe85", "-"),
    ("amazon/chronos-bolt-mini", "Bolt mini", "#fd8d3c", "-"),
    ("amazon/chronos-bolt-small", "Bolt small", "#e6550d", "-"),
    ("amazon/chronos-bolt-base", "Bolt base", "#a63603", "-"),
]

REFERENCE_RUN = EXP_DIR.parent / "llm_horizon_degradation" / "results" / "run_20260919_185524"


def slug(model_id):
    return model_id.split("/")[-1]
