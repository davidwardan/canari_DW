"""Run the original AR(1) and sinusoid toy checks against every model.

    python scripts/model_check.py
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import BASE_SCRIPTS, MODELS, common

spec = importlib.util.spec_from_file_location("base_toy_check", BASE_SCRIPTS / "toy_check.py")
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)

if __name__ == "__main__":
    toy.check_block_structure()
    toy.check_kalman_closed_form()
    # The two checks above are the pipeline checks. The ones below test each
    # model's competence on known signals; a failure is a finding about that
    # model, so it is reported and the remaining models are still checked.
    failures = []
    for model_id, label, _, _ in MODELS:
        print(f"--- {label} ({model_id})")
        predict_fn = common.make_predict_fn(common.load_chronos(model_id))
        for check in (toy.check_ar1, toy.check_sinusoid):
            try:
                check(predict_fn)
            except AssertionError as error:
                print(f"    FAILED {check.__name__}: {error}")
                failures.append((label, check.__name__))
    print(f"model checks failed: {failures}" if failures else "all model checks passed")
