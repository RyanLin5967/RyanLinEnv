"""Gate evaluator for the freshpool CPU dry run only: the real evaluator, update counts scaled.

Loads scripts/evaluate_freshpool_gates.py unchanged and replaces exactly two constants with the
dry run's (dryrun_scale.py): G2's update count (150 -> RL_SCREEN_UPDATES) and FINAL_UPDATES
(600 -> scale.FINAL_UPDATES). Every gate function, threshold and selection rule is the real
one. The dry-run plan pins both this file and the real evaluator by SHA256; a real launch plan
names the real evaluator, never this file.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dryrun_scale as scale  # noqa: E402

REAL = HERE.parent / "evaluate_freshpool_gates.py"
_spec = importlib.util.spec_from_file_location("evaluate_freshpool_gates_dryrun", REAL)
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)
_real.G2 = {**_real.G2, "updates": scale.RL_SCREEN_UPDATES}
_real.FINAL_UPDATES = scale.FINAL_UPDATES

# The launcher reads this file through runpy.run_path, so expose the real module's names.
globals().update({k: v for k, v in vars(_real).items() if not k.startswith("__")})

if __name__ == "__main__":
    _real.main()
