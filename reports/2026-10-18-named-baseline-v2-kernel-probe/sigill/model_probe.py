"""One subprocess = the plan's real engine path: pin, load, fingerprint, and one TRAIN image
(vision tower incl. rotary, full prefill, first token). Not a score; no VAL or TEST."""
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v2", HERE / "scripts/aircraft_named_baseline_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
local_hf, local_vlm = v2.local_hf, v2.local_vlm

plan = v2.load_plan(HERE / "docs/aircraft-named-baseline-plan-v2.json", require_activated=False)
inputs = plan["inputs"]
spec_file = HERE / inputs["dataset_root"] / inputs["data_spec"]["path"]
prompt = v2.prompt_text(v2.load_spec(spec_file), "answer_only")
config = {**local_vlm.native_profile(plan["generation"]["profile"]),
          "seed": plan["generation"]["seed"],
          "generation": {**plan["generation"]["sampling"], "max_tokens": 1},
          "engine": v2.engine_identity(plan), "preprocessing_id": v2.PREPROCESSING_ID,
          "stream_progress": False, "generation_progress_path": None, "identity_sha256": None}
start = time.monotonic()
backend = local_hf.TransformersBackend(sys.argv[1], config)
fingerprint = local_hf.kernel_fingerprint(plan["engine"]["threads"])
identifier = sys.argv[2]
image = HERE / f"data/fgvc-aircraft-2013b/development/images/train/{identifier}.jpg"
seed = int(local_vlm._digest([config["seed"], identifier])[:8], 16)
native = backend.infer({"id": identifier, "image_path": str(image), "prompt": prompt}, seed)
warmup = backend.load_info["device"]["cpu_numerics"].get("vml_single_thread_warmup")
matches = fingerprint == plan["engine"]["kernel_fingerprint_sha256"]
print(json.dumps({"ok": warmup is True and matches, "warmup": warmup,
                  "fingerprint_matches": matches,
                  "tokens": native["native_response"]["token_ids"],
                  "prefill_seconds": native["native_response"]["prefill_seconds"],
                  "wall_seconds": time.monotonic() - start}))
# A forward without the warm-up or with another fingerprint is not admission evidence:
# exit non-zero so run_warm.sh counts it as a failure rather than "ok".
sys.exit(0 if warmup is True and matches else 1)
