"""Kernel fingerprint under one kernel configuration (one process per config; no model)."""
import importlib.util
import json
import os
import sys
from pathlib import Path

CONFIGS = {
    "pinned": ({"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "AVX"}, False),
    "dnnavx": ({"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "AVX",
                "ONEDNN_MAX_CPU_ISA": "AVX"}, True),
    "unpinned": ({}, True),
}
config = sys.argv[1]
env, mkldnn = CONFIGS[config]
for key in ("ATEN_CPU_CAPABILITY", "MKL_CBWR", "ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA"):
    os.environ.pop(key, None)
os.environ.update(env)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
spec = importlib.util.spec_from_file_location("hf", HERE / "scripts/aircraft_named_baseline_hf.py")
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)
import torch  # noqa: E402

torch.backends.mkldnn.enabled = mkldnn
print(json.dumps({"config": config, "host": hf.host_identity(),
                  "capability": torch.backends.cpu.get_cpu_capability(),
                  "fp32": hf.kernel_fingerprint(32), "fp16": hf.kernel_fingerprint(16),
                  "fp32_again": hf.kernel_fingerprint(32)}), flush=True)
