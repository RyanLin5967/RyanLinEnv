"""One subprocess = the engine's own pin (pin_cpu_numerics with the plan's numerics, warm-up
included), then the elementwise ops that SIGILLed, at the plan's 32 threads."""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
spec = importlib.util.spec_from_file_location("hf", HERE / "scripts/aircraft_named_baseline_hf.py")
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)
plan = json.loads((HERE / "docs/aircraft-named-baseline-plan-v2.json").read_text())
observed = hf.pin_cpu_numerics(plan["engine"]["cpu_numerics"])
import torch  # noqa: E402

torch.set_num_threads(plan["engine"]["threads"])
torch.manual_seed(int(sys.argv[1]))
total = 0.0
for n in (4096, 65536, 262144, 1048576):
    x = torch.rand(n) * 20 - 10
    for op in (torch.cos, torch.sin, torch.exp, torch.tanh):
        total += float(op(x).sum())
    total += float(torch.log(x.abs() + 1).sum())
    pos = torch.arange(n // 64, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    freqs = torch.outer(pos, inv)
    total += float(freqs.cos().sum() + freqs.sin().sum())
print(json.dumps({"ok": True, "warmup": observed["vml_single_thread_warmup"], "total": total}))
