"""One subprocess = the plan's kernel pin, then elementwise ops like the reported SIGILLs."""
import json
import os
import sys

for key in ("ATEN_CPU_CAPABILITY", "MKL_CBWR", "ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA"):
    os.environ.pop(key, None)
os.environ.update({"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "AVX",
                   "ONEDNN_MAX_CPU_ISA": "AVX"})
import torch  # noqa: E402

if len(sys.argv) > 3 and sys.argv[3] == "warm":
    # one single-threaded pass through every VML op before any parallel region
    torch.set_num_threads(1)
    w = torch.rand(1 << 20)
    for op in (torch.cos, torch.sin, torch.exp, torch.tanh, torch.log):
        op(w)
torch.set_num_threads(int(sys.argv[1]))
torch.manual_seed(int(sys.argv[2]))
total = 0.0
for n in (4096, 65536, 262144, 1048576):
    x = torch.rand(n) * 20 - 10
    for op in (torch.cos, torch.sin, torch.exp, torch.tanh):
        total += float(op(x).sum())
    total += float(torch.log(x.abs() + 1).sum())
    # vision rotary: outer product of positions and inverse frequencies, then cos/sin
    pos = torch.arange(n // 64, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    freqs = torch.outer(pos, inv)
    total += float(freqs.cos().sum() + freqs.sin().sum())
print(json.dumps({"ok": True, "cpu": os.sched_getaffinity(0).__len__(), "total": total}))
