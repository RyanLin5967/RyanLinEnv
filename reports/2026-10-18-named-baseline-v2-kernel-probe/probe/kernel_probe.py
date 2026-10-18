"""Kernel-path probe for the named-baseline v2 CPU engine (TRAIN images only; not a score).

One process = one kernel configuration, because MKL reads MKL_CBWR once at initialisation.
Runs the v2 smoke path (same TransformersBackend, prompt, sampler) over TRAIN images and
records, per image: answer, drawn token ids, latency, prefill, per-step decode intervals, and
the SHA256 + top-5 of the first-step score row (a numerics fingerprint far stronger than the
5-7-token answer). Then a text-only forced-length greedy decode measures s/token directly.
CPU time vs wall (getrusage) and /proc/loadavg are recorded around every phase, so host load
can be told apart from kernel cost.
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

CONFIGS = {
    "pinned": ({"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "AVX"}, False),
    "unpinned": ({}, True),
    "nodnn": ({}, False),
    "cbwr": ({"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "AVX"}, True),
    # oneDNN on but capped at the AVX ISA every pool CPU has, with ATen and MKL pinned too.
    "dnnavx": ({"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "AVX",
                "ONEDNN_MAX_CPU_ISA": "AVX"}, True),
}

parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=CONFIGS, required=True)
parser.add_argument("--ids", nargs="+", required=True)
parser.add_argument("--threads", type=int, required=True)
parser.add_argument("--decode-tokens", type=int, default=16)
parser.add_argument("--weights", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

env, mkldnn = CONFIGS[args.config]
for key in ("ATEN_CPU_CAPABILITY", "MKL_CBWR", "ONEDNN_MAX_CPU_ISA", "DNNL_MAX_CPU_ISA"):
    os.environ.pop(key, None)
os.environ.update(env)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "src"), str(HERE / "scripts")]


def loadavg():
    try:
        return Path("/proc/loadavg").read_text().split()[:3]
    except OSError:
        return None


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("v2", HERE / "scripts/aircraft_named_baseline_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
local_hf = v2.local_hf
import torch  # noqa: E402

observed = {}


def pin(_numerics):
    torch.backends.mkldnn.enabled = mkldnn
    observed.update({
        "aten_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "mkl_cbwr": os.environ.get("MKL_CBWR"),
        "aten_cpu_capability_env": os.environ.get("ATEN_CPU_CAPABILITY"),
        "mkl_available": torch.backends.mkl.is_available(),
    })
    return dict(observed)


local_hf.pin_cpu_numerics = pin
fingerprints = []
Base = local_hf.MLXEquivalentSampler


class Recording(Base):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.record = {"cpu_start": cpu_seconds(), "wall_start": time.monotonic(),
                       "loadavg_start": loadavg()}
        fingerprints.append(self.record)

    def __call__(self, input_ids, scores):
        if not self.drawn:
            row = scores[0].float().contiguous()
            self.record["first_scores_sha256"] = hashlib.sha256(row.numpy().tobytes()).hexdigest()
            top = torch.topk(row, 5)
            self.record["first_top5"] = [[int(i), float(v)]
                                         for v, i in zip(top.values, top.indices)]
            self.record["cpu_at_first_token"] = cpu_seconds() - self.record["cpu_start"]
        out = super().__call__(input_ids, scores)
        self.record["drawn"] = list(self.drawn)
        self.record["step_times"] = [t - self.record["wall_start"] for t in self.step_times]
        self.record["cpu_total"] = cpu_seconds() - self.record["cpu_start"]
        self.record["loadavg_end"] = loadavg()
        return out


local_hf.MLXEquivalentSampler = Recording
backends = []


def factory(directory, config):
    backend = local_hf.TransformersBackend(directory, config)
    backends.append(backend)
    return backend


repository = HERE
plan = v2.load_plan(HERE / "docs/aircraft-named-baseline-plan-v2.json", require_activated=False)
smoke = v2.smoke(plan, repository, args.weights, args.ids, args.threads, "float32",
                 backend_factory=factory)
for result, record in zip(smoke["results"], fingerprints):
    steps = record.get("step_times", [])
    record["decode_seconds_per_token"] = (
        (steps[-1] - steps[0]) / (len(steps) - 1) if len(steps) > 1 else None)
    result["fingerprint"] = record

# Forced-length text-only greedy decode: s/token with negligible prefill.
backend = backends[0]
import copy  # noqa: E402

from transformers import LogitsProcessorList  # noqa: E402

messages = [{"role": "user", "content": [{"type": "text", "text":
            "Write the numbers from one to two hundred in English words, separated by commas."}]}]
prompt = backend.processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
inputs = backend.processor(text=[prompt], return_tensors="pt")
stamps = []


def stamp(input_ids, scores):
    stamps.append((time.monotonic(), cpu_seconds()))
    return scores


config = copy.deepcopy(backend.neutral)
config.max_new_tokens = config.min_new_tokens = args.decode_tokens
load_before = loadavg()
start, cpu0 = time.monotonic(), cpu_seconds()
with torch.inference_mode():
    output = backend.model.generate(**inputs, generation_config=config,
                                    logits_processor=LogitsProcessorList([stamp]))
end = time.monotonic()
intervals = [b[0] - a[0] for a, b in zip(stamps, stamps[1:])]
cpu_intervals = [b[1] - a[1] for a, b in zip(stamps, stamps[1:])]
decode = {
    "prompt_tokens": int(inputs["input_ids"].shape[1]), "tokens": args.decode_tokens,
    "prefill_seconds": stamps[0][0] - start,
    "seconds_per_token_mean": sum(intervals) / len(intervals),
    "seconds_per_token_min": min(intervals), "seconds_per_token_max": max(intervals),
    "cpu_cores_busy_during_decode": sum(cpu_intervals) / sum(intervals),
    "intervals": intervals, "token_ids": output[0, inputs["input_ids"].shape[1]:].tolist(),
    "loadavg_start": load_before, "loadavg_end": loadavg(), "wall_seconds": end - start,
}
report = {"config": args.config, "threads": args.threads, "observed": observed,
          "torch_num_threads": torch.get_num_threads(),
          "env": {k: os.environ.get(k) for k in ("ATEN_CPU_CAPABILITY", "MKL_CBWR",
                                                  "ONEDNN_MAX_CPU_ISA",
                                                  "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
          "smoke": smoke, "decode_bench": decode,
          "nice": os.nice(0)}
Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"config": args.config, "answers": [r["text"] for r in smoke["results"]],
                  "latency": [round(r["latency_seconds"], 1) for r in smoke["results"]],
                  "decode_s_per_tok": round(decode["seconds_per_token_mean"], 2),
                  "cores_busy": round(decode["cpu_cores_busy_during_decode"], 1),
                  "fp": (smoke.get("kernel_fingerprint") or "")[:12],
                  "sha": [r["fingerprint"].get("first_scores_sha256", "")[:12]
                          for r in smoke["results"]]}), flush=True)
