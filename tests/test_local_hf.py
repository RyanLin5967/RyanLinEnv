"""Sampler equivalence of the CPU transformers engine with mlx-vlm 0.6.8 (torch only).

The reference below is written from mlx_vlm.sample_utils' definitions with plain Python
loops, independently of local_hf's tensor code.
"""

import importlib.util
import math
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/aircraft_named_baseline_hf.py"
_spec = importlib.util.spec_from_file_location("aircraft_named_baseline_hf", _SCRIPT)
local_hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(local_hf)

SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "max_tokens": 16384}


def reference(logits, history, *, temperature, top_p, top_k, presence_penalty, context=20):
    values = list(logits)
    for token in set(history[-context:]):
        values[token] -= presence_penalty
    norm = math.log(sum(math.exp(v) for v in values))
    logprobs = [v - norm for v in values]
    ascending = sorted(range(len(values)), key=lambda i: logprobs[i])
    running, keep = 0.0, set()
    for index in ascending:
        running += math.exp(logprobs[index])
        if running > 1 - top_p:
            keep.add(index)
    masked = [lp if i in keep else -math.inf for i, lp in enumerate(logprobs)]
    ranked = sorted(range(len(values)), key=lambda i: masked[i], reverse=True)[:top_k]
    return [masked[i] / temperature if i in set(ranked) else -math.inf
            for i in range(len(values))]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_filter_matches_mlx_definition(seed):
    rng = random.Random(seed)
    logits = [rng.gauss(0, 3) for _ in range(64)]
    generated = [rng.randrange(64) for _ in range(30)]
    got = local_hf.mlx_filter(torch.tensor([logits], dtype=torch.float32), generated,
                              temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                              presence_penalty=1.5)[0].tolist()
    want = reference(logits, generated, temperature=0.7, top_p=0.8, top_k=20,
                     presence_penalty=1.5)
    assert [math.isinf(v) for v in got] == [math.isinf(v) for v in want]
    for g, w in zip(got, want):
        if not math.isinf(w):
            assert g == pytest.approx(w, abs=1e-4)


def test_presence_history_follows_mlx_prefill_rule():
    # mlx-vlm 0.6.8 generate_step: the whole prompt when it is prefilled in one step,
    # only its last token when a prompt longer than 2048 tokens is prefilled in chunks.
    short, long = list(range(100, 2148)), list(range(100, 2149))
    assert len(short) == 2048 and len(long) == 2049
    assert local_hf.presence_history(short, [5, 6]) == short + [5, 6]
    assert local_hf.presence_history(long, [5, 6]) == [2148, 5, 6]
    assert local_hf.presence_history([1, 2, 3], []) == [1, 2, 3]


def test_sampler_penalises_prompt_tail_at_the_first_step():
    scores = torch.zeros(1, 40)
    sampler = local_hf.MLXEquivalentSampler(dict(SAMPLING, top_p=1.0, top_k=40), 0, 3)
    filtered = []
    original = local_hf.mlx_filter

    def spy(scores, history, **kwargs):
        filtered.append(list(history))
        return original(scores, history, **kwargs)

    local_hf.mlx_filter = spy
    try:
        sampler(torch.tensor([[7, 8, 9]]), scores.clone())
    finally:
        local_hf.mlx_filter = original
    assert filtered == [[7, 8, 9]]


def test_presence_penalty_only_counts_last_twenty_history_tokens():
    logits = torch.zeros(1, 30)
    out = local_hf.mlx_filter(logits, [5] + list(range(10, 30)), temperature=1.0, top_p=1.0,
                              top_k=30, min_p=0.0, presence_penalty=1.5)[0]
    assert out[5] > out[10]  # token 5 is 21 tokens back: not penalised
    assert out[10] == pytest.approx(out[29])


def test_min_p_is_refused():
    with pytest.raises(ValueError, match="min_p"):
        local_hf.mlx_filter(torch.zeros(1, 4), [], temperature=1.0, top_p=1.0, top_k=2,
                            min_p=0.1, presence_penalty=0.0)


def test_sampler_forces_its_draw_and_is_seeded():
    prompt = torch.tensor([[7, 8, 9]])
    scores = torch.randn(1, 50, generator=torch.Generator().manual_seed(3))
    first = local_hf.MLXEquivalentSampler(SAMPLING, 1234, 3)
    forced = first(prompt, scores.clone())
    token = first.drawn[0]
    assert forced[0, token] == 0 and torch.isinf(forced).sum() == 49
    second = local_hf.MLXEquivalentSampler(SAMPLING, 1234, 3)
    second(prompt, scores.clone())
    assert second.drawn == [token]
    follow = torch.cat([prompt, torch.tensor([[token]])], dim=1)
    first(follow, scores.clone())
    assert len(first.drawn) == 2
    wrong = torch.cat([prompt, torch.tensor([[(token + 1) % 50]])], dim=1)
    with pytest.raises(ValueError, match="did not draw"):
        second(wrong, scores.clone())


def test_draws_follow_the_filtered_distribution():
    scores = torch.tensor([[2.0, 1.0, 0.5, -1.0, -3.0]])
    filtered = local_hf.mlx_filter(scores, [], temperature=0.7, top_p=0.8, top_k=20,
                                   min_p=0.0, presence_penalty=1.5)
    expected = torch.softmax(filtered, dim=-1)[0]
    counts = torch.zeros(5)
    for seed in range(4000):
        sampler = local_hf.MLXEquivalentSampler(SAMPLING, seed, 0)
        sampler(torch.zeros(1, 0, dtype=torch.long), scores.clone())
        counts[sampler.drawn[0]] += 1
    assert counts[3] == counts[4] == 0  # outside top-p
    assert (counts / 4000 - expected).abs().max() < 0.03


def test_kernel_fingerprint_is_stable_and_fires_on_a_changed_kernel(monkeypatch):
    threads = torch.get_num_threads()
    try:
        first = local_hf.kernel_fingerprint(2)
        assert len(first) == 64 and local_hf.kernel_fingerprint(2) == first
        original = torch.nn.functional.linear

        def nudged(x, weight, bias=None):  # one ulp-scale change in a single kernel
            return original(x, weight, bias) * (1 + 2.0 ** -20)

        monkeypatch.setattr(torch.nn.functional, "linear", nudged)
        assert local_hf.kernel_fingerprint(2) != first
        monkeypatch.undo()
        assert local_hf.kernel_fingerprint(2) == first
    finally:
        torch.set_num_threads(threads)


def test_kernel_pin_refuses_a_onednn_isa_other_than_the_plan_once_torch_is_loaded(monkeypatch):
    numerics = {"aten_cpu_capability_env": "default", "mkl_cbwr": "AVX",
                "onednn_max_cpu_isa": "AVX"}
    monkeypatch.setenv("ATEN_CPU_CAPABILITY", "default")
    monkeypatch.setenv("MKL_CBWR", "AVX")
    monkeypatch.delenv("DNNL_MAX_CPU_ISA", raising=False)
    for isa in ("AVX2", None):
        if isa is None:
            monkeypatch.delenv("ONEDNN_MAX_CPU_ISA", raising=False)
        else:
            monkeypatch.setenv("ONEDNN_MAX_CPU_ISA", isa)
        with pytest.raises(ValueError, match="loaded before the CPU kernel path"):
            local_hf.pin_cpu_numerics(numerics)
    monkeypatch.setenv("ONEDNN_MAX_CPU_ISA", "AVX")
    monkeypatch.setenv("DNNL_MAX_CPU_ISA", "AVX")  # the legacy alias must stay unset
    with pytest.raises(ValueError, match="loaded before the CPU kernel path"):
        local_hf.pin_cpu_numerics(numerics)


def test_vml_warmup_runs_each_op_single_threaded_then_restores(monkeypatch):
    original = torch.get_num_threads()
    torch.set_num_threads(3)  # neither 1 nor the default, so "restored" is a real check
    previous = torch.get_num_threads()
    assert previous == 3
    seen = {}
    for name in local_hf.VML_WARMUP_OPS:
        real = getattr(torch, name)

        def record(x, real=real, name=name):
            seen.setdefault(name, torch.get_num_threads())
            return real(x)

        monkeypatch.setattr(torch, name, record)
    try:
        assert local_hf.vml_warmup(torch) is True
        assert seen == {name: 1 for name in ("cos", "sin", "exp", "tanh", "log")}
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(original)


def test_kernel_pin_applies_and_records_the_vml_warmup(monkeypatch):
    monkeypatch.setenv("ATEN_CPU_CAPABILITY", "default")
    monkeypatch.setenv("MKL_CBWR", "AVX")
    monkeypatch.setenv("ONEDNN_MAX_CPU_ISA", "AVX")
    monkeypatch.delenv("DNNL_MAX_CPU_ISA", raising=False)
    numerics = {"aten_cpu_capability_env": "default",
                "aten_cpu_capability": torch.backends.cpu.get_cpu_capability(),
                "mkldnn_enabled": torch.backends.mkldnn.enabled, "mkl_cbwr": "AVX",
                "mkl_available": torch.backends.mkl.is_available(),
                "onednn_max_cpu_isa": "AVX", "vml_single_thread_warmup": True}
    calls = []
    monkeypatch.setattr(local_hf, "vml_warmup", lambda module: calls.append(module) or True)
    observed = local_hf.pin_cpu_numerics(numerics)
    assert calls == [torch] and observed["vml_single_thread_warmup"] is True
    monkeypatch.setattr(local_hf, "vml_warmup", lambda module: False)
    with pytest.raises(ValueError, match="not the pre-registered one"):
        local_hf.pin_cpu_numerics(numerics)
