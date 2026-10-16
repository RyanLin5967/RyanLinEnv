import dataclasses
import sys
from types import SimpleNamespace

import pytest

from vision_rl.local_vlm_execution import (
    collect_stream,
    generate_with_progress,
    missing_error,
    plan_chunk,
)


@dataclasses.dataclass
class NativeChunk:
    text: str = ""
    token: int | None = None
    logprobs: object = None
    prompt_tokens: int = 10
    generation_tokens: int = 0
    total_tokens: int = 10
    prompt_tps: float = 5.0
    generation_tps: float = 2.0
    peak_memory: float = 1.0
    cached_tokens: int = 0
    finish_reason: str | None = None
    is_draft: bool = False


def test_four_completion_chunks_keep_fixed16_and_never_repeat_unknown_or_bad_answers():
    identifiers = [str(i) for i in range(16)]
    reserved, completed, calls = {}, {}, []
    # Unknown interruption consumes its reservation; a malformed answer is final.
    reserved["0"] = {}
    reserved["1"] = {}
    completed["1"] = {"prediction": {"error": "malformed_answer"}}
    while True:
        plan = plan_chunk(identifiers, reserved, completed, 4)
        assert plan["expected"] == 16 and plan["unknown_before"] == 1
        if not plan["selected_ids"]:
            break
        assert len(plan["selected_ids"]) <= 4
        for identifier in plan["selected_ids"]:
            calls.append(identifier)
            reserved[identifier] = {}
            completed[identifier] = {"prediction": {"response": "native answer"}}
    assert calls == identifiers[2:]
    assert not plan["scope_complete"] and not plan["all_attempts_known"]
    assert missing_error("0", reserved, cooperative_stop=True) == "unknown_after_interruption"
    assert missing_error("next", reserved, cooperative_stop=True) == (
        "not_requested_cooperative_stop")
    assert missing_error("next", reserved, hard_stop=True) == "not_requested_hard_stop"


def test_chunks_stop_after_hard_failure_and_reject_invalid_limits_or_changed_scope():
    plan = plan_chunk(["a", "b"], {"a": {}}, {"a": {"hard_stop": True}}, 4)
    assert plan["selected_ids"] == [] and plan["deferred_ids"] == ["b"]
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            plan_chunk(["a"], {}, {}, limit)
    with pytest.raises(ValueError, match="unique"):
        plan_chunk(["a", "a"], {}, {})
    with pytest.raises(ValueError, match="scope"):
        plan_chunk(["a"], {"other": {}}, {})
    with pytest.raises(ValueError, match="scope"):
        plan_chunk(["a"], {}, {"a": {}})


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_stream_preserves_final_flush_finish_and_cumulative_usage(finish):
    chunks = [
        NativeChunk(text="thinking", token=4, generation_tokens=1, total_tokens=11),
        NativeChunk(text="</think>\n", token=5, generation_tokens=2, total_tokens=12),
        NativeChunk(text="A", token=6, generation_tokens=3, total_tokens=13,
                    finish_reason=finish, peak_memory=2.0),
    ]
    observed = []
    result = collect_stream(chunks, result_factory=NativeChunk, peak_memory=lambda: 99,
                            progress=observed.append, progress_tokens=2)
    assert result == dataclasses.replace(chunks[-1], text="thinking</think>\nA")
    assert result.total_tokens == 13  # Never sum cumulative chunk totals.
    assert len(observed) == 2 and observed[-1]["finish_reason"] == finish
    assert observed[-1]["text_characters"] == len(result.text)


def test_progress_time_boundary_empty_and_interrupted_stream():
    times = iter([0.0, 0.1, 6.0, 6.1])
    events = []
    chunks = [NativeChunk(text="a", generation_tokens=i) for i in (1, 2, 3)]
    result = collect_stream(chunks, result_factory=NativeChunk, peak_memory=lambda: 7,
                            progress=events.append, clock=lambda: next(times))
    assert result.finish_reason is None  # An absent final stop is never invented.
    assert len(events) == 2
    assert collect_stream([], result_factory=NativeChunk, peak_memory=lambda: 7).peak_memory == 7

    def broken():
        yield NativeChunk(text="partial")
        raise MemoryError("preserve the reservation")

    with pytest.raises(MemoryError, match="reservation"):
        collect_stream(broken(), result_factory=NativeChunk, peak_memory=lambda: 0)
    with pytest.raises(ValueError, match="autoregressive"):
        collect_stream([NativeChunk(is_draft=True)], result_factory=NativeChunk,
                       peak_memory=lambda: 0)


def test_public_stream_initializes_native_eos_and_forwards_unchanged_generation(monkeypatch):
    calls = {}
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5", eos_token_id=[1, 2]))
    processor = SimpleNamespace(tokenizer=SimpleNamespace(stopping_criteria=SimpleNamespace(
        reset=lambda value: calls.update(eos=value))))

    def stream(received_model, received_processor, prompt, **kwargs):
        assert received_model is model and received_processor is processor
        calls.update(prompt=prompt, kwargs=kwargs)
        yield NativeChunk(text="result", generation_tokens=1, finish_reason="stop")

    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=SimpleNamespace(
        get_peak_memory=lambda: 100)))
    monkeypatch.setitem(sys.modules, "mlx.core", sys.modules["mlx"].core)
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(stream_generate=stream))
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate",
                        SimpleNamespace(GenerationResult=NativeChunk))
    settings = {"max_tokens": 16384, "temperature": 1.0, "top_p": .95,
                "top_k": 20, "enable_thinking": True, "seed": 791}
    result = generate_with_progress(model, processor, "frozen prompt", **settings)
    assert result.text == "result" and calls["eos"] == [1, 2]
    assert calls["kwargs"] == {"verbose": False, **settings}
    with pytest.raises(ValueError, match="Custom stopping"):
        generate_with_progress(model, processor, "frozen prompt", thinking_budget=20)
