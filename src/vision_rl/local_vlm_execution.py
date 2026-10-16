"""Prospective local inference controls; importing this module allocates no model.

This is separate from the frozen v1 runner. Integration requires a new prepared
protocol whose identity includes this source. Completion limits are execution
boundaries, never changes to generation budgets or the fixed evaluation scope.
"""

from __future__ import annotations

import dataclasses
import time


def plan_chunk(ordered_ids, reserved, completed, stop_after_completions=None):
    """Choose only the next unattempted prefix, retaining every previous answer.

    ``reserved`` and ``completed`` are already replayed journal mappings. An
    ambiguous reservation is never retried. Malformed completed answers count
    toward completed coverage and are never regenerated.
    """
    ordered_ids = list(ordered_ids)
    expected = set(ordered_ids)
    if len(expected) != len(ordered_ids) or not expected:
        raise ValueError("The fixed scope must contain unique IDs")
    if (not set(reserved) <= expected or not set(completed) <= set(reserved)):
        raise ValueError("Journal coverage differs from the fixed scope")
    if stop_after_completions is not None and (
        type(stop_after_completions) is not int or stop_after_completions < 1
    ):
        raise ValueError("Completion limit must be a positive integer or None")
    pending = [identifier for identifier in ordered_ids if identifier not in reserved]
    hard_stopped = any(event.get("hard_stop") for event in completed.values())
    selected = [] if hard_stopped else pending[:stop_after_completions]
    return {
        "expected": len(ordered_ids),
        "completed_before": len(completed),
        "unknown_before": len(set(reserved) - set(completed)),
        "unattempted_before": len(pending),
        "selected_ids": selected,
        "deferred_ids": pending[len(selected):],
        "stop_after_completions": stop_after_completions,
        "hard_stop": hard_stopped,
        "boundary_policy": "Stop only after a newly completed answer is durable",
        "scope_complete": len(completed) == len(ordered_ids),
        "all_attempts_known": set(reserved) == set(completed),
    }


def missing_error(identifier, reserved, *, hard_stop=False, cooperative_stop=False):
    if identifier in reserved:
        return "unknown_after_interruption"
    if hard_stop:
        return "not_requested_hard_stop"
    if cooperative_stop:
        return "not_requested_cooperative_stop"
    return "not_requested"


def collect_stream(stream, *, result_factory, peak_memory, progress=None,
                   progress_tokens=64, progress_seconds=5.0, clock=time.monotonic):
    """Aggregate released MLX AR chunks without adding their cumulative usage.

    The last native chunk supplies finish reason, usage and timing. Text is the
    concatenation of every delta, including the final detokenizer flush. This is
    the released ``generate`` aggregation, restricted to autoregressive models.
    Progress observes the stream; it neither selects answers nor truncates it.
    """
    if type(progress_tokens) is not int or progress_tokens < 1 or progress_seconds <= 0:
        raise ValueError("Progress intervals must be positive")
    text, last = [], None
    started = last_emitted = clock()
    last_emitted_tokens, characters = 0, 0
    for response in stream:
        if response.is_draft:
            raise ValueError("This adapter only supports native autoregressive streams")
        text.append(response.text)
        characters += len(response.text)
        current = clock()
        if progress is not None and (
            last is None or response.finish_reason is not None
            or response.generation_tokens - last_emitted_tokens >= progress_tokens
            or current - last_emitted >= progress_seconds
        ):
            progress({
                "elapsed_seconds": current - started,
                "generation_tokens": response.generation_tokens,
                "prompt_tokens": response.prompt_tokens,
                "total_tokens": response.total_tokens,
                "prompt_tps": response.prompt_tps,
                "generation_tps": response.generation_tps,
                "peak_memory_decimal_gb": response.peak_memory,
                "text_characters": characters,
                "finish_reason": response.finish_reason,
                "usage_semantics": "Cumulative native counters, not additive chunks",
            })
            last_emitted, last_emitted_tokens = current, response.generation_tokens
        last = response
    if last is None:
        return result_factory(text="", peak_memory=peak_memory())
    return dataclasses.replace(last, text="".join(text))


def generate_with_progress(model, processor, prompt, *, progress=None, **kwargs):
    """Use mlx-vlm0.6.8's public AR stream with its normal EOS initialization.

    Only the currently pinned Qwen architecture and unchanged standard stopping
    criteria are supported. Real model parity is a separate prospective check;
    CPU parity tests execute the released aggregator against synthetic chunks.
    """
    if model.config.model_type != "qwen3_5":
        raise ValueError("Streaming contract is pinned to the local Qwen architecture")
    if any(kwargs.get(key) is not None for key in (
        "eos_tokens", "stopping_criteria", "draft_model", "thinking_budget",
    )):
        raise ValueError("Custom stopping, speculative decoding or forced thinking is unsupported")
    if kwargs.pop("verbose", False):
        raise ValueError("Use journal progress instead of verbose console output")
    import mlx.core as mx
    from mlx_vlm import stream_generate
    from mlx_vlm.generate import GenerationResult

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.stopping_criteria.reset(model.config.eos_token_id)
    stream = stream_generate(model, processor, prompt, verbose=False, **kwargs)
    return collect_stream(
        stream, result_factory=GenerationResult,
        peak_memory=lambda: mx.get_peak_memory() / 1e9, progress=progress,
    )
