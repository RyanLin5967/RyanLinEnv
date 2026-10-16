import base64
import io
import json
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from PIL import Image

from vision_rl.frontier import _digest, _native, _request, run_frontier
from vision_rl.roboflow import RoboflowError


@pytest.fixture
def rows(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "private-frontier-key")
    result = []
    for i in range(5):
        path = tmp_path / f"image-{i}.png"
        Image.new("RGB", (12, 12), (i * 40, 30, 100)).save(path)
        result.append(
            {
                "id": str(i),
                "image": str(path),
                "prompt": "Count the objects.",
                "answer": "THIS LABEL MUST NEVER BE SENT",
                "split": "test",
            }
        )
    return result


def gemini(text="42", model="gemini-4-argon", finish="STOP"):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
        "modelVersion": model,
        "responseId": "request-id",
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2},
    }


def astra(text='{"detections": []}', status="completed", model="gpt-6-astra"):
    return {
        "id": "resp_astra",
        "model": model,
        "status": status,
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": text},
                ],
            },
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 200,
            "output_tokens_details": {"reasoning_tokens": 175},
            "total_tokens": 300,
        },
    }


def test_native_request_and_evidence_exclude_labels_keys_and_image_bytes(rows, tmp_path):
    output = tmp_path / "run"
    with patch("vision_rl.frontier._request", return_value=(200, gemini())) as request:
        predictions = run_frontier(rows[:1], output, max_requests=1)
    payload = request.call_args.args[1]
    assert payload["generationConfig"]["thinking_config"] == {"thinking_level": "high"}
    assert payload["contents"][0]["parts"][1]["text"] == rows[0]["prompt"]
    assert "THIS LABEL" not in json.dumps(payload)
    assert predictions["0"]["response"] == "42"
    evidence = "".join(p.read_text() for p in output.glob("*.json*"))
    assert "private-frontier-key" not in evidence
    assert "image_base64" not in evidence
    assert "THIS LABEL" not in evidence
    events = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["reserved", "completed"]
    assert events[1]["raw_response"] == gemini()


def test_fewshot_uses_training_examples_only_and_rejects_pixel_overlap(rows, tmp_path):
    example = {**rows[1], "split": "train", "response": "17"}
    with patch("vision_rl.frontier._request", return_value=(200, gemini())) as request:
        run_frontier(rows[:1], tmp_path / "fewshot", examples=[example])
    contents = request.call_args.args[1]["contents"]
    assert [message["role"] for message in contents] == ["user", "model", "user"]
    assert contents[1]["parts"] == [{"text": "17"}]
    with pytest.raises(ValueError, match="split='train'"):
        run_frontier(rows[:1], tmp_path / "invalid", examples=[{**example, "split": "test"}])
    with pytest.raises(ValueError, match="overlap"):
        run_frontier(
            rows[:1], tmp_path / "overlap", examples=[{**example, "image": rows[0]["image"]}]
        )


def test_claude_adaptive_preserves_native_usage_and_discards_thinking(rows, tmp_path):
    native = {
        "model": "claude-opus-5-5",
        "id": "msg_123",
        "stop_reason": "end_turn",
        "content": [{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "17"}],
        "usage": {"output_tokens": 100},
    }
    with patch("vision_rl.frontier._request", return_value=(200, native)) as request:
        predictions = run_frontier(rows[:1], tmp_path / "claude", provider="anthropic")
    payload = request.call_args.args[1]
    assert payload["thinking"] == {"type": "adaptive"}
    assert "temperature" not in payload
    assert "output_config" not in payload
    assert predictions["0"]["response"] == "17"
    assert predictions["0"]["usage"] == native["usage"]
    with pytest.raises(ValueError, match="provider-default"):
        run_frontier(rows[:1], tmp_path / "false-high", provider="anthropic", effort="high")


def test_exact_resume_makes_no_duplicate_calls(rows, tmp_path):
    output = tmp_path / "resume"
    with patch("vision_rl.frontier._request", return_value=(200, gemini())) as request:
        first = run_frontier(rows, output, max_requests=3)
        second = run_frontier(rows, output, max_requests=3, resume=True)
    assert request.call_count == 3
    assert first == second
    assert second["3"]["error"] == "not_requested_budget"


@pytest.mark.parametrize("change", ["prompt", "pixels", "budget", "effort", "examples"])
def test_resume_rejects_changed_identity_before_network(rows, tmp_path, change):
    output = tmp_path / "fixed"
    with patch("vision_rl.frontier._request", return_value=(200, gemini())):
        run_frontier(rows[:1], output)
    args = {}
    if change == "prompt":
        rows[0]["prompt"] += " Different question."
    elif change == "pixels":
        Image.new("RGB", (12, 12), "green").save(rows[0]["image"])
    elif change == "budget":
        args["max_requests"] = 31
    elif change == "effort":
        args["effort"] = "low"
    else:
        args["examples"] = [{**rows[1], "split": "train", "response": "3"}]
    with (
        patch("vision_rl.frontier._request") as request,
        pytest.raises(ValueError, match="identity"),
    ):
        run_frontier(rows[:1], output, resume=True, **args)
    request.assert_not_called()


def test_crash_reservation_is_not_retried(rows, tmp_path):
    output = tmp_path / "crashed"
    with (
        patch("vision_rl.frontier._infer", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        run_frontier(rows[:1], output, max_inflight=1)
    events = (output / "attempts.jsonl").read_text().splitlines()
    assert len(events) == 1
    assert json.loads(events[0])["event"] == "reserved"
    with patch("vision_rl.frontier._request") as request:
        result = run_frontier(rows[:1], output, max_inflight=1, resume=True)
    request.assert_not_called()
    assert result["0"]["error"] == "unknown_after_interruption"


@pytest.mark.parametrize("status", [401, 402, 403, 429])
def test_auth_billing_and_quota_stop_current_and_resumed_run(rows, tmp_path, status):
    output = tmp_path / "stopped"
    with patch(
        "vision_rl.frontier._request", return_value=(status, {"error": "unavailable"})
    ) as request:
        result = run_frontier(rows, output, max_inflight=1)
        run_frontier(rows, output, max_inflight=1, resume=True)
    assert request.call_count == 1
    assert result["0"]["error"] == f"HTTP {status}"
    assert result["1"]["error"] == "not_requested_hard_stop"


@pytest.mark.parametrize("provider", ["gemini", "kimi"])
def test_actual_managed_credit_exhaustion_is_not_a_transient_rate_limit(
    rows, tmp_path, provider
):
    output = tmp_path / "credits-exceeded"
    native = {"error": "Roboflow credits exceeded"}
    with patch(
        "vision_rl.frontier._request",
        return_value=(429, native, {"retry_after": None}),
    ) as request:
        run_frontier(
            rows, output, provider=provider, max_inflight=1, max_transient_retries=2
        )
        before = (output / "attempts.jsonl").read_bytes()
        run_frontier(
            rows, output, provider=provider, max_inflight=1,
            max_transient_retries=2, resume=True,
        )
    assert request.call_count == 1
    assert (output / "attempts.jsonl").read_bytes() == before
    summary = json.loads((output / "summary.json").read_text())
    assert summary["hard_stopped"] and summary["stop_reasons"] == ["billing_or_credits"]
    completed = json.loads(before.splitlines()[1])
    assert completed["raw_response"] == native
    assert completed["transport"] == {"retry_after": None}


def test_transient_429_remains_distinct_from_credit_exhaustion(rows, tmp_path):
    output = tmp_path / "rate-limit"
    with patch(
        "vision_rl.frontier._request",
        return_value=(429, {"error": "Too many requests"}, {"retry_after": "60"}),
    ) as request:
        run_frontier(rows, output, max_inflight=1, max_transient_retries=2)
    assert request.call_count == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["stop_reasons"] == ["rate_limit"]


def test_concurrency_never_exceeds_two(rows, tmp_path):
    lock = threading.Lock()
    current = peak = 0

    def transport(*args):
        nonlocal current, peak
        with lock:
            current += 1
            peak = max(peak, current)
        time.sleep(0.02)
        with lock:
            current -= 1
        return 200, gemini()

    with patch("vision_rl.frontier._request", side_effect=transport):
        result = run_frontier(rows, tmp_path / "parallel", max_inflight=2)
    assert peak == 2
    assert all(not prediction.get("error") for prediction in result.values())
    with pytest.raises(ValueError, match="max_inflight"):
        run_frontier(rows, tmp_path / "excess", max_inflight=3)


def test_run_lock_prevents_double_submission(rows, tmp_path):
    started = threading.Event()
    release = threading.Event()
    output = tmp_path / "locked"

    def transport(*args):
        started.set()
        release.wait(timeout=3)
        return 200, gemini()

    with patch("vision_rl.frontier._request", side_effect=transport) as request:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_frontier, rows[:1], output)
            assert started.wait(timeout=3)
            try:
                with pytest.raises(RuntimeError, match="Another process"):
                    run_frontier(rows[:1], output, resume=True)
            finally:
                release.set()
            future.result()
    assert request.call_count == 1


def test_native_truncation_and_model_fallback_are_failures():
    config = {"provider": "gemini", "model": "gemini-4-argon"}
    assert _native(gemini(finish="MAX_TOKENS"), config)["error"]
    assert _native(gemini(model="gemini-2.5-flash"), config)["error"]
    assert "thoughtsTokenCount" not in _native(gemini(), config)["usage"]


def test_http_error_secret_redaction_and_no_retry(rows, tmp_path):
    key = "private-frontier-key"
    error = urllib.error.HTTPError(
        "https://api.roboflow.com/",
        402,
        "Payment",
        {},
        io.BytesIO(json.dumps({"error": key}).encode()),
    )
    with patch("urllib.request.urlopen", side_effect=error):
        status, raw, transport = _request("https://api.roboflow.com/apiproxy/gemini", {}, key, 10)
    assert status == 402
    assert transport["retry_after"] is None
    assert key not in json.dumps(raw)
    with patch("vision_rl.frontier._request", side_effect=RoboflowError(key)) as request:
        predictions = run_frontier(rows[:1], tmp_path / "transport")
    assert request.call_count == 1
    assert predictions["0"]["error"] == "RoboflowError"
    assert key not in (tmp_path / "transport" / "attempts.jsonl").read_text()


def test_reservation_is_durable_before_network(rows, tmp_path):
    output = tmp_path / "durable"

    def transport(*args):
        event = json.loads((output / "attempts.jsonl").read_text())
        assert event["event"] == "reserved"
        assert event["id"] == rows[0]["id"]
        return 200, gemini()

    with patch("vision_rl.frontier._request", side_effect=transport):
        run_frontier(rows[:1], output)


def test_original_jpeg_is_transmitted_without_recompression(rows, tmp_path):
    jpeg = tmp_path / "native.jpg"
    Image.new("RGB", (20, 20), "blue").save(jpeg, quality=81)
    rows[0]["image"] = str(jpeg)
    with patch("vision_rl.frontier._request", return_value=(200, gemini())) as request:
        run_frontier(rows[:1], tmp_path / "native")
    wire = request.call_args.args[1]["contents"][0]["parts"][0]["inline_data"]
    assert wire["mime_type"] == "image/jpeg"
    assert base64.b64decode(wire["data"]) == jpeg.read_bytes()
    identity = json.loads((tmp_path / "native" / "run.json").read_text())["identity"]
    assert "pixel_sha256" in identity["rows"][0]


def test_rate_limit_and_exhausted_credits_have_distinct_stop_reasons(rows, tmp_path):
    with patch("vision_rl.frontier._request", return_value=(429, {}, {"retry_after": "60"})):
        run_frontier(rows[:1], tmp_path / "rate")
    with patch(
        "vision_rl.frontier._request", return_value=(429, {"error": "insufficient credits"})
    ):
        run_frontier(rows[:1], tmp_path / "credits")
    rate = json.loads((tmp_path / "rate" / "summary.json").read_text())
    credits = json.loads((tmp_path / "credits" / "summary.json").read_text())
    assert rate["stop_reasons"] == ["rate_limit"]
    assert credits["stop_reasons"] == ["billing_or_credits"]
    event = json.loads((tmp_path / "rate" / "attempts.jsonl").read_text().splitlines()[1])
    assert event["transport"]["retry_after"] == "60"


def test_journal_corruption_fails_before_any_new_request(rows, tmp_path):
    output = tmp_path / "corrupt"
    with patch("vision_rl.frontier._request", return_value=(200, gemini())):
        run_frontier(rows[:1], output)
    with (output / "attempts.jsonl").open("a") as handle:
        handle.write('{"event":')
    with patch("vision_rl.frontier._request") as request, pytest.raises(ValueError):
        run_frontier(rows[:1], output, resume=True)
    request.assert_not_called()


@pytest.mark.parametrize("status", [502, 503, 504])
def test_opt_in_transport_retries_preserve_all_attempts_then_first_response(rows, tmp_path, status):
    output = tmp_path / "retries"
    with patch(
        "vision_rl.frontier._request",
        side_effect=[(status, {"error": "gateway unavailable"}), (status, {}), (200, gemini("42"))],
    ) as request:
        first = run_frontier(
            rows[:1], output, max_requests=3, max_transient_retries=2, retry_delay_seconds=0
        )
        second = run_frontier(
            rows[:1],
            output,
            max_requests=3,
            max_transient_retries=2,
            retry_delay_seconds=0,
            resume=True,
        )
    assert request.call_count == 3
    assert first == second
    assert first["0"]["response"] == "42"
    assert first["0"]["attempts"] == 3
    assert first["0"]["transport_statuses"] == [status, status, 200]
    events = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    reservations = [e for e in events if e["event"] == "reserved"]
    assert [e["attempt"] for e in reservations] == [1, 2, 3]
    assert [e["retry_of"] for e in reservations] == [None, 1, 2]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["retry_attempts"] == 2
    assert summary["completed_attempts"] == 3
    assert summary["usage_unknown_attempts"] == 2


def test_gateway_errors_are_not_retried_by_default(rows, tmp_path):
    with patch("vision_rl.frontier._request", return_value=(502, {})) as request:
        result = run_frontier(rows[:1], tmp_path / "default")
    assert request.call_count == 1
    assert result["0"]["error"] == "HTTP 502"


@pytest.mark.parametrize("raw", [{}, gemini(finish="MAX_TOKENS"), gemini(text="malformed boxes")])
def test_200_answer_is_never_resampled_even_if_malformed_or_truncated(rows, tmp_path, raw):
    with patch("vision_rl.frontier._request", return_value=(200, raw)) as request:
        result = run_frontier(
            rows[:1], tmp_path / "answer", max_transient_retries=2, retry_delay_seconds=0
        )
    assert request.call_count == 1
    assert result["0"]["attempts"] == 1


@pytest.mark.parametrize("status", [400, 401, 402, 403, 408, 429, 500, 501])
def test_other_http_errors_are_never_retried(rows, tmp_path, status):
    with patch("vision_rl.frontier._request", return_value=(status, {})) as request:
        run_frontier(rows[:1], tmp_path / "other", max_transient_retries=2, retry_delay_seconds=0)
    assert request.call_count == 1


def test_ambiguous_transport_failure_is_never_retried(rows, tmp_path):
    with patch("vision_rl.frontier._request", side_effect=RoboflowError("Timeout")) as request:
        run_frontier(rows[:1], tmp_path / "timeout", max_transient_retries=2, retry_delay_seconds=0)
    assert request.call_count == 1


def test_request_budget_counts_attempts_and_cannot_grow_on_resume(rows, tmp_path):
    output = tmp_path / "attempt-budget"
    with patch("vision_rl.frontier._request", return_value=(502, {})) as request:
        result = run_frontier(
            rows[:1], output, max_requests=2, max_transient_retries=2, retry_delay_seconds=0
        )
        run_frontier(
            rows[:1],
            output,
            max_requests=2,
            max_transient_retries=2,
            retry_delay_seconds=0,
            resume=True,
        )
        with pytest.raises(ValueError, match="identity"):
            run_frontier(
                rows[:1],
                output,
                max_requests=3,
                max_transient_retries=2,
                retry_delay_seconds=0,
                resume=True,
            )
    assert request.call_count == 2
    assert result["0"]["attempts"] == 2
    assert result["0"]["error"] == "HTTP 502"


def test_interrupted_retry_is_not_resampled_on_resume(rows, tmp_path):
    output = tmp_path / "interrupted-retry"
    with (
        patch("vision_rl.frontier._request", side_effect=[(502, {}), KeyboardInterrupt]),
        pytest.raises(KeyboardInterrupt),
    ):
        run_frontier(
            rows[:1], output, max_requests=3, max_transient_retries=2, retry_delay_seconds=0
        )
    with patch("vision_rl.frontier._request") as request:
        result = run_frontier(
            rows[:1],
            output,
            max_requests=3,
            max_transient_retries=2,
            retry_delay_seconds=0,
            resume=True,
        )
    request.assert_not_called()
    assert result["0"]["error"] == "unknown_after_interruption"
    assert result["0"]["attempts"] == 2


def test_schema_one_evidence_resumes_without_upgrading_its_no_retry_policy(rows, tmp_path):
    output = tmp_path / "legacy"
    with patch("vision_rl.frontier._request", return_value=(502, {})):
        run_frontier(rows[:1], output)
    run = json.loads((output / "run.json").read_text())
    run["identity"]["schema_version"] = 1
    for name in ("max_transient_retries", "retry_delay_seconds", "retryable_http_statuses"):
        del run["identity"]["config"][name]
    run["identity_sha256"] = _digest(run["identity"])
    (output / "run.json").write_text(json.dumps(run))
    legacy_events = []
    for line in (output / "attempts.jsonl").read_text().splitlines():
        event = json.loads(line)
        for name in ("item_attempt", "identity_sha256", "retry_of"):
            event.pop(name, None)
        legacy_events.append(event)
    (output / "attempts.jsonl").write_text("".join(json.dumps(e) + "\n" for e in legacy_events))
    with patch("vision_rl.frontier._request") as request:
        result = run_frontier(rows[:1], output, resume=True)
        with pytest.raises(ValueError, match="identity"):
            run_frontier(rows[:1], output, resume=True, max_transient_retries=2)
    request.assert_not_called()
    assert result["0"]["error"] == "HTTP 502"
    assert json.loads((output / "run.json").read_text()) == run


def test_astra_native_responses_request_original_pixels_and_usage(rows, tmp_path):
    response_format = {
        "format": {
            "type": "json_schema",
            "name": "detections",
            "strict": True,
            "schema": {"type": "object", "properties": {}, "additionalProperties": False},
        }
    }
    with patch("vision_rl.frontier._request", return_value=(200, astra())) as request:
        result = run_frontier(
            rows[:1], tmp_path / "astra", provider="openai", response_format=response_format
        )
    url, payload, *_ = request.call_args.args
    assert url.endswith("/apiproxy/openai/v2")
    assert payload["model"] == "gpt-6-astra"
    assert payload["openai_api_key"] == "rf_key:account"
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["max_output_tokens"] == 16384
    assert payload["text"] == response_format
    assert payload["input"][0]["content"][0]["type"] == "input_text"
    image = payload["input"][0]["content"][1]
    assert image["type"] == "input_image" and image["detail"] == "original"
    assert image["image_url"].startswith("data:image/png;base64,")
    assert "temperature" not in payload
    assert result["0"]["provider_model"] == "gpt-6-astra"
    assert result["0"]["provider_response_id"] == "resp_astra"
    assert result["0"]["usage"] == astra()["usage"]


def test_astra_fewshot_messages_preserve_answers_and_native_roles(rows, tmp_path):
    example = {**rows[1], "split": "train", "response": '{"detections":[]}'}
    with patch("vision_rl.frontier._request", return_value=(200, astra())) as request:
        run_frontier(rows[:1], tmp_path / "astra-example", provider="openai", examples=[example])
    messages = request.call_args.args[1]["input"]
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == example["response"]


@pytest.mark.parametrize("status", ["in_progress", "queued", "incomplete", "failed", "cancelled"])
def test_astra_requires_completed_response_and_never_retries_answers(rows, tmp_path, status):
    with patch("vision_rl.frontier._request", return_value=(200, astra(status=status))) as request:
        result = run_frontier(
            rows[:1], tmp_path / "astra-unfinished", provider="openai", max_transient_retries=2
        )
    assert result["0"]["error"] == "Provider did not finish normally"
    assert request.call_count == 1


def test_astra_refusal_and_fallback_models_are_not_valid_answers():
    config = {"provider": "openai", "model": "gpt-6-astra"}
    assert "identity" in _native(astra(model="gpt-6-luna"), config)["error"]
    refused = astra()
    refused["output"][1]["content"] = [{"type": "refusal", "refusal": "Unable"}]
    assert _native(refused, config)["error"] == "Provider returned no answer text"


def test_astra_image_detail_and_schema_are_frozen(rows, tmp_path):
    output = tmp_path / "astra-frozen"
    with patch("vision_rl.frontier._request", return_value=(200, astra())):
        run_frontier(rows[:1], output, provider="openai")
    with patch("vision_rl.frontier._request") as request:
        with pytest.raises(ValueError, match="identity"):
            run_frontier(rows[:1], output, provider="openai", image_detail="high", resume=True)
        with pytest.raises(ValueError, match="identity"):
            run_frontier(
                rows[:1],
                output,
                provider="openai",
                response_format={
                    "format": {
                        "type": "json_schema",
                        "strict": True,
                        "schema": {"type": "object"},
                    }
                },
                resume=True,
            )
    request.assert_not_called()


def test_gemini_native_schema_is_opt_in_and_pins_exact_request_contract(rows, tmp_path):
    schema = {"type": "ARRAY", "items": {"type": "STRING", "enum": ["pill"]}}
    output = tmp_path / "gemini-schema"
    with patch("vision_rl.frontier._request", return_value=(200, gemini('["pill"]'))) as request:
        result = run_frontier(rows[:1], output, response_schema=schema)
    generation = request.call_args.args[1]["generationConfig"]
    assert generation["response_mime_type"] == "application/json"
    assert generation["response_schema"] == schema
    assert generation["thinking_config"] == {"thinking_level": "high"}
    assert result["0"]["response"] == '["pill"]'
    with patch("vision_rl.frontier._request") as request:
        assert run_frontier(rows[:1], output, response_schema=schema, resume=True) == result
        with pytest.raises(ValueError, match="identity"):
            run_frontier(rows[:1], output, resume=True)
        with pytest.raises(ValueError, match="identity"):
            run_frontier(rows[:1], output, response_schema={"type": "OBJECT"}, resume=True)
    request.assert_not_called()


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_gemini_schema_rejects_other_providers_before_sending(rows, tmp_path, provider):
    with patch("vision_rl.frontier._request") as request:
        with pytest.raises(ValueError, match="Gemini-specific"):
            run_frontier(rows[:1], tmp_path / provider, provider=provider, response_schema={})
    request.assert_not_called()


def router_native(provider="qwen", finish="stop", content="[]"):
    return {
        "id": "router-request", "model": (
            "qwen/qwen3.8-27b" if provider == "qwen" else "moonshotai/kimi-k3"
        ),
        "provider": "native-route-name",
        "choices": [{"finish_reason": finish, "message": {
            "content": content, "reasoning": "private reasoning retained only in raw response"
        }}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
                  "completion_tokens_details": {"reasoning_tokens": 20}},
    }


@pytest.mark.parametrize("provider,effort", [("qwen", "xhigh"), ("kimi", "max")])
def test_openrouter_native_payload_usage_and_identity(rows, tmp_path, provider, effort):
    native = router_native(provider)
    example = {**rows[1], "split": "train", "response": "[]"}
    with patch("vision_rl.frontier._request", return_value=(200, native)) as request:
        result = run_frontier(rows[:1], tmp_path / provider, provider=provider, examples=[example])
    url, payload, *_ = request.call_args.args
    assert url.endswith("/apiproxy/openrouter")
    assert payload["openrouter_api_key"] == "rf_key:account"
    assert payload["reasoning"] == {"effort": effort}
    assert payload["privacy_level"] == "deny" and payload["max_tokens"] == 16384
    assert [x["role"] for x in payload["messages"]] == ["user", "assistant", "user"]
    assert payload["messages"][0]["content"][0]["type"] == "image_url"
    assert payload["messages"][0]["content"][1]["type"] == "text"
    assert (payload.get("quantizations") == ["fp8", "bf16", "fp32"]) == (provider == "qwen")
    assert result["0"]["response"] == "[]" and result["0"]["usage"] == native["usage"]
    assert result["0"]["provider_model"] == native["model"]
    assert "temperature" not in payload


@pytest.mark.parametrize("finish", ["length", "content_filter", "error", None])
def test_openrouter_never_retries_truncated_or_failed_native_answers(rows, tmp_path, finish):
    with patch(
        "vision_rl.frontier._request", return_value=(200, router_native(finish=finish))
    ) as request:
        result = run_frontier(
            rows[:1], tmp_path / "router-failed", provider="qwen", max_transient_retries=2
        )
    assert result["0"]["error"] == "Provider did not finish normally"
    assert request.call_count == 1


def test_openrouter_rejected_effort_has_no_silent_fallback(rows, tmp_path):
    with patch(
        "vision_rl.frontier._request", return_value=(400, {"error": "invalid reasoning"})
    ) as request:
        result = run_frontier(
            rows[:1], tmp_path / "router-rejected", provider="qwen", max_transient_retries=2
        )
    assert result["0"]["error"] == "HTTP 400" and request.call_count == 1
    with pytest.raises(ValueError, match="Unsupported effort"):
        run_frontier(rows[:1], tmp_path / "stale-high", provider="qwen", effort="high")
    with pytest.raises(ValueError, match="16384"):
        run_frontier(
            rows[:1], tmp_path / "too-many-tokens", provider="kimi", max_output_tokens=32768
        )


def test_openrouter_route_filters_are_pinned_and_model_aliases_are_not_guessed(rows, tmp_path):
    output = tmp_path / "router-fixed"
    with patch("vision_rl.frontier._request", return_value=(200, router_native())):
        run_frontier(rows[:1], output, provider="qwen")
    with patch("vision_rl.frontier._request") as request:
        for change in ({"privacy_level": "allow"}, {"quantizations": ["bf16"]}):
            with pytest.raises(ValueError, match="identity"):
                run_frontier(rows[:1], output, provider="qwen", resume=True, **change)
    request.assert_not_called()
    native = router_native()
    native["model"] = "qwen/qwen3.8-max"
    assert "identity" in _native(native, {"provider": "qwen", "model": "qwen/qwen3.8-27b"})["error"]
