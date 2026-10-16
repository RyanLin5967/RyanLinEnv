import io
import json
import urllib.error
from unittest.mock import patch

import pytest
from PIL import Image

from vision_rl.evaluation import prompt_text, score
from vision_rl.roboflow_inference import _native_prediction, _request, predict_roboflow_gemini


def sample(tmp_path):
    image = tmp_path / "one.png"
    Image.new("RGB", (12, 12), "red").save(image)
    return {
        "id": "example",
        "image": str(image),
        "image_sha256": "test-image-hash",
        "prompt": "Count red squares.",
        "choices": ["0", "1"],
        "answer": "B",
        "category": "counting",
        "split": "test",
    }


def response(text="<answer>B</answer>", finish="STOP"):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 10,
            "thoughtsTokenCount": 20,
            "totalTokenCount": 130,
        },
        "modelVersion": "gemini-4-argon-provider-revision",
        "responseId": "provider-request-id",
    }


def test_native_identity_usage_and_answer_are_preserved():
    result = _native_prediction(response())
    assert result["provider_model"] == "gemini-4-argon-provider-revision"
    assert result["provider_response_id"] == "provider-request-id"
    assert result["usage"]["reasoning_tokens"] == 20
    assert result["response"] == "<answer>B</answer>"


def test_truncated_response_is_not_accepted_as_correct(tmp_path):
    result = _native_prediction(response(finish="MAX_TOKENS"))
    report = score([sample(tmp_path)], {"example": result}, {})
    assert report["accuracy"] == 0
    assert report["errors"] == 1


def test_model_thought_parts_are_not_answer_text():
    payload = response()
    payload["candidates"][0]["content"]["parts"].insert(
        0, {"thought": True, "text": "Private reasoning is separate."}
    )
    assert _native_prediction(payload)["response"] == "<answer>B</answer>"


def test_exact_prompt_single_request_and_native_audit_log(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "secret-value")
    row = sample(tmp_path)
    raw = tmp_path / "raw.jsonl"
    with patch("vision_rl.roboflow_inference._request", return_value=(200, response())) as call:
        predictions = predict_roboflow_gemini([row], raw_output=raw)
    assert call.call_count == 1
    sent = call.call_args.args[0]
    assert sent["contents"]["parts"][1] == {"text": prompt_text(row)}
    assert "answer" not in sent
    assert sent["generationConfig"]["candidate_count"] == 1
    assert sent["contents"]["parts"][0]["inline_data"]["mime_type"] == "image/png"
    record = json.loads(raw.read_text())
    assert record["raw_response"] == response()
    assert record["prediction"] == predictions[row["id"]]
    assert record["prediction"]["latency_seconds"] >= 0
    assert "secret-value" not in raw.read_text()


def test_budget_and_existing_log_rejected_before_requests(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "secret")
    row = sample(tmp_path)
    with patch("vision_rl.roboflow_inference._request") as call:
        with pytest.raises(ValueError, match="budget"):
            predict_roboflow_gemini([row], max_requests=0)
        with pytest.raises(ValueError, match="Duplicate"):
            predict_roboflow_gemini([row, row])
        log = tmp_path / "existing.jsonl"
        log.write_text("previous evidence")
        with pytest.raises(FileExistsError):
            predict_roboflow_gemini([row], raw_output=log)
    assert call.call_count == 0
    assert log.read_text() == "previous evidence"


def test_http_errors_keep_raw_response_but_redact_key():
    key = "private-key"
    error = urllib.error.HTTPError(
        "https://api.roboflow.com/apiproxy/gemini?api_key=" + key,
        403,
        key,
        {},
        io.BytesIO(json.dumps({"message": "api_key=" + key}).encode()),
    )
    with patch("urllib.request.urlopen", side_effect=error):
        status, payload = _request({}, key)
    assert status == 403
    assert key not in json.dumps(payload)
    assert payload["message"] == "api_key=[REDACTED]"


def test_http_failure_counts_once_and_is_not_retried(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "secret")
    row = sample(tmp_path)
    with patch("vision_rl.roboflow_inference._request", return_value=(402, {})) as call:
        predictions = predict_roboflow_gemini([row])
    report = score([row], predictions, {})
    assert call.call_count == 1
    assert report["n"] == 1
    assert report["errors"] == 1
    assert report["correct"] == 0


@pytest.mark.parametrize("payload", [{}, [], {"candidates": [None]}])
def test_invalid_provider_responses_do_not_guess(payload):
    with pytest.raises(ValueError):
        _native_prediction(payload)
