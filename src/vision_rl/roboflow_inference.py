"""Auditable Gemini inference using Roboflow's managed model proxy.

This is the same ``apiproxy/gemini`` route used by Roboflow's Gemini Workflow
block, with the native provider response retained so model identity and token
usage remain inspectable. Requests consume Roboflow credits. There are no
automatic retries, provider keys, model deployments, or training jobs.

Source: https://github.com/roboflow/inference/blob/main/workflows/
roboflow_workflows/core_steps/models/foundation/google_gemini/v6.py
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .data import load_manifest, write_json
from .evaluation import prompt_text, score
from .roboflow import RoboflowError, _api_key

GEMINI_PROXY_URL = "https://api.roboflow.com/apiproxy/gemini"
DEFAULT_MODEL = "gemini-4-argon"


def _configuration(model: str, thinking_level: str, max_tokens: int) -> dict:
    if not isinstance(model, str) or not re.fullmatch(r"gemini-[A-Za-z0-9.-]+", model):
        raise ValueError("model must be a Gemini model ID")
    if thinking_level not in ("low", "medium", "high", "minimal"):
        raise ValueError("thinking_level must be low, medium, high, or minimal")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    return {
        "response_mime_type": "text/plain",
        "candidate_count": 1,
        "max_output_tokens": max_tokens,
        "thinking_config": {"thinking_level": thinking_level},
    }


def _native_prediction(payload: Any) -> dict:
    """Retain provider identity and reasoning usage, without scoring partial output."""
    if not isinstance(payload, dict):
        raise ValueError("unexpected provider response")
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("expected exactly one provider candidate")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise ValueError("unexpected provider candidate")
    finish_reason = candidate.get("finishReason")
    usage = payload.get("usageMetadata", {})
    if not isinstance(usage, dict):
        raise ValueError("unexpected provider usage")
    result = {
        "response": "",
        "provider_model": payload.get("modelVersion"),
        "provider_response_id": payload.get("responseId"),
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount"),
            "completion_tokens": usage.get("candidatesTokenCount"),
            "reasoning_tokens": usage.get("thoughtsTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount"),
        },
    }
    if finish_reason != "STOP":
        result["error"] = "Provider did not finish normally"
        return result
    parts = candidate.get("content", {}).get("parts", [])
    if not isinstance(parts, list):
        raise ValueError("unexpected provider content")
    texts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and not part.get("thought") and isinstance(part.get("text"), str)
    ]
    if not texts:
        raise ValueError("provider returned no answer text")
    result["response"] = "".join(texts)
    return result


def _request(payload: dict, key: str) -> tuple[int, Any]:
    """Use urllib so HTTP debug logging cannot print a credential-bearing URL."""
    url = GEMINI_PROXY_URL + "?" + urllib.parse.urlencode({"api_key": key})
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read()
        exc.close()
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RoboflowError("Cannot reach Roboflow Gemini proxy") from None
    # API errors can echo request metadata. Never persist the known credential.
    text = body.decode("utf-8", errors="replace").replace(key, "[REDACTED]")
    try:
        return status, json.loads(text)
    except ValueError:
        return status, {"unparsed_response": text}


def predict_roboflow_gemini(
    rows: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    thinking_level: str = "low",
    max_tokens: int = 4096,
    max_requests: int = 30,
    raw_output: Path | None = None,
) -> dict[str, dict]:
    """Make one request per example; retain all failures in the denominator.

    ``raw_output`` is append-only JSONL created exclusively, flushed after every
    request. The answer label is never sent to the model. Provider sampling
    defaults are used; reasoning models may not support temperature control.
    """
    config = _configuration(model, thinking_level, max_tokens)
    if (
        isinstance(max_requests, bool)
        or not isinstance(max_requests, int)
        or max_requests < 1
        or len(rows) > max_requests
    ):
        raise ValueError("Examples exceed the request budget, or budget is invalid")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate example IDs")
    key = _api_key()
    handle = None
    if raw_output is not None:
        raw_output = Path(raw_output)
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        handle = raw_output.open("x", encoding="utf-8")
    predictions = {}
    try:
        for row in rows:
            # Lossless PNG encoding presents identical decoded pixels for every backend.
            with Image.open(row["image"]) as source:
                stream = io.BytesIO()
                source.convert("RGB").save(stream, format="PNG")
            image_bytes = stream.getvalue()
            prompt = prompt_text(row)
            payload = {
                "model": model,
                "google_api_key": "rf_key:account",
                "contents": {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(image_bytes).decode(),
                            }
                        },
                        {"text": prompt},
                    ],
                },
                "generationConfig": config,
            }
            record = {
                "id": row["id"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "requested_model": model,
                "generation_config": config,
                "prompt": prompt,
                "transmitted_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            }
            start = time.monotonic()
            try:
                status, raw = _request(payload, key)
                record.update(status_code=status, raw_response=raw)
                if status != 200:
                    prediction = {"response": "", "error": f"HTTP {status}"}
                else:
                    prediction = _native_prediction(raw)
            except (
                RoboflowError,
                ValueError,
                KeyError,
                IndexError,
                TypeError,
                AttributeError,
            ) as exc:
                # Exception messages may contain provider data or credential-bearing URLs.
                prediction = {"response": "", "error": type(exc).__name__}
            prediction["latency_seconds"] = time.monotonic() - start
            predictions[row["id"]] = prediction
            if handle is not None:
                handle.write(json.dumps({**record, "prediction": prediction}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        if handle is not None:
            handle.close()
    return predictions


def run_baseline(
    manifest: Path,
    output: Path,
    *,
    model: str = DEFAULT_MODEL,
    thinking_level: str = "low",
    max_tokens: int = 4096,
    max_requests: int = 30,
) -> dict:
    """Write score, prediction JSONL and native responses in a fresh directory."""
    rows = load_manifest(manifest)
    _configuration(model, thinking_level, max_tokens)
    if len(rows) > max_requests:
        raise ValueError("Examples exceed the request budget")
    _api_key()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    predictions = predict_roboflow_gemini(
        rows,
        model=model,
        thinking_level=thinking_level,
        max_tokens=max_tokens,
        max_requests=max_requests,
        raw_output=output / "raw-responses.jsonl",
    )
    provider_models = sorted(
        {p["provider_model"] for p in predictions.values() if p.get("provider_model")}
    )
    report = score(
        rows,
        predictions,
        {
            "name": model,
            "backend": "roboflow-gemini-proxy",
            "provider_reported_models": provider_models,
            "thinking_level": thinking_level,
            "max_tokens": max_tokens,
            "max_requests": max_requests,
            "temperature": "provider-default",
            "billing": "Roboflow credits; invoice amount not available from this API",
        },
    )
    (output / "predictions.jsonl").write_text(
        "".join(
            json.dumps({"id": identifier, **p}) + "\n" for identifier, p in predictions.items()
        ),
        encoding="utf-8",
    )
    write_json(output / "evaluation.json", report)
    return report


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking-level", default="low")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-requests", type=int, default=30)
    args = parser.parse_args()
    try:
        report = run_baseline(
            args.manifest,
            args.output,
            model=args.model,
            thinking_level=args.thinking_level,
            max_tokens=args.max_tokens,
            max_requests=args.max_requests,
        )
    except (ValueError, OSError, RoboflowError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps({k: report[k] for k in ("model", "n", "correct", "accuracy", "errors")}))


if __name__ == "__main__":
    main()
