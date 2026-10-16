"""Resumable, bounded native VLM requests through Roboflow managed credits.

This module performs inference only: it never changes a billing plan, purchases
credits, or interprets a benchmark answer. Optional transport retries apply only
to completed HTTP 502/503/504 responses and preserve every attempt. A durable reservation
precedes every network call. A reservation without a completion is *unknown*, not
safe to retry: the provider may already have charged for or completed that call.

Wire schemas follow Roboflow inference commit
150577969c6dbe5c0da854942c5961fe3f293088, Gemini v6, Anthropic v5 and OpenAI v7 blocks.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .roboflow import RoboflowError, _api_key

SOURCE_COMMIT = "150577969c6dbe5c0da854942c5961fe3f293088"
PROVIDERS = {
    "gemini": ("gemini-4-argon", "https://api.roboflow.com/apiproxy/gemini"),
    "anthropic": ("claude-opus-5-5", "https://api.roboflow.com/apiproxy/anthropic"),
    "openai": ("gpt-6-astra", "https://api.roboflow.com/apiproxy/openai/v2"),
    "qwen": ("qwen/qwen3.8-27b", "https://api.roboflow.com/apiproxy/openrouter"),
    "kimi": ("moonshotai/kimi-k3", "https://api.roboflow.com/apiproxy/openrouter"),
}
OPENROUTER_PROVIDERS = frozenset({"qwen", "kimi"})
DEFAULT_EFFORTS = {
    "gemini": "high", "anthropic": "adaptive", "openai": "high",
    "qwen": "xhigh", "kimi": "max",
}
SUPPORTED_EFFORTS = {
    "gemini": ("low", "high"), "anthropic": ("adaptive",),
    "openai": ("low", "medium", "high", "xhigh", "max"),
    "qwen": ("none", "low", "medium", "xhigh"), "kimi": ("low", "high", "max"),
}
MAX_RESPONSE_BYTES = 16 * 1024**2
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_BYTES = 20 * 1024**2
RETRYABLE_HTTP_STATUSES = frozenset({502, 503, 504})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _encoded_image(path: str) -> tuple[bytes, dict]:
    with Image.open(path) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError(f"Images must contain at most {MAX_IMAGE_PIXELS} pixels")
        rgb = source.convert("RGB")
        pixels = hashlib.sha256(
            f"RGB:{rgb.width}:{rgb.height}:".encode() + rgb.tobytes()
        ).hexdigest()
        # Preserve original JPEG bytes without recompression or PNG expansion.
        # Normalize unusual color modes/orientation to the RGB pixels used locally.
        if source.format == "JPEG" and source.mode == "RGB" and source.getexif().get(274, 1) == 1:
            data, mime = Path(path).read_bytes(), "image/jpeg"
        else:
            stream = io.BytesIO()
            rgb.save(stream, format="PNG")
            data, mime = stream.getvalue(), "image/png"
        metadata = {
            "pixel_sha256": pixels,
            "image_media_type": mime,
            "width": rgb.width,
            "height": rgb.height,
            "image_bytes": len(data),
        }
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Encoded images must contain at most {MAX_IMAGE_BYTES} bytes")
    return data, metadata


def _prepare(row: dict, *, example: bool = False) -> dict:
    for key in ("id", "prompt"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Every row needs a nonempty {key}")
    data, metadata = _encoded_image(row["image"])
    result = {
        "id": row["id"],
        "prompt": row["prompt"],
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "image_path": str(row["image"]),
        **metadata,
    }
    if example:
        if row.get("split") != "train":
            raise ValueError("Few-shot examples must explicitly have split='train'")
        if not isinstance(row.get("response"), str) or not row["response"].strip():
            raise ValueError("Few-shot examples need an explicit response string")
        result["response"] = row["response"]
    return result


def _public(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "image_path"}


def _image_data(row: dict) -> str:
    # Encode lazily within the bounded worker pool, never preload an entire benchmark.
    data, _ = _encoded_image(row["image_path"])
    if hashlib.sha256(data).hexdigest() != row["image_sha256"]:
        raise ValueError("Image pixels changed after run identity was recorded")
    return base64.b64encode(data).decode()


def _payload(row: dict, examples: list[dict], config: dict) -> dict:
    if config["provider"] in OPENROUTER_PROVIDERS:
        messages = []
        for item in [*examples, row]:
            messages.append({
                "role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:{item['image_media_type']};base64,{_image_data(item)}"
                    }},
                    {"type": "text", "text": item["prompt"]},
                ],
            })
            if "response" in item:
                messages.append({"role": "assistant", "content": item["response"]})
        payload = {
            "model": config["model"], "openrouter_api_key": "rf_key:account",
            "messages": messages, "max_tokens": config["max_output_tokens"],
            "privacy_level": config["privacy_level"], "reasoning": {"effort": config["effort"]},
        }
        if config["quantizations"] is not None:
            payload["quantizations"] = config["quantizations"]
        return payload
    if config["provider"] == "gemini":
        contents = []
        for item in [*examples, row]:
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": item["image_media_type"],
                                "data": _image_data(item),
                            }
                        },
                        {"text": item["prompt"]},
                    ],
                }
            )
            if "response" in item:
                contents.append({"role": "model", "parts": [{"text": item["response"]}]})
        generation = {
            "response_mime_type": "text/plain",
            "candidate_count": 1,
            "max_output_tokens": config["max_output_tokens"],
            "thinking_config": {"thinking_level": config["effort"]},
        }
        if config.get("response_schema") is not None:
            generation.update(
                response_mime_type="application/json", response_schema=config["response_schema"]
            )
        return {
            "model": config["model"],
            "google_api_key": "rf_key:account",
            "contents": contents,
            "generationConfig": generation,
        }
    if config["provider"] == "openai":
        messages = []
        for item in [*examples, row]:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": item["prompt"]},
                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:{item['image_media_type']};base64,{_image_data(item)}"
                            ),
                            "detail": config["image_detail"],
                        },
                    ],
                }
            )
            if "response" in item:
                messages.append({"role": "assistant", "content": item["response"]})
        payload = {
            "model": config["model"],
            "openai_api_key": "rf_key:account",
            "input": messages,
            "reasoning": {"effort": config["effort"]},
            "max_output_tokens": config["max_output_tokens"],
        }
        if config.get("response_format") is not None:
            payload["text"] = config["response_format"]
        return payload
    messages = []
    for item in [*examples, row]:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item["image_media_type"],
                            "data": _image_data(item),
                        },
                    },
                    {"type": "text", "text": item["prompt"]},
                ],
            }
        )
        if "response" in item:
            messages.append({"role": "assistant", "content": item["response"]})
    return {
        "model": config["model"],
        "anthropic_api_key": "rf_key:account",
        "messages": messages,
        "max_tokens": config["max_output_tokens"],
        "thinking": {"type": "adaptive"},
    }


def _request(url: str, payload: dict, key: str, timeout: float) -> tuple[int, Any, dict]:
    # Do not log URLs: the public Roboflow API uses query authentication.
    authenticated_url = url + "?" + urllib.parse.urlencode({"api_key": key})
    request = urllib.request.Request(
        authenticated_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, body = response.status, response.read(MAX_RESPONSE_BYTES + 1)
            retry_after = response.headers.get("Retry-After")
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read(MAX_RESPONSE_BYTES + 1)
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        exc.close()
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RoboflowError("Frontier request failed; delivery/billing may be unknown") from None
    if len(body) > MAX_RESPONSE_BYTES:
        raise RoboflowError("Provider response exceeds the size limit")
    sanitized = body.decode("utf-8", errors="replace").replace(key, "[REDACTED]")
    try:
        return status, json.loads(sanitized), {"retry_after": retry_after}
    except ValueError:
        return status, {"unparsed_response": sanitized}, {"retry_after": retry_after}


def _native(raw: Any, config: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Provider response must be an object")
    if config["provider"] == "gemini":
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError("Expected one provider candidate")
        candidate = candidates[0]
        finish = candidate.get("finishReason")
        texts = [
            part["text"]
            for part in candidate.get("content", {}).get("parts", [])
            if isinstance(part, dict)
            and not part.get("thought")
            and isinstance(part.get("text"), str)
        ]
        model, response_id = raw.get("modelVersion"), raw.get("responseId")
        usage = raw.get("usageMetadata", {})
        finished = finish == "STOP"
    elif config["provider"] == "openai":
        finish = raw.get("status")
        output = raw.get("output", [])
        if not isinstance(output, list):
            raise ValueError("Provider output must be an array")
        texts = [
            part["text"]
            for item in output
            if isinstance(item, dict) and item.get("type") == "message"
            for part in item.get("content", [])
            if isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ]
        model, response_id = raw.get("model"), raw.get("id")
        usage = raw.get("usage", {})
        finished = finish == "completed"
    elif config["provider"] in OPENROUTER_PROVIDERS:
        choices = raw.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Expected one native OpenRouter choice")
        choice = choices[0]
        finish = choice.get("finish_reason")
        content = choice.get("message", {}).get("content")
        texts = [content] if isinstance(content, str) else []
        model, response_id = raw.get("model"), raw.get("id")
        usage = raw.get("usage", {})
        finished = finish == "stop"
    else:
        finish = raw.get("stop_reason")
        texts = [
            part["text"]
            for part in raw.get("content", [])
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ]
        model, response_id = raw.get("model"), raw.get("id")
        usage = raw.get("usage", {})
        finished = finish == "end_turn"
    result = {
        "response": "".join(texts),
        "provider_model": model,
        "provider_response_id": response_id,
        "finish_reason": finish,
        # Preserve the native dictionary: absent reasoning usage is unknown, never zero.
        "usage": usage,
    }
    if not finished:
        result["error"] = "Provider did not finish normally"
    elif not isinstance(model, str) or not (
        model == config["model"] or model.startswith(config["model"] + "-")
    ):
        result["error"] = "Provider model identity missing or inconsistent with requested model"
    elif not result["response"].strip():
        result["error"] = "Provider returned no answer text"
    return result


def _stop_reason(status: int, raw: Any) -> str | None:
    if status in (401, 403):
        return "authentication_or_access"
    if status == 402:
        return "billing_or_credits"
    if status < 400:
        return None
    text = json.dumps(raw).lower()
    if any(
        term in text
        for term in (
            "insufficient_quota",
            "quota_exceeded",
            "quota exhausted",
            "quota limit exceeded",
        )
    ):
        return "quota_exhausted"
    if any(
        term in text
        for term in (
            "insufficient credits",
            "billing",
            "payment required",
            "credits exhausted",
            "credits exceeded",
            "credit limit exceeded",
            "out of credits",
            "not enough credits",
        )
    ):
        return "billing_or_credits"
    if status == 429:
        return "rate_limit"
    return None


def _infer(
    row: dict, examples: list[dict], config: dict, key: str, retry_delay_seconds: float = 0
) -> dict:
    if retry_delay_seconds:
        time.sleep(retry_delay_seconds)
    start = time.monotonic()
    result: dict = {"hard_stop": False, "retry_delay_seconds": retry_delay_seconds}
    try:
        response = _request(
            PROVIDERS[config["provider"]][1],
            _payload(row, examples, config),
            key,
            config["timeout_seconds"],
        )
        status, raw = response[:2]
        transport = response[2] if len(response) > 2 else {}
        # Redact again at the journal boundary, including for injected/custom transports.
        raw = json.loads(json.dumps(raw).replace(key, "[REDACTED]"))
        transport = json.loads(json.dumps(transport).replace(key, "[REDACTED]"))
        stop_reason = _stop_reason(status, raw)
        result.update(
            status_code=status,
            raw_response=raw,
            transport=transport,
            hard_stop=stop_reason is not None,
            stop_reason=stop_reason,
        )
        if status == 200:
            prediction = _native(raw, config)
        else:
            prediction = {"response": "", "error": f"HTTP {status}"}
    except (RoboflowError, ValueError, KeyError, TypeError, AttributeError) as exc:
        prediction = {"response": "", "error": type(exc).__name__}
    prediction["latency_seconds"] = time.monotonic() - start
    result["prediction"] = prediction
    return result


def _append(handle, event: dict) -> None:
    handle.write(json.dumps(event, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run_frontier(
    rows: list[dict],
    output: Path,
    *,
    provider: str = "gemini",
    model: str | None = None,
    effort: str | None = None,
    max_output_tokens: int = 16384,
    max_requests: int = 30,
    examples: list[dict] | tuple = (),
    resume: bool = False,
    max_inflight: int = 2,
    timeout_seconds: float = 240,
    max_transient_retries: int = 0,
    retry_delay_seconds: float = 2,
    image_detail: str | None = None,
    response_format: dict | None = None,
    response_schema: dict | None = None,
    privacy_level: str | None = None,
    quantizations: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """Infer with a crash-safe journal and a fixed, bounded transport policy.

    Input rows require ``id``, ``image``, and an already-complete ``prompt``.
    Few-shot examples also require ``split='train'`` and explicit ``response``.
    Only these fields are sent; benchmark labels and other metadata are ignored.
    Gemini defaults to high thinking; Claude uses adaptive/provider-default effort
    because the documented Roboflow proxy does not expose an effort override.
    Astra uses high reasoning and original image detail by default. Its optional
    ``response_format`` is the native Responses ``text`` configuration, typically
    a strict JSON schema. Gemini optionally accepts its native ``response_schema``
    in generationConfig. These settings are pinned in the journal identity.
    Qwen and Kimi use managed OpenRouter, with xhigh/max requested respectively.
    Routing privacy defaults to deny. Qwen requests fp8/bf16/fp32 endpoints;
    Kimi uses native routing without a precision allowlist. Returned route/model
    and usage remain in the raw evidence; no unsupported-effort fallback occurs.

    ``max_requests`` bounds total reserved attempts across resumes, not per invocation.
    Retries are disabled by default. ``max_transient_retries`` permits up to two
    additional attempts only after completed HTTP 502/503/504 errors. The first
    HTTP 200 response is final, including malformed/truncated answers; no answer
    selection occurs. Connection failures and interrupted reservations are never
    retried. Failed gateway requests may still incur charges; usage is unknown.
    Authentication, billing and quota errors stop new scheduling; up to two already
    reserved requests may be in flight. Native replies, prompts, settings, encoded
    image hashes and decoded RGB hashes are retained. Credentials and image bytes
    are excluded from artifacts. Native RGB JPEGs are sent without recompression;
    other images use PNG. No image is resized implicitly.
    """
    if provider not in PROVIDERS:
        raise ValueError("Unsupported frontier provider")
    model = model or PROVIDERS[provider][0]
    if model != PROVIDERS[provider][0]:
        raise ValueError("Only the verified current frontier model for this provider is supported")
    effort = effort or DEFAULT_EFFORTS[provider]
    if effort not in SUPPORTED_EFFORTS[provider]:
        raise ValueError(
            "Unsupported effort; Claude uses adaptive/provider-default. "
            f"Allowed for {provider}: {SUPPORTED_EFFORTS[provider]}"
        )
    if provider in OPENROUTER_PROVIDERS:
        privacy_level = privacy_level or "deny"
        if privacy_level not in {"allow", "deny", "zdr"}:
            raise ValueError("Unsupported OpenRouter privacy_level")
        if quantizations is None and provider == "qwen":
            quantizations = ["fp8", "bf16", "fp32"]
        if quantizations is not None:
            if not isinstance(quantizations, (list, tuple)) or not quantizations or any(
                q not in {"fp4", "mxfp4", "fp8", "fp16", "bf16", "fp32"} for q in quantizations
            ):
                raise ValueError("quantizations must name explicit supported precision labels")
            quantizations = list(quantizations)
        if max_output_tokens > 16384:
            raise ValueError("Managed OpenRouter max_output_tokens cannot exceed 16384")
    elif privacy_level is not None or quantizations is not None:
        raise ValueError("privacy_level and quantizations are OpenRouter-specific parameters")
    if provider != "openai" and (image_detail is not None or response_format is not None):
        raise ValueError("image_detail and response_format are OpenAI-specific parameters")
    if response_schema is not None:
        if provider != "gemini":
            raise ValueError("response_schema is a Gemini-specific parameter")
        serialized_schema = json.dumps(response_schema, allow_nan=False)
        if len(serialized_schema) > 64 * 1024:
            raise ValueError("response_schema exceeds 64 KiB")
        response_schema = json.loads(serialized_schema)
        if not isinstance(response_schema, dict) or not isinstance(
            response_schema.get("type"), str
        ):
            raise ValueError("response_schema must be a native Gemini schema object with type")
    if provider == "openai":
        image_detail = image_detail or "original"
        if image_detail not in ("low", "high", "original", "auto"):
            raise ValueError("Unsupported OpenAI image_detail")
        if response_format is not None:
            serialized_format = json.dumps(response_format, allow_nan=False)
            if len(serialized_format) > 64 * 1024:
                raise ValueError("response_format exceeds 64 KiB")
            response_format = json.loads(serialized_format)
            fmt = response_format.get("format") if isinstance(response_format, dict) else None
            if (
                not isinstance(fmt, dict)
                or fmt.get("type") != "json_schema"
                or fmt.get("strict") is not True
                or not isinstance(fmt.get("schema"), dict)
            ):
                raise ValueError("response_format must contain a strict native JSON schema format")
    for name, value, lower, upper in (
        ("max_requests", max_requests, 1, 100000),
        ("max_inflight", max_inflight, 1, 2),
        ("max_output_tokens", max_output_tokens, 16, 128000),
        ("max_transient_retries", max_transient_retries, 0, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    if not 1 <= timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be between 1 and 600")
    if (
        isinstance(retry_delay_seconds, bool)
        or not isinstance(retry_delay_seconds, (int, float))
        or not 0 <= retry_delay_seconds <= 60
    ):
        raise ValueError("retry_delay_seconds must be between 0 and 60")
    prepared = [_prepare(row) for row in rows]
    fewshot = [_prepare(row, example=True) for row in examples]
    if len({row["id"] for row in prepared}) != len(prepared):
        raise ValueError("Duplicate request IDs")
    request_ids = {row["id"] for row in prepared}
    request_hashes = {row["pixel_sha256"] for row in prepared}
    if any(row["id"] in request_ids or row["pixel_sha256"] in request_hashes for row in fewshot):
        raise ValueError("Few-shot training examples overlap evaluation images")
    if provider == "anthropic" and any(
        row["image_bytes"] > 5 * 1024**2 for row in [*prepared, *fewshot]
    ):
        raise ValueError(
            "Anthropic input exceeds 5 MiB per image; explicit preprocessing is required"
        )
    config = {
        "provider": provider,
        "model": model,
        "effort": effort,
        "max_output_tokens": max_output_tokens,
        "max_requests": max_requests,
        "max_inflight": max_inflight,
        "timeout_seconds": timeout_seconds,
        "temperature": "provider-default",
        "source_commit": SOURCE_COMMIT,
        "anthropic_effort": "provider-default" if provider == "anthropic" else None,
        "max_transient_retries": max_transient_retries,
        "retry_delay_seconds": retry_delay_seconds,
        "retryable_http_statuses": sorted(RETRYABLE_HTTP_STATUSES),
    }
    if provider == "openai":
        config.update(image_detail=image_detail, response_format=response_format)
    if response_schema is not None:
        config["response_schema"] = response_schema
    if provider in OPENROUTER_PROVIDERS:
        config.update(privacy_level=privacy_level, quantizations=quantizations)
    identity = {
        "schema_version": 2,
        "config": config,
        "rows": [_public(row) for row in prepared],
        "examples": [_public(row) for row in fewshot],
    }
    output = Path(output)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("Cannot resume a missing run")
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process owns this frontier run") from None
        run_path = output / "run.json"
        if resume:
            stored = json.loads(run_path.read_text())
            # Legacy no-retry runs remain resumable without rewriting their evidence.
            # A frozen schema-1 run can never be upgraded to a retry policy in place.
            if stored.get("identity", {}).get("schema_version") == 1 and max_transient_retries == 0:
                identity["schema_version"] = 1
                for name in (
                    "max_transient_retries",
                    "retry_delay_seconds",
                    "retryable_http_statuses",
                ):
                    config.pop(name)
            if stored.get("identity") != identity or stored.get("identity_sha256") != _digest(
                identity
            ):
                raise ValueError(
                    "Resume identity differs: model, settings, prompts or image pixels changed"
                )
        else:
            with run_path.open("x") as handle:
                json.dump(
                    {
                        "created_at": _now(),
                        "identity": identity,
                        "identity_sha256": _digest(identity),
                    },
                    handle,
                    indent=2,
                )
                handle.flush()
                os.fsync(handle.fileno())
        return _run_locked(prepared, fewshot, config, output, _digest(identity))


def _retry_allowed(completion: dict, attempts_so_far: int, config: dict) -> bool:
    return (
        completion.get("status_code") in RETRYABLE_HTTP_STATUSES
        and not completion.get("hard_stop")
        and attempts_so_far <= config.get("max_transient_retries", 0)
    )


def _run_locked(
    prepared: list[dict], fewshot: list[dict], config: dict, output: Path, identity_hash: str
) -> dict:
    journal = output / "attempts.jsonl"
    reserved: dict[int, dict] = {}
    completed: dict[int, dict] = {}
    item_attempts: dict[str, list[int]] = {}
    stopped = False
    stop_reasons = set()
    public_rows = {row["id"]: _public(row) for row in prepared}
    if journal.exists():
        for line in journal.read_text().splitlines():
            event = json.loads(line)
            item_id, attempt = event["id"], event["attempt"]
            if item_id not in public_rows:
                raise ValueError("Journal contains an unknown example")
            if "identity_sha256" in event and event["identity_sha256"] != identity_hash:
                raise ValueError("Journal belongs to a different run identity")
            if event["event"] == "reserved":
                previous = item_attempts.get(item_id, [])
                if attempt != len(reserved) + 1:
                    raise ValueError("Journal has duplicate or out-of-order reservations")
                if event.get("request") != public_rows[item_id]:
                    raise ValueError("Journal reservation differs from the fixed input identity")
                if event.get("item_attempt", 1) != len(previous) + 1:
                    raise ValueError("Journal item attempt sequence is invalid")
                if previous and (
                    previous[-1] not in completed
                    or not _retry_allowed(completed[previous[-1]], len(previous), config)
                ):
                    raise ValueError("Journal retry violates the frozen transport policy")
                reserved[attempt] = event
                item_attempts.setdefault(item_id, []).append(attempt)
            elif event["event"] == "completed":
                if (
                    attempt not in reserved
                    or attempt in completed
                    or reserved[attempt]["id"] != item_id
                ):
                    raise ValueError("Journal completion lacks a unique reservation")
                completed[attempt] = event
                stopped |= event.get("hard_stop", False)
                if event.get("stop_reason"):
                    stop_reasons.add(event["stop_reason"])
            else:
                raise ValueError("Unknown journal event")
    if len(reserved) > config["max_requests"]:
        raise ValueError("Journal exceeds the original request budget")

    # A missing completion is permanently ambiguous. Only an explicit HTTP error
    # can make an item eligible for another attempt; no score/answer is examined.
    pending = deque()
    for row in prepared:
        previous = item_attempts.get(row["id"], [])
        if not previous or (
            previous[-1] in completed
            and _retry_allowed(completed[previous[-1]], len(previous), config)
        ):
            pending.append(row)
    key = _api_key() if pending and not stopped and len(reserved) < config["max_requests"] else ""
    with (
        journal.open("a") as handle,
        ThreadPoolExecutor(max_workers=config["max_inflight"]) as pool,
    ):
        active = {}
        while True:
            while not stopped and pending and len(active) < config["max_inflight"]:
                if len(reserved) >= config["max_requests"]:
                    break
                row = pending.popleft()
                previous = item_attempts.get(row["id"], [])
                reservation = {
                    "event": "reserved",
                    "id": row["id"],
                    "attempt": len(reserved) + 1,
                    "item_attempt": len(previous) + 1,
                    "retry_of": previous[-1] if previous else None,
                    "identity_sha256": identity_hash,
                    "created_at": _now(),
                    "request": _public(row),
                }
                _append(handle, reservation)
                reserved[reservation["attempt"]] = reservation
                item_attempts.setdefault(row["id"], []).append(reservation["attempt"])
                delay = config.get("retry_delay_seconds", 2) if previous else 0
                active[pool.submit(_infer, row, fewshot, config, key, delay)] = (reservation, row)
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                reservation, row = active.pop(future)
                result = future.result()
                event = {
                    "event": "completed",
                    "id": reservation["id"],
                    "attempt": reservation["attempt"],
                    "item_attempt": reservation["item_attempt"],
                    "identity_sha256": identity_hash,
                    "created_at": _now(),
                    **result,
                }
                _append(handle, event)
                completed[reservation["attempt"]] = event
                stopped |= result["hard_stop"]
                if result.get("stop_reason"):
                    stop_reasons.add(result["stop_reason"])
                if _retry_allowed(event, reservation["item_attempt"], config):
                    pending.append(row)

    predictions = {}
    for row in prepared:
        attempts = item_attempts.get(row["id"], [])
        if not attempts:
            prediction = {
                "response": "",
                "error": "not_requested_hard_stop" if stopped else "not_requested_budget",
            }
        elif attempts[-1] not in completed:
            prediction = {
                "response": "",
                "error": "unknown_after_interruption",
                "delivery_status": "unknown; this reserved request will not be repeated",
            }
        else:
            prediction = dict(completed[attempts[-1]]["prediction"])
        history = [completed[attempt] for attempt in attempts if attempt in completed]
        prediction.update(
            attempts=len(attempts),
            completed_attempts=len(history),
            total_latency_seconds=sum(event["prediction"]["latency_seconds"] for event in history),
            retry_delay_seconds=sum(event.get("retry_delay_seconds", 0) for event in history),
            transport_statuses=[event.get("status_code") for event in history],
        )
        predictions[row["id"]] = prediction
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps({"id": row["id"], **predictions[row["id"]]}) + "\n" for row in prepared)
    )
    status_counts = {}
    for event in completed.values():
        status = str(event.get("status_code", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    (output / "summary.json").write_text(
        json.dumps(
            {
                "requested_model": config["model"],
                "provider": config["provider"],
                "examples": len(prepared),
                "reserved_attempts": len(reserved),
                "completed_attempts": len(completed),
                "retry_attempts": sum(
                    max(0, len(attempts) - 1) for attempts in item_attempts.values()
                ),
                "max_transient_retries": config.get("max_transient_retries", 0),
                "http_status_counts": status_counts,
                "hard_stopped": stopped,
                "stop_reasons": sorted(stop_reasons),
                "errors": sum(bool(p.get("error")) for p in predictions.values()),
                "latency_seconds_all_completed_attempts": sum(
                    event["prediction"]["latency_seconds"] for event in completed.values()
                ),
                "usage_unknown_attempts": len(reserved)
                - len(completed)
                + sum(event.get("status_code") != 200 for event in completed.values()),
                "billing": "Existing Roboflow credits; billed amount not returned by this API. "
                "Failed gateway requests may still be charged; their usage is unknown.",
            },
            indent=2,
        )
        + "\n"
    )
    return predictions
