"""Shared inference protocol, strict scoring, and paired comparisons."""

import base64
import hashlib
import io
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np
from PIL import Image

from vision_rl.data import fingerprint, load_manifest, write_json
from vision_rl.spatial import features
from vision_rl.training import load_policy

PROTOCOL = "image-mcq-v1/strict-answer/pass@1"


def prompt_text(row: dict) -> str:
    options = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(row["choices"]))
    return f"{row['prompt']}\n{options}\nReturn only <answer>LETTER</answer>."


def parse_answer(text: str, choices: list[str]) -> str | None:
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r"(?:<answer>\s*([A-Z])\s*</answer>|([A-Z]))", text.strip())
    if not match:
        return None
    letter = match.group(1) or match.group(2)
    return letter if ord(letter) - 65 < len(choices) else None


def wilson(successes: int, count: int) -> list[float]:
    z, p = 1.959963984540054, successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [max(0.0, float(center - half)), min(1.0, float(center + half))]


def score(rows: list[dict], predictions: dict[str, dict | str], model: dict) -> dict:
    if set(predictions) - {r["id"] for r in rows}:
        raise ValueError("predictions contain unknown example IDs")
    items = []
    for row in rows:
        pred = predictions.get(row["id"], {"error": "missing prediction"})
        if isinstance(pred, str):
            pred = {"response": pred}
        response = pred.get("response", "")
        answer = parse_answer(response, row["choices"])
        items.append(
            {
                "id": row["id"],
                "category": row["category"],
                "answer": answer,
                "expected": row["answer"],
                "correct": bool(answer == row["answer"]),
                "response": response,
                "error": pred.get("error"),
                "latency_seconds": pred.get("latency_seconds"),
                "usage": pred.get("usage"),
            }
        )
    correct = sum(item["correct"] for item in items)
    categories = {}
    for category in sorted({r["category"] for r in rows}):
        group = [i for i in items if i["category"] == category]
        n_correct = sum(i["correct"] for i in group)
        categories[category] = {
            "n": len(group),
            "accuracy": n_correct / len(group),
            "accuracy_ci95": wilson(n_correct, len(group)),
        }
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "attempts_per_item": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": fingerprint(rows),
        "splits": sorted({r["split"] for r in rows}),
        "model": model,
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "accuracy_ci95": wilson(correct, len(rows)),
        "invalid_answers": sum(i["answer"] is None for i in items),
        "errors": sum(bool(i["error"]) for i in items),
        "categories": categories,
        "items": items,
    }


def predict_endpoint(
    rows: list[dict], endpoint: str, model: str, max_requests: int = 100, max_tokens: int = 128
) -> dict:
    url = urlsplit(endpoint)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or url.username
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "endpoint must be an HTTP(S) base URL without credentials or query strings"
        )
    if len(rows) > max_requests or max_requests < 1 or max_tokens < 1:
        raise ValueError(
            f"{len(rows)} items exceed request budget {max_requests}, or token limit invalid"
        )
    headers = {}
    if key := os.environ.get("MODEL_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    predictions = {}
    # No automatic retries: retries can alter both cost and pass@1 semantics.
    with httpx.Client(timeout=120, headers=headers) as client:
        for row in rows:
            with Image.open(row["image"]) as source:
                image = source.convert("RGB")
                stream = io.BytesIO()
                image.save(stream, format="PNG")
            data_url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text(row)},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
            start = time.monotonic()
            try:
                response = client.post(
                    endpoint.rstrip("/") + "/chat/completions",
                    json={
                        "model": model,
                        "messages": [message],
                        "max_tokens": max_tokens,
                        "temperature": 0,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                text = payload["choices"][0]["message"]["content"]
                if not isinstance(text, str):
                    raise ValueError("nontext response")
                pred = {"response": text, "usage": payload.get("usage")}
            except httpx.HTTPStatusError as exc:
                pred = {"error": f"HTTP {exc.response.status_code}"}
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                pred = {"error": type(exc).__name__}
            predictions[row["id"]] = {**pred, "latency_seconds": time.monotonic() - start}
    return predictions


def read_predictions(path: Path) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("id"), str) or not isinstance(row.get("response"), str):
            raise ValueError("predictions must have string id and response fields")
        if row["id"] in result:
            raise ValueError("duplicate prediction ID")
        result[row["id"]] = row
    return result


def evaluate(
    manifest: Path,
    output: Path,
    checkpoint: Path | None = None,
    endpoint: str | None = None,
    model: str | None = None,
    predictions: Path | None = None,
    max_requests: int = 100,
    max_tokens: int = 128,
    vlm_model: str | None = None,
    adapter: Path | None = None,
) -> dict:
    if sum(x is not None for x in (checkpoint, endpoint, predictions, vlm_model)) != 1:
        raise ValueError("choose exactly one of checkpoint, endpoint, predictions, or vlm_model")
    if adapter and not vlm_model:
        raise ValueError("adapter requires vlm_model")
    rows = load_manifest(manifest)
    if checkpoint:
        weights = load_policy(checkpoint)
        labels = np.argmax(features(rows) @ weights, axis=1)
        responses = {r["id"]: chr(65 + int(a)) for r, a in zip(rows, labels, strict=True)}
        identity = {
            "name": model or "cpu-spatial-policy",
            "backend": "numpy",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    elif endpoint:
        if not model:
            raise ValueError("endpoint evaluation requires --model")
        responses = predict_endpoint(rows, endpoint, model, max_requests, max_tokens)
        identity = {
            "name": model,
            "backend": "chat-completions",
            "endpoint": endpoint,
            "max_tokens": max_tokens,
            "temperature": 0,
            "max_requests": max_requests,
        }
    elif vlm_model:
        from vision_rl.vlm import predict_vlm

        responses = predict_vlm(rows, vlm_model, adapter=adapter, max_new_tokens=max_tokens)
        identity = {
            "name": model or vlm_model,
            "backend": "transformers",
            "base_model": vlm_model,
            "adapter": str(adapter) if adapter else None,
            "max_new_tokens": max_tokens,
            "do_sample": False,
        }
        if adapter:
            hashes = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(adapter.glob("*.safetensors"))
            }
            identity["adapter_sha256"] = hashes
    else:
        responses = read_predictions(predictions)
        identity = {
            "name": model or "imported-predictions",
            "backend": "imported",
            "prediction_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            "provenance": "user-supplied; generation settings not independently verified",
        }
    result = score(rows, responses, identity)
    write_json(output, result)
    return result


def compare(
    specialist: Path, baseline: Path, output: Path, label_audit: Path | None = None
) -> dict:
    a, b = json.loads(specialist.read_text()), json.loads(baseline.read_text())
    for field in ("dataset_fingerprint", "protocol", "attempts_per_item", "splits"):
        if a[field] != b[field]:
            raise ValueError(f"cannot compare different {field}")
    ai, bi = {i["id"]: i for i in a["items"]}, {i["id"]: i for i in b["items"]}
    if not ai or ai.keys() != bi.keys() or len(ai) != len(a["items"]) or len(bi) != len(b["items"]):
        raise ValueError("comparison requires identical unique example IDs")
    differences = np.array([int(ai[k]["correct"]) - int(bi[k]["correct"]) for k in sorted(ai)])
    rng = np.random.default_rng(42)
    # Chunk to avoid allocating a huge resample matrix on real datasets.
    boot = np.array([rng.choice(differences, size=len(differences)).mean() for _ in range(5000)])
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    wins = Counter(int(d) for d in differences)
    valid = a["errors"] == 0 and b["errors"] == 0
    audit = {"status": "unreviewed"}
    if label_audit:
        audit = json.loads(label_audit.read_text())
        if audit.get("dataset_fingerprint") != a["dataset_fingerprint"]:
            raise ValueError("label audit belongs to a different dataset")
        if audit.get("status") not in {"unreviewed", "reviewed", "issues_found"}:
            raise ValueError("invalid label audit status")
    result = {
        "dataset_fingerprint": a["dataset_fingerprint"],
        "protocol": a["protocol"],
        "n": len(ai),
        "specialist": a["model"],
        "baseline": b["model"],
        "accuracy_difference": float(differences.mean()),
        "paired_bootstrap_ci95": ci,
        "specialist_wins": wins[1],
        "baseline_wins": wins[-1],
        "ties": wins[0],
        "label_audit": audit,
        "capability_comparison_valid": valid and audit["status"] == "reviewed",
        "positive_difference_ci": bool(ci[0] > 0 and valid and audit["status"] != "issues_found"),
        "confidence_method": (
            "paired item bootstrap, 5000 resamples, seed 42; assumes independent items"
        ),
        "evaluation_errors": {"specialist": a["errors"], "baseline": b["errors"]},
        "interpretation": "Evidence is limited to this dataset and protocol. Check model "
        "versions, inference budgets, leakage, and multiple comparisons before claiming a win.",
    }
    write_json(output, result)
    return result
