"""Separate numeric-only expanded grader; legacy Q/V grader is unchanged.

The verifier supplies a real native endpoint and its independently frozen
receipt SHA. Submission consists of one safetensors file, never config/code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import expanded_contract as contract
from . import vlm_local_training as native


def predict_images(submitted_weights, image_rows, spec, profile, model_directory, *,
                   endpoint, expected_receipt_sha256, initial_path=None, device="cpu"):
    if (not image_rows or any(set(row) != {"id", "image"} for row in image_rows)
            or len({row["id"] for row in image_rows}) != len(image_rows)):
        raise ValueError("expanded prediction takes unique image-only id/image rows")
    profile.validate()
    values, approved = contract.validate_submission(
        submitted_weights, endpoint=endpoint, expected_receipt_sha256=expected_receipt_sha256,
        initial_path=initial_path)
    if asdict(profile) != approved["control_identity"]["profile"]:
        raise ValueError("verifier profile differs from approved endpoint")
    native.preflight(device)
    source = native._verified_model_directory(Path(model_directory), native.DEFAULT_MODEL)
    import torch
    import transformers

    torch.set_num_threads(2)
    processor = native._processor(Path(model_directory), profile)
    audit = native._audit_processor(processor, image_rows, spec, profile)
    base = transformers.AutoModelForImageTextToText.from_pretrained(
        str(model_directory), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float32, attn_implementation="sdpa")
    if base.config.model_type != "qwen2_vl":
        raise ValueError("expanded grader requires the pinned Qwen2-VL architecture")
    policy = contract._attach_values(base, values, layout=contract.active_layout(),
                                     is_trainable=False).to(device).eval()
    eos = policy.generation_config.eos_token_id
    eos = set(([eos] if isinstance(eos, int) else list(eos or [])) +
              [processor.tokenizer.eos_token_id])
    predictions = {}
    with torch.inference_mode():
        for row in image_rows:
            inputs = native._inputs(processor, row, spec, profile).to(device)
            tokens = policy.generate(**inputs, max_new_tokens=profile.max_completion_length,
                                     do_sample=False, use_cache=True)
            tail = tokens[:, inputs["input_ids"].shape[-1]:]
            predictions[row["id"]] = {
                "response": processor.batch_decode(tail, skip_special_tokens=True)[0],
                "raw_response": processor.batch_decode(tail, skip_special_tokens=False)[0],
                "terminated": bool(tail.shape[-1] and int(tail[0, -1].item()) in eos),
                "completion_ids": tail[0].tolist(),
            }
    return predictions, {"adapter_profile": contract.PROFILE_ID,
                         "weights_sha256": approved["weights_sha256"],
                         "endpoint_receipt_sha256": expected_receipt_sha256,
                         "model_source": source, "processor_audit": audit,
                         "submitted_config_or_code_read": False,
                         "prediction_input_fields": ["id", "image"]}


def grade_predictions(verifier_rows, predictions, labels):
    return native.classification_metrics(verifier_rows, predictions, labels, allow_reasoning=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("initial", "endpoint", "submission"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--endpoint", type=Path)
    parser.add_argument("--expected-receipt-sha256")
    parser.add_argument("--initial", type=Path,
                        help="this verifier's copy of the approved INIT (default: recorded path)")
    args = parser.parse_args()
    if args.kind != "initial" and args.expected_receipt_sha256 is None:
        parser.error("endpoint and submission checks require --expected-receipt-sha256")
    if args.kind == "submission" and args.endpoint is None:
        parser.error("submission checks require --endpoint")
    if args.kind == "initial":
        result = contract.validate_initial(args.path)
    elif args.kind == "endpoint":
        result = contract.validate_endpoint(
            args.path, expected_receipt_sha256=args.expected_receipt_sha256,
            initial_path=args.initial)
    else:
        _, result = contract.validate_submission(
            args.path, endpoint=args.endpoint,
            expected_receipt_sha256=args.expected_receipt_sha256, initial_path=args.initial)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
