"""Prepare a shared16-TRAIN Aircraft probe and run the pinned local27B baseline."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from vision_rl import aircraft_data
from vision_rl.frontier import _public
from vision_rl.local_vlm import (
    GENERATION,
    REPOSITORY,
    REVISION,
    native_profile,
    prepare_local,
    run_local,
    sha_file,
)
from vision_rl.vlm_local_training import (
    OUTPUT_GRAMMAR,
    classification_metrics,
    load_spec,
    parse_label,
    prompt_text,
)

V2_INSTRUCTION = (
    "Inspect the aircraft’s visible distinguishing features. First give a brief "
    "explanation of the visible details, enclosed by <think> and </think>. Then "
    "write one label copied exactly from the allowed list on its own line. Do not "
    "add a prefix, quotes, or any text after that label."
)
INSTRUCT_INSTRUCTION = (
    "Return one label copied exactly from the allowed list on its own line. "
    "Do not add a prefix, quotes, explanations, or any text after that label."
)


def parser_identity():
    return {"name": OUTPUT_GRAMMAR, "parser_source_sha256": hashlib.sha256(
        inspect.getsource(parse_label).encode()).hexdigest()}


def shared_prompt_plan(path, spec, ordered_ids, pixels, *, profile="xhigh"):
    """Bind a prospective shared prompt, without loading or modifying actor code."""
    path = Path(path).resolve(strict=True)
    plan = json.loads(path.read_text())
    prompt_path = path.parent / "prompt.txt"
    prompt = prompt_path.read_text()
    grammar = parser_identity()
    mode = native_profile(profile)
    version = ("aircraft-required-brief-think-v2" if profile == "xhigh"
               else "aircraft-native-instruct-v1")
    if (plan.get("prompt_version") != version
            or plan.get("output_grammar") != grammar["name"]
            or plan.get("parser_source_sha256") != grammar["parser_source_sha256"]
            or plan.get("ids") != ordered_ids or plan.get("model_visible_pixels") != pixels
            or plan.get("prompt_sha256") != sha_file(prompt_path)):
        raise ValueError("Shared prompt plan, parser, fixed IDs or cropped pixels differ")
    if profile == "instruct" and plan.get("native_profile") != mode:
        raise ValueError("Instruct prompt plan must bind the exact native profile")
    instruction = V2_INSTRUCTION if profile == "xhigh" else INSTRUCT_INSTRUCTION
    expected = (spec["question"] + "\nAllowed labels (the complete class list):\n"
                + "\n".join(spec["labels"]) + "\n" + instruction)
    if prompt != expected:
        raise ValueError("Shared prompt must contain the exact question and full vocabulary")
    repository = Path(__file__).resolve().parents[1]
    for source, expected_hash in plan["inputs"].items():
        source = Path(source)
        source = source if source.is_absolute() else repository / source
        if sha_file(source) != expected_hash:
            raise ValueError("Frozen shared prompt plan input changed")
    return prompt, {
        "version": plan["prompt_version"], "plan_path": str(path),
        "plan_sha256": sha_file(path), "prompt_path": str(prompt_path),
        "prompt_sha256": sha_file(prompt_path), "grammar": grammar,
    }


def derive_instruct_plan(thinking_plan, output):
    """Freeze one new author-supported candidate from the exact shared TRAIN scope."""
    thinking_plan, output = Path(thinking_plan).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Native instruct prompt plans are immutable")
    parent = json.loads(thinking_plan.read_text())
    spec_paths = [Path(path) for path in parent["inputs"]
                  if Path(path).name == "data_spec.json"]
    if len(spec_paths) != 1:
        raise ValueError("Shared parent plan must identify exactly one data spec")
    spec_path = spec_paths[0]
    if not spec_path.is_absolute():
        spec_path = Path(__file__).resolve().parents[1] / spec_path
    spec = load_spec(spec_path)
    if len(spec["labels"]) != 100 or len(parent["ids"]) != 16:
        raise ValueError("Native instruct candidate retains fixed16 and full100 labels")
    shared_prompt_plan(thinking_plan, spec, parent["ids"], parent["model_visible_pixels"])
    prompt = (spec["question"] + "\nAllowed labels (the complete class list):\n"
              + "\n".join(spec["labels"]) + "\n" + INSTRUCT_INSTRUCTION)
    plan = {
        "schema_version": 1, "status": "prospectively_frozen_no_model_calls",
        "scope": "Same16 TRAIN discovery scope; candidate only, no strongest-baseline claim",
        "prompt_version": "aircraft-native-instruct-v1",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "output_grammar": OUTPUT_GRAMMAR,
        "parser_source_sha256": parser_identity()["parser_source_sha256"],
        "ids": parent["ids"], "model_visible_pixels": parent["model_visible_pixels"],
        "inputs": {**parent["inputs"], str(thinking_plan): sha_file(thinking_plan),
                   str(thinking_plan.parent / "prompt.txt"): sha_file(
                       thinking_plan.parent / "prompt.txt")},
        "derived_from_thinking_plan": str(thinking_plan),
        "native_profile": native_profile("instruct"),
        "selection": "Compare complete identical scopes by task accuracy; cost only tie-breaks",
        "caveat": "Incomplete xhigh is unmeasured, not defeated; test remains sealed",
    }
    output.mkdir(parents=True)
    (output / "prompt.txt").write_text(prompt)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def grammar_identity():
    text = OUTPUT_GRAMMAR + inspect.getsource(prompt_text) + inspect.getsource(parse_label)
    return {"name": OUTPUT_GRAMMAR, "prompt_and_parser_sha256": hashlib.sha256(
        text.encode()).hexdigest()}


def select_probe(rows, labels, seed=791):
    """Prospective stratified hash selection; no image decoding or model outputs."""
    if (len(labels) != 100 or len(set(labels)) != 100
            or len({r["image_id"] for r in rows}) != len(rows)):
        raise ValueError("Require complete100-class vocabulary and unique native IDs")
    if any(row["split"] != "train" or row["variant"] not in labels for row in rows):
        raise ValueError("Probe selection accepts only native TRAIN metadata")
    if {row["variant"] for row in rows} != set(labels):
        raise ValueError("Source training metadata must cover all100 canonical classes")

    def rank(kind, value):
        return hashlib.sha256(f"aircraft-probe-v1:{seed}:{kind}:{value}".encode()).hexdigest()

    chosen = sorted(labels, key=lambda label: rank("class", label))[:16]
    selected = []
    for label in chosen:
        candidates = [row for row in rows if row["variant"] == label]
        if not candidates:
            raise ValueError("Every selected class needs a native training image")
        selected.append(min(candidates, key=lambda row: rank("image", row["image_id"])))
    return selected


def prepare(root, spec_path, output, seed=791):
    root, spec_path, output = (Path(value).resolve() for value in (root, spec_path, output))
    if output.exists():
        raise ValueError("Shared Aircraft preparations are immutable")
    audit_path, train_path = root / "audit/image-audit.json", root / "audit/train-images.json"
    audit = json.loads(audit_path.read_text())
    if (not audit["training_integrity_clear"] or not audit["annotation_and_evaluator_exact_matches"]
            or audit["preprocessing_id"] != aircraft_data.PREPROCESSING_ID
            or audit["splits"]["train"]["metadata_image_manifest_sha256"] != sha_file(train_path)):
        raise ValueError("Aircraft images need a complete clean audit before preparation")
    spec = load_spec(spec_path)
    if spec.get("preprocessing_id") != aircraft_data.PREPROCESSING_ID:
        raise ValueError("Both model arms must use the shared lazy banner removal")
    selected = select_probe(json.loads(train_path.read_text()), spec["labels"], seed)
    prompt = prompt_text(spec, response_format="think_then_answer")
    rows, prepared, shared = [], [], []
    for source in selected:
        path = (root / source["path"]).resolve(strict=True)
        if not path.is_relative_to(root / "development/images/train"):
            raise ValueError("Selected image escapes audited native training paths")
        row = {"id": source["image_id"], "split": "train", "image": str(path), "prompt": prompt}
        image = prepare_local(row, aircraft_data.PREPROCESSING_ID)
        if (sha_file(path) != source["encoded_sha256"]
                or image["pixel_sha256"] != source["model_rgb_sha256"]
                or [image["width"], image["height"]] != source["model_image_size"]):
            raise ValueError("Selected pixels differ from the audited lazy crop")
        rows.append(row)
        prepared.append(_public(image))
        shared.append({"id": source["image_id"], "split": "train", "image": str(path),
                       "label": source["variant"]})
    protocol = {
        "schema_version": 1, "kind": "aircraft_shared_train16", "phase": "train_probe",
        "scope": "16TRAIN images spanning16classes; headroom/latency only, no benchmark claim",
        "seed": seed, "selection": "Hash16of100classes, then hash one native TRAIN image/class",
        "hash_key_template": "aircraft-probe-v1:{seed}:{class|image}:{canonical_label|image_id}",
        "image_root": str(root / "development/images/train"),
        "rows": rows, "prepared_images": prepared,
        "source_files": {str(path): sha_file(path) for path in [audit_path, train_path, spec_path]},
        "preprocessing_id": aircraft_data.PREPROCESSING_ID,
        "preprocessing_source_sha256": sha_file(aircraft_data.__file__),
        "task_contract": {"task": "classification", "benchmark": spec["benchmark"],
                          "probe_images": 16, "labels": spec["labels"],
                          "response_format": "think_then_answer", "grammar": grammar_identity()},
        "model": {"repository": REPOSITORY, "revision": REVISION,
                  "quantization": {"bits": 8, "group_size": 64, "mode": "affine"}},
        "generation": dict(GENERATION), "reasoning_effort": "xhigh",
        "processor": {"native": True, "min_pixels": 65536, "max_pixels": 16777216,
                      "external_resize": None, "same_cropped_input_as_actor": True,
                      "same_processor_architecture_as_actor": False},
        "examples": [], "data_spec_sha256": sha_file(spec_path),
    }
    output.mkdir(parents=True)
    (output / "data_spec.json").write_bytes(spec_path.read_bytes())
    (output / "shared-train16.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in shared))
    protocol["shared_rows_sha256"] = sha_file(output / "shared-train16.jsonl")
    (output / "inputs.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def prepare_mirror(directory, output, shared_plan=None, *, profile="xhigh"):
    """Bind the exact approved16-image mirror probe, keeping original equivalence pending."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    mode = native_profile(profile)
    if profile == "instruct" and shared_plan is None:
        raise ValueError("Native instruct preparation requires its separate frozen prompt plan")
    if output.exists():
        raise ValueError("Shared Aircraft preparations are immutable")
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    selection_path = Path(provenance["selection_path"]).resolve()
    selection = json.loads(selection_path.read_text())
    train_path, spec_path = directory / "train.jsonl", directory / "data_spec.json"
    if (provenance["status"] != "fixed_train_probe_ready_original_author_image_equivalence_pending"
            or provenance["selection_sha256"] != sha_file(selection_path)
            or provenance["train_manifest_sha256"] != sha_file(train_path)
            or provenance["data_spec_sha256"] != sha_file(spec_path)
            or provenance["preprocessing_source_sha256"] != sha_file(aircraft_data.__file__)
            or provenance["preprocessing_id"] != aircraft_data.PREPROCESSING_ID
            or provenance["test_images_accessed"] != 0):
        raise ValueError("Mirror preparation, exact selection or source provenance changed")
    spec = load_spec(spec_path)
    shared = [json.loads(line) for line in train_path.read_text().splitlines()]
    if ([{k: row[k] for k in ("id", "label", "split")} for row in shared]
            != selection["selected"] or len(shared) != 16
            or spec["labels"] != selection["canonical_labels"]
            or spec["preprocessing_id"] != aircraft_data.PREPROCESSING_ID):
        raise ValueError("Mirror IDs, class vocabulary or shared crop differ")
    prompt = prompt_text(spec, "think_then_answer")
    rows, prepared, resolved = [], [], []
    audit_images = {row["id"]: row for row in provenance["images"]}
    prompt_contract = None
    if shared_plan is not None:
        prompt, prompt_contract = shared_prompt_plan(
            shared_plan, spec, [row["id"] for row in shared],
            {identifier: row["model_rgb_sha256"] for identifier, row in audit_images.items()},
            profile=profile,
        )
    for source in shared:
        if set(source) != {"id", "split", "image", "label"} or source["split"] != "train":
            raise ValueError("Mirror manifest requires strict native TRAIN rows")
        path = (directory / source["image"]).resolve(strict=True)
        if not path.is_relative_to(directory / "original-jpeg"):
            raise ValueError("Mirror image escapes its fixed probe image directory")
        row = {"id": source["id"], "split": "train", "image": str(path), "prompt": prompt}
        image = prepare_local(row, aircraft_data.PREPROCESSING_ID)
        audit = audit_images[row["id"]]
        if (image["source_file_sha256"] != audit["encoded_sha256"]
                or image["pixel_sha256"] != audit["model_rgb_sha256"]
                or [image["width"], image["height"]] != audit["model_image_size"]
                or source["label"] != audit["label"]):
            raise ValueError("Mirror source or cropped pixels differ from provenance")
        rows.append(row)
        prepared.append(_public(image))
        resolved.append({**source, "image": str(path)})
    protocol = {
        "schema_version": 1, "kind": "aircraft_shared_train16", "phase": "train_probe",
        "scope": "16TRAIN mirror images; headroom/latency only; author equivalence pending",
        "seed": selection["seed"], "selection": selection["selection"],
        "image_root": str(directory / "original-jpeg"), "rows": rows,
        "prepared_images": prepared, "preprocessing_id": aircraft_data.PREPROCESSING_ID,
        "preprocessing_source_sha256": sha_file(aircraft_data.__file__),
        "source_files": {str(p): sha_file(p) for p in [
            provenance_path, selection_path, train_path, spec_path,
        ]},
        "image_provenance_scope": {
            key: provenance[key] for key in (
                "mirror_repo", "mirror_revision", "author_original_image_byte_equivalence_verified",
                "full_dataset_cross_split_pixel_audit_complete",
            )
        },
        "task_contract": {"task": "classification", "benchmark": spec["benchmark"],
                          "probe_images": 16, "labels": spec["labels"],
                          "response_format": "think_then_answer", "grammar": grammar_identity()},
        "model": {"repository": REPOSITORY, "revision": REVISION,
                  "quantization": {"bits": 8, "group_size": 64, "mode": "affine"}},
        "generation": dict(GENERATION), "reasoning_effort": "xhigh",
        "processor": {"native": True, "min_pixels": 65536, "max_pixels": 16777216,
                      "external_resize": None, "same_cropped_input_as_actor": True,
                      "same_processor_architecture_as_actor": False},
        "examples": [], "data_spec_sha256": sha_file(spec_path),
    }
    if prompt_contract is not None:
        protocol["schema_version"] = 2
        protocol["prompt_contract"] = prompt_contract
        protocol["task_contract"]["grammar"] = parser_identity()
        protocol["source_files"].update({
            prompt_contract["plan_path"]: prompt_contract["plan_sha256"],
            prompt_contract["prompt_path"]: prompt_contract["prompt_sha256"],
        })
    if profile == "instruct":
        protocol.update(schema_version=3, profile=profile, native_profile=mode["native_profile"],
                        generation=mode["generation"], reasoning_effort=None)
        protocol["task_contract"]["response_format"] = "answer_only"
    output.mkdir(parents=True)
    (output / "data_spec.json").write_bytes(spec_path.read_bytes())
    (output / "shared-train16.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in resolved))
    protocol["shared_rows_sha256"] = sha_file(output / "shared-train16.jsonl")
    (output / "inputs.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def verify(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "inputs.json").read_text())
    if protocol.get("kind") != "aircraft_shared_train16" or protocol.get("phase") != "train_probe":
        raise ValueError("Aircraft probe never opens validation or test")
    if len(protocol["rows"]) != 16 or any(row["split"] != "train" for row in protocol["rows"]):
        raise ValueError("Aircraft probe requires exactly16TRAIN images")
    for path, expected in protocol["source_files"].items():
        if sha_file(path) != expected:
            raise ValueError("Aircraft source audit, training metadata or spec changed")
    spec = load_spec(directory / "data_spec.json")
    profile = protocol.get("profile", "xhigh")
    mode = native_profile(profile)
    if protocol.get("schema_version") in {1, 2} and profile != "xhigh":
        raise ValueError("Older frozen protocols support only native xhigh")
    if protocol.get("schema_version") == 1:
        grammar, prompt = grammar_identity(), prompt_text(spec, "think_then_answer")
    elif protocol.get("schema_version") in {2, 3}:
        grammar = parser_identity()
        prompt, contract = shared_prompt_plan(
            protocol["prompt_contract"]["plan_path"], spec,
            [row["id"] for row in protocol["rows"]],
            {row["id"]: row["pixel_sha256"] for row in protocol["prepared_images"]},
            profile=profile,
        )
        if contract != protocol["prompt_contract"]:
            raise ValueError("Frozen shared prompt contract changed")
    else:
        raise ValueError("Unsupported Aircraft probe schema")
    if (sha_file(directory / "data_spec.json") != protocol["data_spec_sha256"]
            or sha_file(directory / "shared-train16.jsonl") != protocol["shared_rows_sha256"]
            or sha_file(aircraft_data.__file__) != protocol["preprocessing_source_sha256"]
            or grammar != protocol["task_contract"]["grammar"]):
        raise ValueError("Shared Aircraft preparation, transform or grammar changed")
    if (protocol["generation"] != mode["generation"]
            or protocol["reasoning_effort"] != mode["reasoning_effort"]):
        raise ValueError("Unsupported baseline sampling contract")
    if protocol.get("schema_version") == 3 and (
        protocol.get("native_profile") != mode["native_profile"]
        or protocol["task_contract"]["response_format"] != (
            "answer_only" if profile == "instruct" else "think_then_answer")
    ):
        raise ValueError("Prepared native profile and answer contract differ")
    expected_model = {"repository": REPOSITORY, "revision": REVISION,
                      "quantization": {"bits": 8, "group_size": 64, "mode": "affine"}}
    if (protocol["model"] != expected_model or protocol["examples"] != []
            or protocol["task_contract"]["labels"] != spec["labels"]
            or protocol["task_contract"]["probe_images"] != 16
            or protocol["processor"] != {
                "native": True, "min_pixels": 65536, "max_pixels": 16777216,
                "external_resize": None, "same_cropped_input_as_actor": True,
                "same_processor_architecture_as_actor": False,
            } or any(row["prompt"] != prompt for row in protocol["rows"])):
        raise ValueError("Model, vocabulary, processor or shared question differs")
    if [_public(prepare_local(row, protocol["preprocessing_id"]))
            for row in protocol["rows"]] != protocol["prepared_images"]:
        raise ValueError("Prepared Aircraft pixels or prompt changed")
    return protocol


def run(args):
    protocol = verify(args.input)
    profile = protocol.get("profile", "xhigh")
    if getattr(args, "profile", None) not in {None, profile}:
        raise ValueError("Run profile must match the frozen prepared profile")
    task = {**protocol["task_contract"], "prepared_protocol_sha256": sha_file(
        args.input / "inputs.json")}
    predictions = run_local(
        protocol["rows"], args.output, model_directory=args.model_directory,
        integrity=args.integrity, phase="train_probe", seed=args.seed,
        memory_limit_bytes=int(args.memory_limit_gib * 1024**3), resume=args.resume,
        image_root=protocol["image_root"], task_contract=task,
        preprocessing_id=protocol["preprocessing_id"],
        stop_after_completions=args.stop_after_completions, stream_progress=args.stream_progress,
        profile=profile,
    )
    metric_rows = [json.loads(line) for line in (args.input / "shared-train16.jsonl").read_text(
    ).splitlines()]
    completed = {
        identifier: {"response": row.get("classification_response", ""),
                     "terminated": row.get("finish_reason") == "stop" and not row.get("error")}
        for identifier, row in predictions.items()
    }
    if set(predictions) != {row["id"] for row in metric_rows}:
        raise ValueError("Partial execution must retain every fixed-scope image")
    executions = [json.loads(line) for line in (
        args.output / "executions.jsonl").read_text().splitlines()]
    if not executions or executions[-1]["event"] != "invocation_finished":
        raise ValueError("Local invocation did not finish; retain its journal without a summary")
    last = executions[-1]
    coverage = {key: last[key] for key in (
        "expected", "reserved", "completed", "unknown", "unattempted", "hard_stop",
        "cooperative_stop", "completed_this_invocation", "all_requests_completed",
    )}
    metric = None
    if coverage["all_requests_completed"] and not coverage["hard_stop"]:
        metric = classification_metrics(
            metric_rows, completed, task["labels"], allow_reasoning=True,
            decoding_protocol=f"native-{profile}-sampling/pass@1",
        )
    report = {"scope": protocol["scope"], "model": protocol["model"],
              "sampling": protocol["generation"], "metric": metric,
              "native_profile": native_profile(profile)["native_profile"],
              "coverage": coverage, "identity_sha256": last["identity_sha256"],
              "incomplete_scope_scored": False,
              "invocation": sum(row["event"] == "invocation_started" for row in executions),
              "request_errors": sum(
                  bool(row.get("error")) and row["error"] != "unknown_after_interruption"
                  and not row["error"].startswith("not_requested")
                  for row in predictions.values()),
              "benchmark_claim": False, "held_out_evaluation": False}
    with (args.output / f"summary-invocation-{report['invocation']:04}.json").open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("root", "spec", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--seed", type=int, default=791)
    mirror = sub.add_parser("prepare-mirror")
    mirror.add_argument("--source", type=Path, required=True)
    mirror.add_argument("--output", type=Path, required=True)
    mirror.add_argument("--shared-prompt-plan", type=Path)
    mirror.add_argument("--profile", choices=["xhigh", "instruct"], default="xhigh")
    mode_plan = sub.add_parser("plan-instruct")
    mode_plan.add_argument("--thinking-plan", type=Path, required=True)
    mode_plan.add_argument("--output", type=Path, required=True)
    infer = sub.add_parser("run")
    for name in ("input", "output", "model-directory", "integrity"):
        infer.add_argument("--" + name, type=Path, required=True)
    infer.add_argument("--seed", type=int, default=1729)
    infer.add_argument("--memory-limit-gib", type=float, default=32)
    infer.add_argument("--resume", action="store_true")
    infer.add_argument("--stop-after-completions", type=int)
    infer.add_argument("--stream-progress", action="store_true")
    infer.add_argument("--profile", choices=["xhigh", "instruct"])
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root, args.spec, args.output, args.seed)
    elif args.command == "prepare-mirror":
        result = prepare_mirror(args.source, args.output, args.shared_prompt_plan,
                                profile=args.profile)
    elif args.command == "plan-instruct":
        result = derive_instruct_plan(args.thinking_plan, args.output)
    else:
        result = run(args)
    print(json.dumps({"scope": result["scope"]}))


if __name__ == "__main__":
    main()
