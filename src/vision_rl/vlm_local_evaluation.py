"""Frozen, resumable FGVC-Aircraft native-VAL evaluation; no training or TEST inference.

Preparation requires the independently reconstructed author/cross-split clearance.
Inference receives only image IDs/paths and the complete task vocabulary. Gold
labels live in a separate file used only by the offline comparison command.
The existing clearance verifier rehashes retained TEST bytes for integrity only;
the evaluator never decodes or presents TEST images to a model.
Every image is durably reserved before generation. An interrupted reservation is
an unknown outcome, never an invitation to regenerate a more favorable answer.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import importlib
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

from . import experiment_loop as persistence
from . import vlm_local_task_builder as safe
from . import vlm_local_training as native
from .aircraft_data import PREPROCESSING_ID, aircraft_pixel_sha256
from .classification_comparison import compare_classification

PACKAGE = Path(__file__).resolve().parent
SCRIPTS = PACKAGE.parents[1] / "scripts"
NATIVE_IMAGES, NATIVE_CLASSES = 3333, 100
SCOPE = "fgvc_aircraft_native_validation"
ARM_IDS = (
    "common",
    "rl-exposure",
    "sft-1e5-exposure",
    "sft-5e5-exposure",
    "sft-1e4-exposure",
    "sft-1e5-time",
    "sft-5e5-time",
    "sft-1e4-time",
)
SCRIPT_FILES = ("verify_aircraft_clearance.py", "extract_aircraft.py", "prepare_aircraft.py")
JSON_LIMIT = 64 * 1024**2
OFFICIAL_EVALUATOR_SHA256 = "4e25f8db8a0d0163b10b167299c04c7a437a5f0de637f1e79628066ec297676a"


def read(path):
    return json.loads(
        safe._bounded_bytes(Path(path), JSON_LIMIT), parse_constant=persistence._bad_constant
    )


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    persistence._atomic_json(Path(path), value)


def _sha(path):
    return persistence._hash_path(Path(path))


def sources():
    # Freeze this package separately from the running TRAIN study. Complete
    # Python closure protects the native helpers' transitive imports as well.
    return {
        str(path): _sha(path)
        for path in [*sorted(PACKAGE.rglob("*.py")), *(SCRIPTS / name for name in SCRIPT_FILES)]
    }


def _require_clearance(root, metadata_archive, probe):
    """Recompute the established author check; no caller-supplied success flag."""
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(SCRIPTS))
        module = importlib.import_module("verify_aircraft_clearance")
        if Path(module.__file__).resolve() != (SCRIPTS / "verify_aircraft_clearance.py").resolve():
            raise ValueError("Clearance verifier imported from an unexpected path")
        return module.require_clearance(Path(root), Path(metadata_archive), Path(probe))
    finally:
        sys.path[:] = previous


def _positive(value, name, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"Invalid bounded {name}")
    return value


def declaration(raw):
    required = {
        "schema_version",
        "activated",
        "scope",
        "profile",
        "device",
        "arms",
        "budget",
        "bootstrap",
    }
    if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
        raise ValueError("Require the exact native-VAL declaration schema")
    if raw["activated"] is not True or raw["scope"] != SCOPE:
        raise ValueError("Native-VAL plan is unactivated; root review is required before outputs")
    if raw["device"] not in {"cpu", "mps"}:
        raise ValueError("Freeze one evaluation backend for every arm")
    if not isinstance(raw["profile"], dict) or set(raw["profile"]) != set(asdict(native.Profile())):
        raise ValueError("Freeze the complete inference profile")
    profile = native.Profile(**raw["profile"])
    profile.validate()
    if (
        profile.dtype != "float32"
        or profile.lora_rank != 8
        or profile.response_format != "answer_only"
    ):
        raise ValueError("This study freezes FP32 rank-eight answer-only inference")
    arms = raw["arms"]
    if not isinstance(arms, list) or [arm.get("id") for arm in arms] != list(ARM_IDS):
        raise ValueError("Freeze the common initialization, RL and all six declared SFT endpoints")
    for arm in arms:
        if set(arm) != {"id", "adapter", "weights_sha256", "record_sha256"}:
            raise ValueError(
                "Every endpoint requires its exact numeric artifact and parent receipt"
            )
        for name in ("weights_sha256", "record_sha256"):
            if (
                not isinstance(arm[name], str)
                or len(arm[name]) != 64
                or any(c not in "0123456789abcdef" for c in arm[name])
            ):
                raise ValueError("Endpoints require final SHA256 pins before evaluation")
    budget = raw["budget"]
    if set(budget) != {
        "max_wall_seconds",
        "max_invocations",
        "chunk_images",
        "chunk_seconds",
        "image_seconds",
        "max_output_bytes",
    }:
        raise ValueError("Freeze original evaluation and chunk/resource budgets")
    for name, maximum in (
        ("max_wall_seconds", 7 * 86400),
        ("chunk_seconds", 14400),
        ("image_seconds", 600),
        ("max_output_bytes", 4 * 1024**3),
    ):
        _positive(budget[name], name, maximum)
    for name, maximum in (("max_invocations", 10000), ("chunk_images", 512)):
        if type(budget[name]) is not int or not 1 <= budget[name] <= maximum:
            raise ValueError(f"Invalid bounded {name}")
    if (
        budget["image_seconds"] > budget["chunk_seconds"]
        or budget["chunk_seconds"] > budget["max_wall_seconds"]
    ):
        raise ValueError("Nested inference budgets are inconsistent")
    if (
        set(raw["bootstrap"]) != {"resamples", "seed"}
        or type(raw["bootstrap"]["resamples"]) is not int
        or not 100 <= raw["bootstrap"]["resamples"] <= 1_000_000
        or type(raw["bootstrap"]["seed"]) is not int
        or raw["bootstrap"]["seed"] < 0
    ):
        raise ValueError("Freeze bounded paired-bootstrap draws and seed")
    return json.loads(persistence._canonical(raw))


def _validation_rows(root, validation, spec):
    """Bind every published row to original author membership, labels and pixels."""
    root = Path(root).resolve()
    expected = read(root / "audit/val-images.json")
    if len(expected) != NATIVE_IMAGES or len(spec["labels"]) != NATIVE_CLASSES:
        raise ValueError("Require all3333 native VAL images and100 canonical variants")
    raw = [json.loads(line) for line in Path(validation).read_text().splitlines() if line.strip()]
    if len(raw) != len(expected) or len({r["image_id"] for r in expected}) != len(expected):
        raise ValueError("Native validation denominator differs")
    images, labels = [], []
    for row, author in zip(raw, expected, strict=True):
        if (
            set(row) != {"id", "image", "split", "label"}
            or row["split"] != "validation"
            or author["split"] != "val"
            or row["id"] != author["image_id"]
            or row["label"] != author["variant"]
        ):
            raise ValueError("Published VAL IDs, order or labels differ from author evidence")
        image = root / "development/images/validation" / (row["id"] + ".jpg")
        if (Path(validation).parent / row["image"]).resolve() != image or author["path"] != str(
            image.relative_to(root)
        ):
            raise ValueError("Only the original native-VAL JPEG path may be predicted")
        if (
            _sha(image) != author["encoded_sha256"]
            or aircraft_pixel_sha256(image) != author["model_rgb_sha256"]
        ):
            raise ValueError("Native VAL encoded bytes or model-visible pixels differ")
        images.append(
            {
                "id": row["id"],
                "image": str(image),
                "encoded_sha256": author["encoded_sha256"],
                "model_rgb_sha256": author["model_rgb_sha256"],
            }
        )
        labels.append({"id": row["id"], "label": row["label"]})
    if {row["label"] for row in labels} != set(spec["labels"]):
        raise ValueError("Every native variant must appear in validation")
    return images, labels


def _adapter(arm, spec_path, profile):
    path = Path(arm["adapter"]).absolute()
    record = path.parent / "local_training.json"
    if _sha(record) != arm["record_sha256"]:
        raise ValueError("Endpoint parent metadata changed")
    identity = {
        "model": native.DEFAULT_MODEL,
        "model_revision": native.MODEL_REVISIONS[native.DEFAULT_MODEL],
        "processor_model": native.DEFAULT_MODEL,
        "processor_revision": native.MODEL_REVISIONS[native.DEFAULT_MODEL],
        "profile": asdict(profile),
        "data_spec_sha256": _sha(spec_path),
    }
    initial = native._inference_adapter_identity(path, identity, profile)
    tensors, digest = safe.read_adapter(path / "adapter_model.safetensors")
    del tensors
    if digest != arm["weights_sha256"] or initial["weights_sha256"] != digest:
        raise ValueError("Endpoint numeric weights changed")
    config = native._validated_lora_config(read(path / "adapter_config.json"))
    trusted = safe._fixed_lora_config()
    if any(
        config[k] != trusted[k] for k in ("r", "lora_alpha", "lora_dropout", "bias", "task_type")
    ) or set(config["target_modules"]) != set(trusted["target_modules"]):
        raise ValueError("Endpoint LoRA scale/config differs from the trusted numeric architecture")
    return initial


def _controlled_endpoints(arms):
    """Verify endpoint roles from immutable receipts, not caller-provided arm names."""
    records = {arm["id"]: read(Path(arm["adapter"]).parent / "local_training.json") for arm in arms}
    common = records["common"]
    anchor = common["final_adapter_sha256"]
    if common["settings"]["max_steps"] != 400 or not common["settings"]["train_only"]:
        raise ValueError("Common initialization must be the declared400-step TRAIN warmup")
    receipts = {}
    for arm in arms[1:]:
        name, record = arm["id"], records[arm["id"]]
        receipt = read(Path(arm["adapter"]).parent / "trainer-receipt.json")
        mode = "grpo" if name == "rl-exposure" else "sft"
        lr = 1e-5 if name == "rl-exposure" or "1e5" in name else 5e-5 if "5e5" in name else 1e-4
        settings = record["settings"]
        if (
            record["control_identity"]["train"] != common["control_identity"]["train"]
            or record["control_identity"]["profile"] != common["control_identity"]["profile"]
            or record["control_identity"]["validation"] is not None
            or settings["init_adapter"]["weights_sha256"] != anchor
            or settings["mode"] != mode
            or settings["learning_rate"] != lr
            or settings["seed"] != 42
            or settings["train_only"] is not True
            or settings.get("initial_adapter_precision_transfer") is not None
            or settings["max_steps"] != (400 if mode == "grpo" else 6000)
            or settings["max_active_seconds"] != 10800
            or receipt["settings_sha256"] != native._object_sha256(settings)
            or receipt["control_identity_sha256"]
            != native._object_sha256(record["control_identity"])
            or receipt["final_adapter"]["weights_sha256"] != arm["weights_sha256"]
            or receipt["source_sha256"] != record["environment"]["adapter_source_sha256"]
            or receipt["source_sha256"]
            != records["rl-exposure"]["environment"]["adapter_source_sha256"]
            or receipt["committed_prompt_presentations"] != receipt["global_step"]
            or len(receipt["committed_prompt_ids"]) != receipt["global_step"]
            or type(receipt["global_step"]) is not int
            or not 1 <= receipt["global_step"] <= settings["max_steps"]
            or not math.isfinite(receipt["committed_active_seconds"])
            or receipt["committed_active_seconds"] <= 0
            or receipt["physical_work"] != record["optimizer_audit"]["active_time"]
        ):
            raise ValueError(
                "Endpoint does not belong to the declared common-warm controlled trajectory"
            )
        if name.endswith("exposure") and (
            receipt["global_step"] != 400
            or len(set(receipt["committed_prompt_ids"])) != 400
            or receipt["requested_boundary"]
            != {"stop_after_steps": 400, "stop_after_active_seconds": None}
        ):
            raise ValueError("Equal-exposure validation requires the exact400-update endpoint")
        if mode == "grpo" and (
            settings["beta"] != 0.04
            or record["reference"]["initial_adapter_weights_sha256"] != anchor
            or not record["optimizer_audit"]["reference_unchanged"]
        ):
            raise ValueError("RL reference differs from the immutable common initialization")
        receipts[name] = receipt
    target = receipts["rl-exposure"]["committed_active_seconds"]
    for name, receipt in receipts.items():
        if name.endswith("time"):
            exposure = receipts[name.removesuffix("time") + "exposure"]
            if receipt["committed_active_seconds"] < target or receipt["global_step"] < 400:
                raise ValueError(
                    "Capped SFT cannot be relabeled as a completed matched-time control"
                )
            if (
                receipt["committed_prompt_ids"][:400] != exposure["committed_prompt_ids"]
                or receipt["settings_sha256"] != exposure["settings_sha256"]
                or receipt["control_identity_sha256"] != exposure["control_identity_sha256"]
                or receipt["requested_boundary"]
                != (
                    exposure["requested_boundary"]
                    if exposure["committed_active_seconds"] >= target
                    else {"stop_after_steps": None, "stop_after_active_seconds": target}
                )
            ):
                raise ValueError(
                    "Matched-time control must continue its original exposure trajectory"
                )
    return receipts


def prepare(validation, data_spec, root, metadata_archive, probe, model_directory, study, output):
    """Freeze a separately approved comparison only after actual author clearance."""
    spec = declaration(read(study))  # Unactivated plans fail before dataset access.
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new evaluation bundle; resume the original frozen bundle")
    paths = {
        k: str(Path(v).resolve())
        for k, v in {
            "validation": validation,
            "data_spec": data_spec,
            "root": root,
            "metadata_archive": metadata_archive,
            "probe": probe,
            "model_directory": model_directory,
            "study": study,
        }.items()
    }
    clearance = _require_clearance(paths["root"], paths["metadata_archive"], paths["probe"])
    task = native.load_spec(Path(paths["data_spec"]))
    if task.get("preprocessing_id") != PREPROCESSING_ID:
        raise ValueError("Use the exact shared original-JPEG bottom20 transform")
    images, gold = _validation_rows(paths["root"], paths["validation"], task)
    profile = native.Profile(**spec["profile"])
    identities = {arm["id"]: _adapter(arm, paths["data_spec"], profile) for arm in spec["arms"]}
    receipts = _controlled_endpoints(spec["arms"])
    base = native._verified_model_directory(Path(paths["model_directory"]), native.DEFAULT_MODEL)
    clearance_path = Path(paths["root"]) / "audit/author-equivalence-v1.json"
    inputs = [Path(paths[k]) for k in ("validation", "data_spec", "study", "metadata_archive")]
    inputs += [clearance_path, Path(paths["root"]) / "audit/val-images.json"]
    inputs += [
        Path(paths["model_directory"]) / name
        for name in ("model-source.json", ".pinned-download.json", *base["file_sha256"])
    ]
    for arm in spec["arms"]:
        adapter = Path(arm["adapter"]).absolute()
        arm["adapter"] = str(adapter)
        inputs += [
            adapter / "adapter_model.safetensors",
            adapter / "adapter_config.json",
            adapter.parent / "local_training.json",
        ]
        if arm["id"] != "common":
            inputs.append(adapter.parent / "trainer-receipt.json")
    output.mkdir(parents=True)
    write(output / "images.json", images)
    write(output / "gold.json", gold)
    inputs += [output / "images.json", output / "gold.json"]
    plan = {
        "schema_version": 1,
        "scope": SCOPE,
        "study": spec,
        "paths": paths,
        "clearance": clearance,
        "base": base,
        "adapters": identities,
        "training_receipts": receipts,
        "source_sha256": sources(),
        "input_sha256": {str(p): _sha(p) for p in dict.fromkeys(inputs)},
        "runtime": safe.RUNTIME,
        "labels": task["labels"],
        "prompt_sha256": hashlib.sha256(
            native.prompt_text(task, profile.response_format).encode()
        ).hexdigest(),
        "official_evaluator_sha256": OFFICIAL_EVALUATOR_SHA256,
        "test_consumed": False,
    }
    write(output / "plan.json", plan)
    (output / "plan.sha256").write_text(_sha(output / "plan.json") + "\n")
    return plan


def _check(bundle):
    plan = read(bundle / "plan.json")
    if (
        _sha(bundle / "plan.json") != (bundle / "plan.sha256").read_text().strip()
        or plan["source_sha256"] != sources()
    ):
        raise ValueError("Frozen evaluation plan or executed source changed")
    declaration(plan["study"])
    for name, digest in plan["input_sha256"].items():
        if _sha(name) != digest:
            raise ValueError(f"Frozen evaluation input changed: {name}")
    if plan["runtime"] != safe.RUNTIME:
        raise ValueError("Frozen inference runtime changed")
    return plan


@contextlib.contextmanager
def _lock(bundle):
    with (bundle / ".evaluation.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("An evaluator already owns this study") from exc
        yield


def _journal_identity(plan, arm):
    """Semantic identity is independent of the directory containing a journal."""
    return {
        "schema_version": 1,
        "plan_sha256": persistence._digest(plan),
        "arm": arm,
        "source_sha256": persistence._digest(plan["source_sha256"]),
        "input_sha256": persistence._digest(plan["input_sha256"]),
        "base_sha256": persistence._digest(plan["base"]),
        "profile": plan["study"]["profile"],
        "device": plan["study"]["device"],
        "runtime": plan["runtime"],
        "prompt_sha256": plan["prompt_sha256"],
    }


def _generation_kwargs(profile):
    return {
        "max_new_tokens": profile["max_completion_length"],
        "do_sample": False,
        "use_cache": True,
    }


def _check_backend(backend, identity):
    if (
        backend.get("adapter_sha256") != identity["arm"]["weights_sha256"]
        or backend.get("environment", {}).get("device") != identity["device"]
        or {k: v.split("+")[0] for k, v in backend.get("runtime", {}).items()}
        != identity["runtime"]
        or backend.get("profile") != identity["profile"]
        or backend.get("prompt_sha256") != identity["prompt_sha256"]
        or backend.get("generation_kwargs") != _generation_kwargs(identity["profile"])
        or not isinstance(backend.get("generation_config"), dict)
        or backend.get("trainable_parameters") != 0
        or backend.get("prediction_fields") != ["id", "image"]
    ):
        raise ValueError("Backend receipt differs from the frozen arm/inference identity")


class Journal:
    """Arm-bound append chain with an independent durable head and invocation receipts.

    The head is committed before generation, so a valid earlier journal prefix
    cannot silently reopen consumed images. A crash after fsync but before the
    head rename can leave exactly one extra event, which is verified and adopted.
    Missing/older history fails closed. This is local integrity, not an external
    tamper-proof witness against rewriting every trusted file together.
    """

    def __init__(self, path, identifiers, identity, head_path, *, create=False):
        self.path, self.ids = Path(path), list(identifiers)
        self.identity, self.head_path = identity, Path(head_path)
        self.previous, self.sequence, self.pending, self.results = "0" * 64, 0, None, {}
        self.pending_invocation = None
        if not self.head_path.exists():
            if self.path.exists() or any(self.path.parent.glob("invocation-*.json")):
                raise ValueError("Prediction journal durable head is missing; preserve history")
            if not create:
                return
            write(self.head_path, self._head())
        head = read(self.head_path)
        if head.get("identity") != identity:
            raise ValueError("Prediction journal head belongs to a different frozen arm/plan")
        anchored = head == self._head()
        if self.path.exists():
            raw = safe._bounded_bytes(self.path, JSON_LIMIT)
            for line in raw.splitlines():
                event = json.loads(line, parse_constant=persistence._bad_constant)
                digest = event.pop("sha256")
                if (
                    digest != persistence._digest(event)
                    or event["previous"] != self.previous
                    or event["sequence"] != self.sequence
                ):
                    raise ValueError("Prediction journal is torn or changed; preserve it")
                self._apply(event)
                self.previous, self.sequence = digest, self.sequence + 1
                anchored |= head == self._head()
        if not anchored or not head["sequence"] <= self.sequence <= head["sequence"] + 1:
            raise ValueError("Prediction journal was deleted, rewound or changed beyond its head")
        if self.sequence != head["sequence"]:
            write(self.head_path, self._head())
        if self.sequence == 0 and create:
            self.append("header", None)

    def _head(self):
        return {"identity": self.identity, "sequence": self.sequence, "sha256": self.previous}

    def _invocation(self, reference):
        if (
            not isinstance(reference, dict)
            or set(reference) != {"number", "receipt_sha256"}
            or type(reference["number"]) is not int
            or reference["number"] < 1
        ):
            raise ValueError("Prediction lacks a bound backend invocation receipt")
        stem = self.path.parent / f"invocation-{reference['number']:05d}"
        receipt_path = stem.with_suffix(".identity.json")
        receipt = read(receipt_path)
        if (
            _sha(receipt_path) != reference["receipt_sha256"]
            or receipt.get("identity") != self.identity
            or receipt.get("invocation") != reference["number"]
            or type(receipt.get("pid")) is not int
            or receipt["pid"] <= 0
        ):
            raise ValueError("Backend invocation receipt changed or belongs to another arm")
        _check_backend(receipt["backend"], self.identity)
        metadata = read(stem.with_suffix(".json"))
        if any(
            metadata.get(k) != receipt[k] for k in ("identity", "invocation", "pid", "backend")
        ) or (metadata.get("receipt_sha256") != reference["receipt_sha256"]):
            raise ValueError("Backend invocation metadata contradicts its immutable receipt")

    def _apply(self, event):
        if event.get("identity") != self.identity:
            raise ValueError("Prediction journal event belongs to a different frozen arm/plan")
        identifier = event["id"]
        if event["event"] == "header":
            if self.sequence != 0 or identifier is not None:
                raise ValueError("Prediction journal header is missing or repeated")
            return
        if self.sequence == 0:
            raise ValueError("Prediction journal must start with its bound header")
        reference = event.get("invocation")
        self._invocation(reference)
        if event["event"] == "reserved":
            if (
                self.pending is not None
                or len(self.results) >= len(self.ids)
                or identifier != self.ids[len(self.results)]
            ):
                raise ValueError("Image was repeated or reordered in the frozen denominator")
            self.pending, self.pending_invocation = identifier, reference
        elif event["event"] == "outcome":
            if (
                self.pending != identifier
                or identifier in self.results
                or reference != self.pending_invocation
            ):
                raise ValueError("Image outcome lacks its unique durable invocation/reservation")
            self.results[identifier], self.pending = event["result"], None
            self.pending_invocation = None
        else:
            raise ValueError("Unknown prediction journal event")

    def append(self, event, identifier, **values):
        if event == "outcome":
            values.setdefault("invocation", self.pending_invocation)
        item = {
            "event": event,
            "id": identifier,
            "sequence": self.sequence,
            "previous": self.previous,
            "time": time.time(),
            **values,
            "identity": self.identity,
        }
        digest = persistence._digest(item)
        # Commit both bytes and independent head before any model call.
        self._apply(item)
        with self.path.open("ab") as handle:
            handle.write(persistence._canonical({**item, "sha256": digest}))
            handle.flush()
            os.fsync(handle.fileno())
        self.previous, self.sequence = digest, self.sequence + 1
        write(self.head_path, self._head())

    def recover(self):
        if self.pending is not None:
            self.append(
                "outcome",
                self.pending,
                result={
                    "status": "unknown_after_interruption",
                    "response": "",
                    "raw_response": "",
                    "terminated": False,
                    "completion_ids": [],
                    "canonical_label": None,
                },
            )


def _journal(bundle, plan, arm, images, *, create=False):
    return Journal(
        bundle / "arms" / arm["id"] / "predictions.jsonl",
        [row["id"] for row in images],
        _journal_identity(plan, arm),
        bundle / "journal-heads" / (arm["id"] + ".json"),
        create=create,
    )


@contextlib.contextmanager
def _deadline(seconds):
    """Worker-only hard image guard; a kill leaves its durable reservation unknown."""
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("Evaluation must run in its own main-thread worker")
    done = threading.Event()

    def watchdog():
        if not done.wait(seconds):
            os.kill(os.getpid(), signal.SIGINT)
            if not done.wait(2):
                os._exit(124)  # Native device calls may not promptly service Python signals.

    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=3)


class Backend:
    """One fresh, frozen, numeric-only policy; no Trainer, labels or optimizer."""

    def __init__(self, plan, arm):
        self.profile = native.Profile(**plan["study"]["profile"])
        self.device = plan["study"]["device"]
        self.spec = native.load_spec(Path(plan["paths"]["data_spec"]))
        self.environment = native.preflight(self.device)
        self.versions = safe._check_runtime()
        tensors, digest = safe.read_adapter(Path(arm["adapter"]) / "adapter_model.safetensors")
        if digest != arm["weights_sha256"]:
            raise ValueError("Numeric adapter identity changed before loading")
        torch, peft, transformers = (
            importlib.import_module(name) for name in ("torch", "peft", "transformers")
        )
        self.torch = torch
        torch.set_num_threads(2)
        transformers.set_seed(42)
        directory = Path(plan["paths"]["model_directory"])
        self.processor = native._processor(directory, self.profile)
        base = transformers.AutoModelForImageTextToText.from_pretrained(
            str(directory),
            trust_remote_code=False,
            local_files_only=True,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        if base.config.model_type != "qwen2_vl":
            raise ValueError("Unexpected pinned model architecture")
        self.policy = peft.get_peft_model(base, peft.LoraConfig(**safe._fixed_lora_config()))
        values = {key: torch.from_numpy(value) for key, value in tensors.items()}
        expected = peft.get_peft_model_state_dict(self.policy)
        if set(expected) != set(values) or any(
            expected[k].shape != v.shape for k, v in values.items()
        ):
            raise ValueError("Native model does not match the fixed numeric LoRA layout")
        peft.set_peft_model_state_dict(self.policy, values)
        if any(
            not torch.equal(value, values[key])
            for key, value in peft.get_peft_model_state_dict(self.policy).items()
        ):
            raise ValueError("Safe adapter reload changed numeric tensors")
        self.policy.requires_grad_(False).to(self.device).eval()
        eos = self.policy.generation_config.eos_token_id
        self.eos = set(
            ([eos] if isinstance(eos, int) else list(eos or []))
            + [self.processor.tokenizer.eos_token_id]
        )
        self.provenance = {
            "environment": self.environment,
            "runtime": self.versions,
            "generation_config": self.policy.generation_config.to_dict(),
            "generation_kwargs": _generation_kwargs(asdict(self.profile)),
            "profile": asdict(self.profile),
            "prompt_sha256": hashlib.sha256(
                native.prompt_text(self.spec, self.profile.response_format).encode()
            ).hexdigest(),
            "adapter_sha256": digest,
            "trainable_parameters": 0,
            "prediction_fields": ["id", "image"],
        }

    def predict(self, row):
        if set(row) != {"id", "image"}:
            raise ValueError("Prediction receives image-only rows")
        with self.torch.inference_mode():
            inputs = native._inputs(self.processor, row, self.spec, self.profile).to(self.device)
            if inputs["input_ids"].shape[-1] > self.profile.max_prompt_tokens:
                raise ValueError("Untruncated prompt exceeds the frozen token cap")
            tokens = self.policy.generate(
                **inputs,
                **_generation_kwargs(asdict(self.profile)),
            )
            tail = tokens[:, inputs["input_ids"].shape[-1] :]
            return {
                "response": self.processor.batch_decode(tail, skip_special_tokens=True)[0],
                "raw_response": self.processor.batch_decode(tail, skip_special_tokens=False)[0],
                "terminated": bool(tail.shape[-1] and int(tail[0, -1].item()) in self.eos),
                "completion_ids": tail[0].tolist(),
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            }

    def close(self):
        self.policy = self.processor = None
        gc.collect()
        if self.device == "mps":
            self.torch.mps.synchronize()
            self.torch.mps.empty_cache()


def _output_size(bundle):
    return sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())


def run_chunk(bundle, arm_id):
    """One bounded invocation. Unknown/completed images are never generated again."""
    bundle = Path(bundle).resolve()
    with _lock(bundle):
        plan = _check(bundle)
        arm = next((a for a in plan["study"]["arms"] if a["id"] == arm_id), None)
        if arm is None:
            raise ValueError("Arm is absent from the prospective comparison")
        images = read(bundle / "images.json")
        directory = bundle / "arms" / arm_id
        directory.mkdir(parents=True, exist_ok=True)
        journal = _journal(bundle, plan, arm, images, create=True)
        journal.recover()
        if len(journal.results) == len(images):
            return _summary(directory, journal, plan)
        state_path = bundle / "execution.json"
        state = (
            read(state_path)
            if state_path.exists()
            else {"created_at": time.time(), "invocations": 0}
        )
        budget = plan["study"]["budget"]
        remaining = state["created_at"] + budget["max_wall_seconds"] - time.time()
        if (
            remaining <= 0
            or state["invocations"] >= budget["max_invocations"]
            or _output_size(bundle) >= budget["max_output_bytes"]
        ):
            raise ValueError("Original evaluation budget exhausted; never reset on resume")
        # Recompute clearance before each actual model invocation, including resume.
        actual = _require_clearance(
            plan["paths"]["root"], plan["paths"]["metadata_archive"], plan["paths"]["probe"]
        )
        if actual != plan["clearance"]:
            raise ValueError("Author/cross-split clearance changed")
        if state["created_at"] + budget["max_wall_seconds"] <= time.time():
            raise ValueError("Original evaluation wall budget exhausted during clearance")
        state["invocations"] += 1
        state["worker"] = {"pid": os.getpid(), "arm": arm_id, "started_at": time.time()}
        write(state_path, state)
        started, backend, made = time.monotonic(), None, 0
        invocation = directory / f"invocation-{state['invocations']:05d}.json"
        metadata = {
            "status": "started",
            "pid": os.getpid(),
            "arm": arm_id,
            "identity": journal.identity,
            "invocation": state["invocations"],
            "plan_sha256": _sha(bundle / "plan.json"),
            "completed_before": len(journal.results),
        }
        write(invocation, metadata)
        try:
            with _deadline(
                min(
                    budget["chunk_seconds"],
                    state["created_at"] + budget["max_wall_seconds"] - time.time(),
                )
            ):
                backend = Backend(plan, arm)
            _check_backend(backend.provenance, journal.identity)
            metadata["backend"] = backend.provenance
            receipt_path = invocation.with_suffix(".identity.json")
            if receipt_path.exists():
                raise ValueError("Backend invocation identity cannot be overwritten")
            write(
                receipt_path, {k: metadata[k] for k in ("identity", "invocation", "pid", "backend")}
            )
            metadata["receipt_sha256"] = _sha(receipt_path)
            reference = {
                "number": state["invocations"],
                "receipt_sha256": metadata["receipt_sha256"],
            }
            write(invocation, metadata)
            for row in images[len(journal.results) :]:
                left = min(
                    state["created_at"] + budget["max_wall_seconds"] - time.time(),
                    budget["chunk_seconds"] - (time.monotonic() - started),
                )
                if (
                    made >= budget["chunk_images"]
                    or left <= 0
                    or _output_size(bundle) >= budget["max_output_bytes"]
                ):
                    break
                if (
                    _sha(row["image"]) != row["encoded_sha256"]
                    or aircraft_pixel_sha256(row["image"]) != row["model_rgb_sha256"]
                ):
                    raise ValueError("Model-visible native image changed")
                journal.append("reserved", row["id"], invocation=reference)
                tick = time.monotonic()
                try:
                    with _deadline(min(budget["image_seconds"], left)):
                        result = backend.predict({"id": row["id"], "image": row["image"]})
                    if (
                        type(result["terminated"]) is not bool
                        or not isinstance(result["response"], str)
                        or len(result["response"]) > 65536
                        or len(result["completion_ids"])
                        > plan["study"]["profile"]["max_completion_length"]
                    ):
                        raise ValueError("Malformed or oversized inference output")
                    result.update(
                        status="completed",
                        canonical_label=native.parse_label(
                            result["response"], plan["labels"], allow_reasoning=True
                        )
                        if result["terminated"]
                        else None,
                        latency_seconds=time.monotonic() - tick,
                    )
                except Exception as exc:
                    journal.append(
                        "outcome",
                        row["id"],
                        result={
                            "status": "generation_error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "response": "",
                            "raw_response": "",
                            "terminated": False,
                            "completion_ids": [],
                            "canonical_label": None,
                            "latency_seconds": time.monotonic() - tick,
                        },
                    )
                    raise
                journal.append("outcome", row["id"], result=result)
                made += 1
            _check(bundle)
            metadata["status"] = "completed_chunk"
        except BaseException as exc:
            metadata.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if backend is not None:
                backend.close()
            metadata.update(
                generated_this_invocation=made, elapsed_seconds=time.monotonic() - started
            )
            write(invocation, metadata)
            state.pop("worker", None)
            write(state_path, state)
        return _summary(directory, journal, plan)


def _summary(directory, journal, plan):
    counts = {
        name: sum(r["status"] == name for r in journal.results.values())
        for name in ("completed", "generation_error", "unknown_after_interruption")
    }
    report = {
        "scope": SCOPE,
        "arm": directory.name,
        "status": "completed" if len(journal.results) == NATIVE_IMAGES else "partial",
        "counts": counts,
        "remaining": NATIVE_IMAGES - len(journal.results),
        "raw_valid_answers": sum(
            r["canonical_label"] is not None for r in journal.results.values()
        ),
        "prediction_journal_sha256": _sha(journal.path),
        "plan_sha256": persistence._digest(plan),
        "metric": None,
        "test_consumed": False,
        "primary_requirement_satisfied": False,
    }
    write(directory / "summary.json", report)
    return report


def compare(bundle):
    """Offline only: full denominator, all arms, no adaptive additional training."""
    bundle = Path(bundle).resolve()
    with _lock(bundle):
        plan = _check(bundle)
        images, rows = read(bundle / "images.json"), read(bundle / "gold.json")
        if len(rows) != NATIVE_IMAGES or [r["id"] for r in rows] != [r["id"] for r in images]:
            raise ValueError("Offline native denominator changed")
        predictions, failures, journals = {}, {}, {}
        for arm in plan["study"]["arms"]:
            journal = _journal(bundle, plan, arm, images)
            if journal.pending is not None or len(journal.results) != len(images):
                raise ValueError(
                    "Every prospectively declared arm must finish the complete denominator"
                )
            parsed = {
                identifier: native.parse_label(r["response"], plan["labels"], allow_reasoning=True)
                if r["status"] == "completed" and r["terminated"]
                else None
                for identifier, r in journal.results.items()
            }
            if any(parsed[k] != v["canonical_label"] for k, v in journal.results.items()):
                raise ValueError("Canonical predictions differ from the frozen raw-output parser")
            predictions[arm["id"]] = parsed
            failures[arm["id"]] = sum(r["status"] != "completed" for r in journal.results.values())
            journals[arm["id"]] = _sha(journal.path)
        kwargs = plan["study"]["bootstrap"]
        comparisons = {
            name: compare_classification(
                rows, pred, predictions["common"], plan["labels"], **kwargs
            )
            for name, pred in predictions.items()
            if name != "common"
        }
        controls = {
            name: compare_classification(
                rows, predictions["rl-exposure"], pred, plan["labels"], **kwargs
            )
            for name, pred in predictions.items()
            if name.startswith("sft-")
        }
        strongest = max(
            sorted(controls), key=lambda name: controls[name]["comparator"]["mean_class_accuracy"]
        )
        report = {
            "schema_version": 1,
            "scope": SCOPE,
            "split": "native_validation",
            "n": NATIVE_IMAGES,
            "classes": NATIVE_CLASSES,
            "metric": "mean(diag(row_normalized_confusion))",
            "official_evaluator_sha256": OFFICIAL_EVALUATOR_SHA256,
            "official_matlab_executed_by_this_command": False,
            "implementation": (
                "Equivalent exact-label equal-class accuracy; missing/invalid outcomes "
                "stay in class denominators"
            ),
            "versus_common_initialization": comparisons,
            "rl_versus_each_sft": controls,
            "strongest_declared_sft": strongest,
            "strongest_sft_selection_rule": (
                "Highest macroaccuracy across the six prospectively declared SFT endpoints; "
                "lexical ID breaks ties. No extra fits or accuracy-dependent endpoint additions."
            ),
            "failures_by_arm": failures,
            "all_inference_completed_without_unknown_or_errors": not any(failures.values()),
            "prediction_journal_sha256": journals,
            "plan_sha256": _sha(bundle / "plan.json"),
            "primary_requirement_satisfied": False,
            "test_consumed": False,
            "claim": (
                "Development generalization comparison; substantial-gain protocol "
                "remains separately unactivated"
            ),
        }
        write(bundle / "comparison.json", report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    for name in (
        "validation",
        "data-spec",
        "root",
        "metadata-archive",
        "probe",
        "model-directory",
        "study",
        "output",
    ):
        preparation.add_argument("--" + name, type=Path, required=True)
    run = commands.add_parser("run-chunk")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--arm", choices=ARM_IDS, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--bundle", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        result = (
            prepare(**args)
            if command == "prepare"
            else run_chunk(args["bundle"], args["arm"])
            if command == "run-chunk"
            else compare(args["bundle"])
        )
        print(
            json.dumps(
                {"status": "prepared", "scope": result["scope"]}
                if command == "prepare"
                else result,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"local VLM evaluation: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
