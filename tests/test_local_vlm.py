import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from vision_rl import local_vlm


@pytest.fixture
def local_run(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "preprocessor_config.json", "processor_config.json"]
    names += [f"model-{i:05d}-of-00006.safetensors" for i in range(1, 7)]
    files = []
    for name in names:
        data = name.encode()
        (model / name).write_bytes(data)
        files.append({"path": name, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    integrity = tmp_path / "integrity.json"
    integrity.write_text(json.dumps({
        "model_repository": local_vlm.REPOSITORY, "revision": local_vlm.REVISION,
        "quantization": {"bits": 8, "mode": "affine", "group_size": 64}, "files": files,
    }))
    train = tmp_path / "train"
    train.mkdir()
    rows = []
    for i in range(2):
        image = train / f"{i}.png"
        Image.new("RGB", (20, 10), (i * 50, 0, 0)).save(image)
        rows.append({"id": str(i), "image": str(image), "prompt": "Find pills."})
    monkeypatch.setattr(local_vlm, "runtime_versions", lambda: {"mlx-vlm": "0.6.8"})
    return rows, tmp_path / "run", {
        "model_directory": model, "integrity": integrity, "phase": "train_probe",
    }


class Backend:
    load_info = {"load_seconds": 0.1, "device": {"test": True}}
    calls = []

    def __init__(self, directory, config):
        self.config = config

    def infer(self, row, seed):
        self.calls.append((row["id"], seed))
        return {"native_response": {
            "text": 'Reasoning with {"not": "the answer"}</think>\n[]',
            "finish_reason": "stop", "prompt_tokens": 10, "generation_tokens": 4,
            "total_tokens": 14, "prompt_tps": 2.0, "generation_tps": 1.0,
            "peak_memory": 0.5,
        }, "preprocessing": {"image_grid_thw": [[1, 2, 3]]}, "peak_memory_bytes": 500000000}


def test_local_resume_preserves_raw_reasoning_usage_and_never_repeats_answers(local_run):
    rows, output, kwargs = local_run
    Backend.calls = []
    first = local_vlm.run_local(rows, output, backend_factory=Backend, **kwargs)
    journal = (output / "attempts.jsonl").read_bytes()
    second = local_vlm.run_local(rows, output, backend_factory=Backend, resume=True, **kwargs)
    assert first == second and len(Backend.calls) == 2
    assert first["0"]["response"] == "[]" and first["0"]["usage"]["total_tokens"] == 14
    assert (output / "attempts.jsonl").read_bytes() == journal
    events = [json.loads(line) for line in journal.splitlines()]
    assert [event["event"] for event in events] == ["reserved", "completed"] * 2
    assert "Reasoning with" in events[1]["native_response"]["text"]
    assert events[1]["preprocessing"]["image_grid_thw"] == [[1, 2, 3]]


def test_unknown_local_attempt_is_never_repeated_after_crash(local_run):
    rows, output, kwargs = local_run

    class Crashed(Backend):
        def infer(self, row, seed):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        local_vlm.run_local(rows, output, backend_factory=Crashed, **kwargs)
    Backend.calls = []
    result = local_vlm.run_local(rows, output, backend_factory=Backend, resume=True, **kwargs)
    assert [identifier for identifier, _ in Backend.calls] == ["1"]
    assert result["0"]["error"] == "unknown_after_interruption"


def test_memory_failure_is_journaled_and_stops_pending_work(local_run):
    rows, output, kwargs = local_run

    class TooLarge(Backend):
        def infer(self, row, seed):
            raise MemoryError("local allocation rejected")

    first = local_vlm.run_local(rows, output, backend_factory=TooLarge, **kwargs)
    Backend.calls = []
    second = local_vlm.run_local(rows, output, backend_factory=Backend, resume=True, **kwargs)
    assert first == second and Backend.calls == []
    assert first["0"]["error"] == "MemoryError"
    assert first["1"]["error"] == "not_requested_hard_stop"
    event = json.loads((output / "attempts.jsonl").read_text().splitlines()[1])
    assert event["hard_stop"] and event["exception_message"] == "local allocation rejected"


@pytest.mark.parametrize("change", ["seed", "prompt", "pixels", "model"])
def test_local_resume_rejects_identity_changes_before_model_load(local_run, change):
    rows, output, kwargs = local_run
    local_vlm.run_local(rows, output, backend_factory=Backend, **kwargs)
    if change == "seed":
        kwargs["seed"] = 33
    elif change == "prompt":
        rows[0]["prompt"] += " Changed."
    elif change == "pixels":
        Image.new("RGB", (20, 10), "green").save(rows[0]["image"])
    else:
        (kwargs["model_directory"] / "config.json").write_text("changed")

    def forbidden(*args):
        raise AssertionError("Changed identity reached model loading")

    with pytest.raises(ValueError, match="identity differs|integrity mismatch"):
        local_vlm.run_local(rows, output, backend_factory=forbidden, resume=True, **kwargs)


@pytest.mark.parametrize("text,finish", [
    ("[]", "stop"), ("thinking</think>[]", "length"),
    ("thinking</think>[]</think>[]", "stop"), ("thinking</think>", "stop"),
])
def test_local_answer_needs_normal_finish_and_one_explicit_reasoning_close(text, finish):
    assert local_vlm.final_answer(text, finish).get("error")


def test_local_development_runner_rejects_test_before_loading(local_run):
    rows, output, kwargs = local_run
    kwargs["phase"] = "test"
    with pytest.raises(ValueError, match="cannot access final-test"):
        local_vlm.run_local(rows, output, **kwargs)
    assert not output.exists()


def test_generic_image_root_requires_explicit_split_and_binds_task_contract(local_run):
    rows, output, kwargs = local_run
    for row in rows:
        row["split"] = "train"
    root = Path(rows[0]["image"]).parent
    task = {"task": "classification", "labels": ["A", "B"], "output_format": "exact_label"}
    local_vlm.run_local(rows, output, image_root=root, task_contract=task,
                        backend_factory=Backend, **kwargs)
    config = json.loads((output / "run.json").read_text())["identity"]["config"]
    assert config["coordinate_format"] is None and config["task_contract"] == task
    result = json.loads((output / "predictions.jsonl").read_text().splitlines()[0])
    assert result["classification_response"].startswith("<think>Reasoning with")
    assert result["classification_response"].endswith("</think>\n[]")
    task["labels"].append("C")
    with pytest.raises(ValueError, match="identity differs"):
        local_vlm.run_local(rows, output, image_root=root, task_contract=task,
                            backend_factory=Backend, resume=True, **kwargs)
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="declared split root"):
        local_vlm.run_local(rows, output, image_root=root, task_contract=task,
                            backend_factory=Backend, resume=True, **kwargs)


def test_model_load_failure_has_durable_evidence_before_any_image_attempt(local_run):
    rows, output, kwargs = local_run

    def cannot_load(*args):
        raise MemoryError("weights exceed available allocation")

    with pytest.raises(MemoryError, match="weights exceed"):
        local_vlm.run_local(rows, output, backend_factory=cannot_load, **kwargs)
    events = [json.loads(line) for line in (output / "runtime.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["load_started", "load_failed"]
    assert events[1]["exception_type"] == "MemoryError"
    assert not (output / "attempts.jsonl").exists()


def test_resume_retains_each_model_load_record_without_repeating_ambiguous_image(local_run):
    rows, output, kwargs = local_run

    class Crashed(Backend):
        def infer(self, row, seed):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        local_vlm.run_local(rows, output, backend_factory=Crashed, **kwargs)
    earlier = (output / "runtime.jsonl").read_bytes()
    local_vlm.run_local(rows, output, backend_factory=Backend, resume=True, **kwargs)
    current = (output / "runtime.jsonl").read_bytes()
    assert current.startswith(earlier)
    assert len(current.splitlines()) == 4


def test_prepare_probe_reads_training_only_and_deterministically_stresses_largest(tmp_path):
    import runpy
    from pathlib import Path

    wrapper = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/pills_local.py"))
    train = tmp_path / "data/train"
    train.mkdir(parents=True)
    images = []
    for identifier, size in enumerate([(20, 20), (30, 20), (40, 20), (40, 20)]):
        name = f"{identifier}.jpg"
        Image.new("RGB", size).save(train / name)
        images.append({"id": identifier, "file_name": name, "width": size[0], "height": size[1]})
    (train / "_annotations.coco.json").write_text(json.dumps({
        "images": images, "annotations": [], "categories": [{"id": 1, "name": "wheat head"}],
    }))
    output = tmp_path / "prepared"
    protocol = wrapper["prepare_probe"](train.parent, output)
    assert [row["id"] for row in protocol["rows"]] == ["3", "2"]
    assert all("wheat head" in row["prompt"] for row in protocol["rows"])
    assert protocol["phase"] == "train_probe" and protocol["examples"] == []
    assert protocol["prepared_images"][0]["width"] == 40
    with pytest.raises(ValueError, match="immutable"):
        wrapper["prepare_probe"](train.parent, output)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("profile", ["xhigh", "instruct"])
def test_native_backend_preserves_coco_pixels_and_explicit_thinking_without_gpu(
    tmp_path, monkeypatch, streaming, profile,
):
    import dataclasses
    import sys
    from types import SimpleNamespace

    import numpy as np

    path = tmp_path / "rotated-tag.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (20, 10), "red").save(path, exif=exif)
    calls = {}

    def template(processor, config, prompt, **kwargs):
        calls["template"] = kwargs
        if profile == "xhigh":
            return "Reasoning effort is set to xhigh.\n<|im_start|>assistant\n<think>\n"
        return "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def prepare(processor, **kwargs):
        image = kwargs["images"][0]
        assert isinstance(image, Image.Image) and image.size == (20, 10)
        assert image.mode == "RGB" and kwargs["resize_shape"] is None
        return {"input_ids": np.array([[1, 2]]), "pixel_values": np.zeros((1, 6)),
                "image_grid_thw": np.array([[1, 2, 3]]), "attention_mask": np.ones((1, 2))}

    @dataclasses.dataclass
    class Result:
        text: str = "reasoning</think>[]"
        finish_reason: str = "stop"
        token: int = 42

    def generate(model, processor, prompt, **kwargs):
        calls["generation"] = kwargs
        assert kwargs["input_ids"].tolist() == [[1, 2]]
        return Result()

    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(generate=generate))
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils",
                        SimpleNamespace(apply_chat_template=template))
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", SimpleNamespace(prepare_inputs=prepare))
    if streaming:
        from vision_rl import local_vlm_execution

        def streamed(model, processor, prompt, *, progress, **kwargs):
            progress({"generation_tokens": 1, "finish_reason": None})
            progress({"generation_tokens": 2, "finish_reason": "stop"})
            return generate(model, processor, prompt, **kwargs)

        monkeypatch.setattr(local_vlm_execution, "generate_with_progress", streamed)
    backend = local_vlm.MLXBackend.__new__(local_vlm.MLXBackend)
    backend.config = {**local_vlm.native_profile(profile), "stream_progress": streaming,
                      "generation_progress_path": str(tmp_path / "progress.jsonl"),
                      "identity_sha256": "frozen_identity"}
    backend.model = SimpleNamespace(config=SimpleNamespace(image_token_index=99))
    backend.processor = object()
    backend.mx = SimpleNamespace(
        clear_cache=lambda: None, reset_peak_memory=lambda: None,
        random=SimpleNamespace(seed=lambda seed: calls.update(seed=seed)),
        get_peak_memory=lambda: 100, get_active_memory=lambda: 90,
        get_cache_memory=lambda: 10,
    )
    result = backend.infer({"id": "one", "image_path": str(path), "prompt": "Find wheat."}, 123)
    assert calls["seed"] == 123
    assert calls["template"]["enable_thinking"] is (profile == "xhigh")
    assert calls["template"].get("reasoning_effort") == (
        "xhigh" if profile == "xhigh" else None)
    assert calls["generation"]["enable_thinking"] is (profile == "xhigh")
    for key, value in local_vlm.native_profile(profile)["generation"].items():
        assert calls["generation"][key] == value
    assert calls["generation"]["max_tokens"] == 16384
    assert result["native_response"]["text"] == "reasoning</think>[]"
    assert result["preprocessing"]["exif_orientation_transform"] is False
    assert result["preprocessing"]["native_profile"] == f"qwen3.8-native-{profile}-v1"
    if streaming:
        observations = [json.loads(line) for line in (
            tmp_path / "progress.jsonl").read_text().splitlines()]
        assert [row["generation_tokens"] for row in observations] == [1, 2]
        assert all(row["identity_sha256"] == "frozen_identity" and row["id"] == "one"
                   and row["seed"] == 123 for row in observations)


def test_cancelled_run_refuses_even_explicit_resume_before_verification(local_run, monkeypatch):
    rows, output, kwargs = local_run
    output.mkdir()
    (output / "cancelled.json").write_text('{"reason":"explicit user priority"}')

    def forbidden(*args):
        raise AssertionError("Cancelled run reached model verification")

    monkeypatch.setattr(local_vlm, "verify_snapshot", forbidden)
    with pytest.raises(ValueError, match="Cancelled"):
        local_vlm.run_local(rows, output, resume=True, **kwargs)


def test_fixed16_journal_chunks_resume_without_changing_identity_or_repeating(local_run):
    rows, output, kwargs = local_run
    train = Path(rows[0]["image"]).parent
    rows = []
    for i in range(16):
        path = train / f"chunk-{i}.png"
        Image.new("RGB", (20, 10), (i * 10, 0, 0)).save(path)
        rows.append({"id": str(i), "split": "train", "image": str(path), "prompt": "Classify."})
    kwargs.update(image_root=train, task_contract={"task": "classification", "probe_images": 16},
                  backend_factory=Backend, stream_progress=True)
    Backend.calls = []
    first = local_vlm.run_local(rows, output, stop_after_completions=4, **kwargs)
    identity = (output / "run.json").read_bytes()
    first_journal = (output / "attempts.jsonl").read_bytes()
    assert [identifier for identifier, _ in Backend.calls] == [str(i) for i in range(4)]
    assert sum(row.get("error") == "not_requested_cooperative_stop"
               for row in first.values()) == 12
    second = local_vlm.run_local(rows, output, stop_after_completions=3, resume=True, **kwargs)
    assert (output / "run.json").read_bytes() == identity
    assert (output / "attempts.jsonl").read_bytes().startswith(first_journal)
    assert sum(row.get("error") == "not_requested_cooperative_stop"
               for row in second.values()) == 9
    final = local_vlm.run_local(rows, output, resume=True, **kwargs)
    assert list(final) == [str(i) for i in range(16)]
    assert [identifier for identifier, _ in Backend.calls] == list(final)
    assert all(not row.get("error") for row in final.values())
    execution = [json.loads(line) for line in (
        output / "executions.jsonl").read_text().splitlines()]
    finishes = [row for row in execution if row["event"] == "invocation_finished"]
    assert [row["completed_this_invocation"] for row in finishes] == [4, 3, 9]
    assert [row["unattempted"] for row in finishes] == [12, 9, 0]
    assert finishes[-1]["all_requests_completed"]
    kwargs["stream_progress"] = False
    with pytest.raises(ValueError, match="identity differs"):
        local_vlm.run_local(rows, output, resume=True, **kwargs)


def test_native_profiles_are_explicit_independent_and_bind_resume(local_run):
    rows, output, kwargs = local_run
    expected = {"max_tokens": 16384, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
                "enable_thinking": False}
    profile = local_vlm.native_profile("instruct")
    assert profile["generation"] == expected and profile["reasoning_effort"] is None
    profile["generation"]["temperature"] = 99
    assert local_vlm.native_profile("instruct")["generation"] == expected
    assert local_vlm.native_profile()["generation"] == local_vlm.GENERATION
    local_vlm.run_local(rows, output, profile="instruct", backend_factory=Backend, **kwargs)
    config = json.loads((output / "run.json").read_text())["identity"]["config"]
    assert config["generation"] == expected
    assert config["assistant_prefill"] == "<think>\n\n</think>\n\n"
    with pytest.raises(ValueError, match="identity differs"):
        local_vlm.run_local(rows, output, profile="xhigh", resume=True,
                            backend_factory=Backend, **kwargs)


def test_unknown_native_profile_rejected_before_model_or_output(local_run, monkeypatch):
    rows, output, kwargs = local_run

    def forbidden(*args):
        raise AssertionError("Unknown profile reached model verification")

    monkeypatch.setattr(local_vlm, "verify_snapshot", forbidden)
    with pytest.raises(ValueError, match="profile"):
        local_vlm.run_local(rows, output, profile="forced-zero-thinking", **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("text", [
    "Saab 340", "Explanation.\nSaab 340", "<answer>Saab 340</answer>",
])
def test_native_instruct_retains_complete_answer_without_forced_thinking(text):
    result = local_vlm.final_answer(text, "stop", profile="instruct")
    assert result == {"response": text, "finish_reason": "stop"}
    assert local_vlm.final_answer(text, "length", profile="instruct").get("error")
    assert local_vlm.final_answer(text, "stop").get("error")


def test_instruct_classification_never_fabricates_open_thinking(local_run):
    rows, output, kwargs = local_run

    class Direct(Backend):
        def infer(self, row, seed):
            native = super().infer(row, seed)
            native["native_response"]["text"] = "Saab 340"
            return native

    result = local_vlm.run_local(rows, output, profile="instruct", backend_factory=Direct,
                                 task_contract={"task": "classification"}, **kwargs)
    assert all(row["classification_response"] == row["response"] == "Saab 340"
               and not row.get("error") for row in result.values())
