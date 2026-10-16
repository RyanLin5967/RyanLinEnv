"""No public model: scoped alias boundaries and a tiny real BF16-to-FP32 load."""

import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "aircraft_cpu_loader", Path(__file__).parents[1] / "scripts/aircraft_cpu_loader.py"
)
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


def fake_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(loader.sys, "platform", "linux")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(loader, "_runtime", lambda: ({}, {}))
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "opened"

    package = ModuleType("transformers")
    modeling = ModuleType("transformers.modeling_utils")
    modeling.safe_open = original
    package.modeling_utils = modeling
    monkeypatch.setitem(sys.modules, "transformers", package)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling)
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=original))
    for name in loader.POLICY["shards"]:
        (tmp_path / name).write_bytes(b"fixture only")
    return modeling, original, calls


def scope(tmp_path):
    return loader.scoped_loader(
        tmp_path, tmp_path / "receipt.json", expected_sha256=loader.sha256(loader.__file__)
    )


@pytest.mark.parametrize("failure", [False, True])
def test_context_restores_alias_environment_and_audits_calls(tmp_path, monkeypatch, failure):
    modeling, original, calls = fake_runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "old-value")
    caught = pytest.raises(InterruptedError) if failure else contextlib.nullcontext()
    with caught, scope(tmp_path):
        assert os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "1"
        assert sys.modules["safetensors"].safe_open is original
        assert (
            modeling.safe_open(
                tmp_path / loader.POLICY["shards"][0], framework="pt", backend="mmap"
            )
            == "opened"
        )
        if failure:
            raise InterruptedError("guarded termination")
    assert modeling.safe_open is original
    assert os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "old-value"
    assert calls[0][1] == {"framework": "pt", "backend": "pread"}
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["status"] == ("failed" if failure else "completed")
    assert receipt["aliases_restored"] and receipt["environment_restored"]


@pytest.mark.parametrize("case", ["path", "device", "framework", "backend", "args", "extra"])
def test_rejects_calls_outside_exact_cpu_base_scope(tmp_path, monkeypatch, case):
    modeling, original, calls = fake_runtime(tmp_path, monkeypatch)
    path = tmp_path / loader.POLICY["shards"][0]
    kwargs = {"framework": "pt"}
    args = ()
    if case == "path":
        path = tmp_path / "adapter_model.safetensors"
    elif case == "args":
        args = ("pt",)
    else:
        kwargs.update(
            {
                case: {
                    "device": "cuda:0",
                    "framework": "numpy",
                    "backend": "unknown",
                    "extra": True,
                }[case]
            }
        )
    with pytest.raises(RuntimeError, match="two pinned CPU base"), scope(tmp_path):
        modeling.safe_open(path, *args, **kwargs)
    assert calls == [] and modeling.safe_open is original


@pytest.mark.parametrize("case", ["platform", "cpu", "cuda", "hash", "missing", "receipt", "alias"])
def test_rejects_unreviewed_runtime_before_open(tmp_path, monkeypatch, case):
    modeling, original, calls = fake_runtime(tmp_path, monkeypatch)
    expected = loader.sha256(loader.__file__)
    if case == "platform":
        monkeypatch.setattr(loader.sys, "platform", "darwin")
    elif case == "cpu":
        monkeypatch.delenv("ACCELERATE_USE_CPU")
    elif case == "cuda":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    elif case == "hash":
        expected = "0" * 64
    elif case == "missing":
        (tmp_path / loader.POLICY["shards"][1]).unlink()
    elif case == "receipt":
        (tmp_path / "receipt.json").write_text("preserved")
    elif case == "alias":
        modeling.safe_open = lambda: None
    with (
        pytest.raises(RuntimeError),
        loader.scoped_loader(tmp_path, tmp_path / "receipt.json", expected_sha256=expected),
    ):
        pytest.fail("must reject before yielding")
    assert calls == []
    if case != "alias":
        assert modeling.safe_open is original


def test_versions_and_installed_sources_are_checked_without_tensor_import(tmp_path, monkeypatch):
    monkeypatch.setattr(loader.metadata, "version", lambda _: "wrong")
    with pytest.raises(RuntimeError, match="package versions"):
        loader._runtime()
    monkeypatch.setattr(loader.metadata, "version", lambda name: loader.POLICY["versions"][name])
    wrong = tmp_path / "wrong.py"
    wrong.write_text("changed")
    monkeypatch.setattr(
        loader.metadata, "distribution", lambda _: SimpleNamespace(locate_file=lambda _: wrong)
    )
    with pytest.raises(RuntimeError, match="implementation source"):
        loader._runtime()


@pytest.mark.skipif(
    os.environ.get("AIRCRAFT_LOADER_PARITY") != "1",
    reason="Run explicitly in the pinned local-VLM stack with AIRCRAFT_LOADER_PARITY=1",
)
def test_real_tiny_fp32_tensor_output_and_rng_parity(tmp_path, monkeypatch):
    # Missing/wrong dependencies in the explicitly selected job must fail, not skip.
    import numpy as np
    import safetensors.torch as safe
    import torch
    import transformers
    from transformers import PretrainedConfig, PreTrainedModel

    # This is a tiny CPU test on the available host, not Linux hardware evidence.
    actual_platform = platform.platform()
    monkeypatch.setattr(loader.sys, "platform", "linux")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    class TinyConfig(PretrainedConfig):
        model_type = "ci_loader_parity_fixture"

    class TinyModel(PreTrainedModel):
        config_class = TinyConfig
        base_model_prefix = ""

        def __init__(self, config):
            super().__init__(config)
            self.weights = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.empty(128)) for _ in range(32)]
            )
            self.post_init()

        def _init_weights(self, module):
            if isinstance(module, torch.nn.ParameterList):
                for value in module:
                    torch.nn.init.zeros_(value)

        def forward(self):
            return torch.stack([value[:8].sum() for value in self.weights])

    try:
        TinyConfig(dtype="bfloat16").save_pretrained(tmp_path)
        mapping = {}
        for shard, filename in enumerate(loader.POLICY["shards"]):
            values = {
                f"weights.{index}": torch.full((128,), (index - 16) / 8, dtype=torch.bfloat16)
                for index in range(shard * 16, (shard + 1) * 16)
            }
            safe.save_file(values, tmp_path / filename, metadata={"format": "pt"})
            mapping.update({key: filename for key in values})
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 8192}, "weight_map": mapping})
        )
        records = []
        for amended in (False, True):
            transformers.set_seed(42)
            before = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
            with scope(tmp_path) if amended else contextlib.nullcontext():
                model, info = TinyModel.from_pretrained(
                    tmp_path,
                    dtype=torch.float32,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                    output_loading_info=True,
                )
            assert not any(info.values())
            assert torch.equal(before[0], torch.get_rng_state())
            assert before[1] == random.getstate()
            current = np.random.get_state()
            assert before[2][0] == current[0] and np.array_equal(before[2][1], current[1])
            assert before[2][2:] == current[2:]
            assert all(value.dtype == torch.float32 for value in model.parameters())
            assert all(
                bool((value.detach() == (index - 16) / 8).all())
                for index, value in enumerate(model.parameters())
            )
            records.append(
                {
                    "tensor_sha256": {
                        name: hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
                        for name, value in model.named_parameters()
                    },
                    "output": model().detach().tolist(),
                }
            )
            del model
        assert records[0] == records[1]
        receipt = json.loads((tmp_path / "receipt.json").read_text())
        assert {row["file"] for row in receipt["opens"]} == set(loader.POLICY["shards"])
        proof = {
            "status": "passed",
            "scope": "tiny synthetic CPU, no pretrained model or training",
            "actual_host": actual_platform,
            "linux_guard_simulated_for_test": actual_platform.startswith("macOS"),
            "torch_python_numpy_rng_unchanged": True,
            "all_32_fp32_tensors_and_outputs_exact": True,
            "loader": loader.identity(),
            "results": records[1],
        }
        (tmp_path / "parity.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    finally:
        torch.set_num_threads(threads)
