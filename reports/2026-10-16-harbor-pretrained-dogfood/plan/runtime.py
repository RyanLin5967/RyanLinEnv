"""Trusted Harbor entrypoint draft; no model, network or process work on import."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ARTIFACTS = Path("/logs/artifacts")
TRAINING = ARTIFACTS / "training"
MODEL = Path("/opt/pinned-model")
GIB = 1024**3


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def checked_builder():
    from vision_rl import vlm_local_task_builder as builder

    pins = read(HERE / "runtime-pins.json")
    package = Path(builder.__file__).parent
    observed = {p.name: sha(p) for p in package.glob("*.py")}
    if observed != pins["package_sha256"]:
        raise ValueError("Installed complete frozen package differs")
    if (
        sha(__file__) != pins["runtime_sha256"]
        or sha(HERE / "aircraft_cpu_loader.py") != pins["loader"]["sha256"]
    ):
        raise ValueError("Trusted runtime/loader source differs")
    return builder, pins


def loader_module():
    spec = importlib.util.spec_from_file_location(
        "harbor_cpu_loader", HERE / "aircraft_cpu_loader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(argv):
    result = {}
    while argv:
        name, *argv = argv
        if not name.startswith("--") or name in result:
            raise ValueError("Expected unique native long options")
        if name == "--train-only":
            value = True
        else:
            if not argv or argv[0].startswith("--"):
                raise ValueError("Missing native argument value")
            value, *argv = argv
        result[name] = value
    return result


def checked_native_argv(command, builder, contract, *, probe=False):
    if command[:4] != [sys.executable, "-I", "-m", "vision_rl.vlm_local_training"]:
        raise ValueError("Only the exact frozen native module may be launched")
    actual = options(command[4:])
    stage = actual.get("--stage-id")
    permitted = {"sft-resource-probe"} if probe else {"rl-exposure", "sft-exposure", "sft-time"}
    if stage not in permitted:
        raise ValueError("Native stage is outside this trusted task")
    mode = "grpo" if stage == "rl-exposure" else "sft"
    work = TRAINING / ("sft-probe" if probe else "rl" if mode == "grpo" else "sft")
    expected = {
        "--train": "/app/data/train.jsonl",
        "--data-spec": "/app/data_spec.json",
        "--output": str(work),
        "--model-directory": str(MODEL),
        "--device": "cpu",
        "--train-only": True,
        "--mode": mode,
        "--stage-id": stage,
        "--max-steps": str(contract["control_step_cap"]),
        "--max-active-seconds": str(contract["max_active_seconds"]),
        "--learning-rate": str(contract["learning_rate"]),
        "--beta": str(contract["beta"]),
        "--seed": str(contract["seed"]),
        "--save-steps": "1",
        "--init-adapter": "/app/initial/final_adapter",
        **{"--" + k.replace("_", "-"): str(v) for k, v in contract["profile"].items()},
    }
    if stage == "sft-time":
        rl = read(TRAINING / "training-evidence.json")["arms"]["rl-exposure"]["receipt"]
        prior = builder.trainer.read_endpoint_receipt(work, expected_stage_id="sft-exposure")
        expected["--stop-after-active-seconds"] = str(rl["committed_active_seconds"])
        expected["--resume-from-checkpoint"] = str(work / prior["checkpoint"]["path"])
    else:
        expected["--stop-after-steps"] = "1" if probe else str(contract["max_steps"])
    if actual != expected:
        raise ValueError("Native argv differs from exact trusted contract/endpoint boundary")
    return stage


def output_size():
    total = 0
    for path in ARTIFACTS.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


def cgroup_memory(root=Path("/sys/fs/cgroup")):
    """Read this container's cgroup; host swap is separately recorded telemetry."""
    result = {}
    for name in (
        "memory.current",
        "memory.max",
        "memory.peak",
        "memory.swap.current",
        "memory.swap.max",
    ):
        try:
            value = (root / name).read_text().strip()
            result[name] = int(value) if value != "max" else value
        except (OSError, ValueError):
            result[name] = None
    try:
        values = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
        result["memory.stat"] = {
            name: int(values[name]) for name in ("anon", "file", "kernel") if name in values
        }
    except (OSError, ValueError):
        result["memory.stat"] = None
    return result


def require_cgroup_memory(memory, limits):
    """Fail closed unless the kernel enforces the exact container/no-swap bounds."""
    expected = {
        "memory.max": limits["docker_memory_bytes"],
        "memory.swap.max": limits["container_swap_bytes"],
        "memory.swap.current": limits["container_swap_bytes"],
    }
    if any(
        type(memory.get(key)) is not int or memory[key] != value for key, value in expected.items()
    ):
        raise RuntimeError("Container memory/swap contract missing or violated")
    current = memory.get("memory.current")
    if type(current) is not int or not 0 <= current <= limits["docker_memory_bytes"]:
        raise RuntimeError("Container memory usage missing or exceeds its bound")


def guarded(command, receipt_directory, wall_seconds):
    import psutil

    if psutil.virtual_memory().available < 13 * GIB:
        raise RuntimeError("Prelaunch memory guard; no model child started")
    initial_cgroup = cgroup_memory()
    limits = {"docker_memory_bytes": 13 * GIB, "container_swap_bytes": 0}
    require_cgroup_memory(initial_cgroup, limits)
    receipt_directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HARBOR_MODEL_PARENT=str(os.getpid()))
    started, initial_swap = time.monotonic(), psutil.swap_memory().used
    receipt = {
        "status": "running",
        "argv": command,
        "peak_tree_rss_bytes": 0,
        "initial_host_swap_bytes": initial_swap,
        "initial_cgroup_memory": initial_cgroup,
        "swap_guard_scope": "container cgroup; host swap is telemetry only",
        "wall_seconds": wall_seconds,
    }
    previous = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"guard signal {signum}")

    for name in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[name] = signal.signal(name, interrupted)
    with (receipt_directory / "actor.log").open("w") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
        receipt["child_pid"] = child.pid
        try:
            process = psutil.Process(child.pid)
            while child.poll() is None:
                rss = psutil.Process().memory_info().rss
                try:
                    for member in [process, *process.children(recursive=True)]:
                        try:
                            rss += member.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                elapsed, swap = time.monotonic() - started, psutil.swap_memory().used
                free, available = (
                    shutil.disk_usage(ARTIFACTS).free,
                    psutil.virtual_memory().available,
                )
                size = output_size()
                memory = cgroup_memory()
                row = {
                    "elapsed_seconds": elapsed,
                    "tree_rss_bytes": rss,
                    "host_available_bytes": available,
                    "host_swap_growth_bytes": swap - initial_swap,
                    "cgroup_memory": memory,
                    "disk_free_bytes": free,
                    "output_bytes": size,
                }
                with (receipt_directory / "resources.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                receipt.update(row, peak_tree_rss_bytes=max(receipt["peak_tree_rss_bytes"], rss))
                write(receipt_directory / "guard.json", receipt)
                require_cgroup_memory(memory, limits)
                if rss > 12 * GIB or available < GIB:
                    raise RuntimeError("Memory resource guard")
                if free < GIB or size > 2 * GIB or elapsed > wall_seconds:
                    raise RuntimeError("Disk/output/wall resource guard")
                time.sleep(0.5)
        except BaseException as exc:
            receipt["stop_reason"] = str(exc) or type(exc).__name__
            if child.poll() is None:
                for name in previous:
                    signal.signal(name, signal.SIG_IGN)
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
        finally:
            receipt.update(exit_code=child.wait(), elapsed_seconds=time.monotonic() - started)
            receipt["status"] = (
                "completed"
                if child.returncode == 0 and not receipt.get("stop_reason")
                else "failed"
            )
            write(receipt_directory / "guard.json", receipt)
            for name, handler in previous.items():
                signal.signal(name, handler)
    if receipt["status"] != "completed":
        raise RuntimeError("Guarded phase failed; no retry")
    return subprocess.CompletedProcess(command, 0)


# Per-stage wall caps, each >= 2x the need measured in local v2 (plan a2cd71df); see
# timing.json. The draft's flat 1800 s killed rl-exposure at 1801 s during its post-train
# reload replay. sft-time keeps the draft's 3600 s; the verifier keeps 1800 s.
STAGE_WALL_SECONDS = {
    "sft-resource-probe": 2000,
    "rl-exposure": 4000,
    "sft-exposure": 2000,
    "sft-time": 3600,
}


def controller(probe=False):
    builder, pins = checked_builder()
    if sha("/app/contract.json") != pins["contract_sha256"]:
        raise ValueError("Private training contract differs")
    contract, profile = builder._read_contract(Path("/app/contract.json"))
    original = builder.subprocess

    def launch(command, *, check):
        if check is not True:
            raise ValueError("Native subprocess must be checked")
        stage = checked_native_argv(command, builder, contract, probe=probe)
        return guarded(
            [sys.executable, "-I", __file__, "native", *command[4:]],
            ARTIFACTS / "runtime" / stage,
            STAGE_WALL_SECONDS[stage],
        )

    # Change only this trusted builder module's dispatcher, not stdlib subprocess.
    builder.subprocess = SimpleNamespace(run=launch)
    try:
        if not probe:
            return builder.run_training()
        builder._check_runtime()
        builder._check_base(contract, MODEL)
        # The resource-only arm must consume the same authored bytes as the main
        # task; checking them does not allocate the pretrained model.
        train, spec_path = Path("/app/data/train.jsonl"), Path("/app/data_spec.json")
        if sha(train) != contract["train_sha256"] or sha(spec_path) != contract["data_spec_sha256"]:
            raise ValueError("SFT probe inputs changed")
        spec = builder.trainer.load_spec(spec_path)
        rows = builder.trainer.load_classification(
            train,
            "train",
            spec["labels"],
            preprocessing_id=spec.get("preprocessing_id"),
            require_all_classes=False,
        )
        if len(rows) != 4 or builder._pixel_identity(rows) != contract["train_pixels_sha256"]:
            raise ValueError("SFT probe TRAIN fixture changed")
        _, digest = builder.read_adapter(
            Path("/app/initial/final_adapter/adapter_model.safetensors")
        )
        if digest != contract["initial_weights_sha256"] or (
            sha("/app/initial/local_training.json") != contract["initial_record_sha256"]
        ):
            raise ValueError("SFT probe initialization changed")
        builder._invoke_training(
            Path("/app/data/train.jsonl"),
            Path("/app/data_spec.json"),
            TRAINING / "sft-probe",
            profile=profile,
            model_directory=MODEL,
            device="cpu",
            train_only=True,
            max_steps=contract["control_step_cap"],
            max_active_seconds=contract["max_active_seconds"],
            learning_rate=contract["learning_rate"],
            beta=contract["beta"],
            seed=contract["seed"],
            save_steps=1,
            init_adapter=Path("/app/initial/final_adapter"),
            mode="sft",
            stop_after_steps=1,
            stage_id="sft-resource-probe",
        )
        receipt = builder.trainer.read_endpoint_receipt(
            TRAINING / "sft-probe", expected_stage_id="sft-resource-probe"
        )
        if receipt["global_step"] != 1:
            raise ValueError("SFT resource probe did not commit exactly one update")
        write(
            ARTIFACTS / "sft-resource-probe.json",
            {"scope": "resource-only TRAIN diagnostic", "receipt": receipt},
        )
        return receipt
    finally:
        builder.subprocess = original


def child(stage, argv):
    parent = int(os.environ.get("HARBOR_MODEL_PARENT", "0"))
    if parent < 2 or parent != os.getppid():
        raise RuntimeError("Model child requires its live resource supervisor")

    def watch():
        while True:
            if os.getppid() != parent:
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(15)
                os.kill(os.getpid(), signal.SIGKILL)
                return
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()
    builder, pins = checked_builder()
    private_contract = "/app/contract.json" if stage == "native" else "/tests/contract.json"
    if sha(private_contract) != pins["contract_sha256"]:
        raise ValueError("Private phase contract differs")
    if stage == "native":
        contract, _ = builder._read_contract(Path("/app/contract.json"))
        role = checked_native_argv(
            [sys.executable, "-I", "-m", "vision_rl.vlm_local_training", *argv],
            builder,
            contract,
            probe=options(argv).get("--stage-id") == "sft-resource-probe",
        )
        receipt_path = ARTIFACTS / "runtime" / role / "loader.json"
    else:
        receipt_path = Path("/logs/verifier/loader.json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with loader_module().scoped_loader(
        MODEL, receipt_path, expected_sha256=pins["loader"]["sha256"]
    ):
        if stage == "native":
            sys.argv = ["vision_rl.vlm_local_training", *argv]
            # The frozen trainer ends with `raise SystemExit(main())`. A clean exit must
            # leave this block normally so the pinned loader records `completed`; any
            # other exit code still propagates and the loader records it as failed.
            try:
                runpy.run_module(
                    "vision_rl.vlm_local_training", run_name="__main__", alter_sys=True
                )
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise
        else:
            report = builder.verify_submission()
            if "error" in report:
                raise RuntimeError("Independent numeric grader rejected submission")


def main():
    stage, *argv = sys.argv[1:]
    if stage in ("train", "sft-probe") and not argv:
        controller(stage == "sft-probe")
    elif stage == "verify" and not argv:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        guarded(
            [sys.executable, "-I", __file__, "verify-child"], Path("/logs/verifier/runtime"), 1800
        )
    elif stage in ("native", "verify-child"):
        child(stage, argv)
    else:
        raise ValueError("No arbitrary runtime command is permitted")


if __name__ == "__main__":
    main()
