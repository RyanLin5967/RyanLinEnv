"""Package the exact tiny TRAIN fixture; default performs no model calls/downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = Path("/Users/idide/RyanLinEnv")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(root, output, *, base_image):
    plan = json.loads((HERE / "plan.json").read_text())
    frozen = root / "runs/aircraft-objective-ablation-v1-plan/source"
    if {p.name: sha(p) for p in (frozen / "vision_rl").glob("*.py")} != plan["source_sha256"]:
        raise ValueError("The complete original37-source package must be unchanged")
    if sha(HERE / "aircraft_cpu_loader.py") != plan["loader"]["sha256"]:
        raise ValueError("Prepared loader differs from its recorded source")
    if output.exists():
        raise ValueError("Never overwrite an earlier prepared task")
    output.mkdir(parents=True)
    sys.path.insert(0, str(frozen))
    from vision_rl import vlm_local_task_builder as builder

    if (
        Path(builder.__file__).resolve()
        != (frozen / "vision_rl/vlm_local_task_builder.py").resolve()
    ):
        raise ValueError("Unexpected task builder import")
    manifests = {}
    for role, token in (("train", "train"), ("checker_train", "validation")):
        selected = [row for row in plan["provenance"] if row["role"] == role]
        rows = []
        for row in selected:
            source = row["original_record"]
            path = root / Path(source["image"]).relative_to(ORIGINAL)
            if source["split"] != "train" or sha(path) != row["encoded_sha256"]:
                raise ValueError("Every fixture must retain its exact original TRAIN identity")
            if builder.trainer.aircraft_pixel_sha256(path) != row["model_visible_pixels_sha256"]:
                raise ValueError("Original model-visible TRAIN pixels differ")
            rows.append(dict(source, image=str(path.resolve()), split=token))
        manifest = output / (role + ".jsonl")
        manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        manifests[role] = manifest
    spec = root / "data/fgvc-aircraft-2013b/candidates/fourshot-seed791/data_spec.json"
    if sha(spec) != plan["data_spec_sha256"]:
        raise ValueError("The original100-label spec changed")
    model = root / "data/models/Qwen2-VL-2B-Instruct-895c3a49"
    task = output / "task"
    bundle = builder.make_task(
        manifests["train"],
        manifests["checker_train"],
        spec,
        root / "runs/aircraft-warmstart-v1/final_adapter",
        model,
        task,
        profile=plan["profile"],
        base_image=base_image,
        timeout_sec=9900,
        **plan["training"],
    )
    environment = task / "environment"
    pins = {
        "package_sha256": plan["source_sha256"],
        "runtime_sha256": sha(HERE / "runtime.py"),
        "loader": plan["loader"],
        "plan_sha256": sha(HERE / "plan.json"),
        "contract_sha256": sha(environment / "contract.json"),
    }
    for name in ("runtime.py", "aircraft_cpu_loader.py"):
        shutil.copyfile(HERE / name, environment / name)
    write(environment / "runtime-pins.json", pins)
    write(
        environment / "training-provenance.json",
        {"native_split": "train", "rows": [r for r in plan["provenance"] if r["role"] == "train"]},
    )
    write(
        task / "tests/fixture-provenance.json",
        {
            "native_split": "train",
            "checker_api_role_only": "validation",
            "rows": plan["provenance"],
        },
    )
    original_scripts = output / "original-generated-scripts"
    original_scripts.mkdir()
    for source in (environment / "train.sh", task / "tests/test.sh"):
        shutil.copyfile(source, original_scripts / source.name)
    (environment / "train.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\npython -I /opt/harbor-runtime/runtime.py train\n"
    )
    (task / "tests/test.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\n"
        "echo 0 > /logs/verifier/reward.txt\n"
        "python -I /opt/harbor-runtime/runtime.py verify\n"
    )
    with (environment / "Dockerfile").open("a") as stream:
        stream.write(
            "COPY runtime.py aircraft_cpu_loader.py runtime-pins.json /opt/harbor-runtime/\n"
            "COPY training-provenance.json /app/training-provenance.json\n"
            'ENV ACCELERATE_USE_CPU=true CUDA_VISIBLE_DEVICES="" '
            "PYTORCH_ENABLE_MPS_FALLBACK=0 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2\n"
        )
    compose = json.loads((environment / "docker-compose.yaml").read_text())
    compose["services"]["main"].update(
        init=True,
        memswap_limit=13 * 1024**3,
        pids_limit=128,
        labels={"vision-rl.harbor-plan": sha(HERE / "plan.json")},
        logging={"driver": "local", "options": {"max-size": "2m", "max-file": "3"}},
    )
    write(environment / "docker-compose.yaml", compose)
    with (task / "instruction.md").open("a") as stream:
        stream.write(
            "\nThis prepared operational profile uses Harbor --override-memory-mb 13312, "
            "with a 12 GiB RSS guard; fit is unproven until hardware gates pass. All fixture "
            "images are native TRAIN. The checker token validation is an API role only; "
            "no native validation or TEST images are used.\n"
        )
    probe = output / "sft-resource-task"
    shutil.copytree(task, probe)
    (probe / "environment/train.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\npython -I /opt/harbor-runtime/runtime.py sft-probe\n"
    )
    write(
        output / "preparation.json",
        {
            "runtime_executed": False,
            "base_tensors_loaded": False,
            "native_validation_used": False,
            "native_test_used": False,
            "bundle": bundle,
            "plan_sha256": sha(HERE / "plan.json"),
            "runtime_pins": pins,
            "source_and_loader_identity": "private trusted image; never submission code",
        },
    )
    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-image", default="vision-rl:aircraft-cpu7946-continuation")
    args = parser.parse_args()
    print(prepare(args.root.resolve(), args.output.resolve(), base_image=args.base_image))
