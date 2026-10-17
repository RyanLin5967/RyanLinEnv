"""Derive the local plan.json from the reviewed draft, recomputing every pin it can.

Read-only except for writing ./plan.json. No Docker, Harbor or model work; the only
image work is the frozen package's model-visible pixel hash of the 7 TRAIN JPEGs.
Every recomputed hash must equal the draft's value, or this refuses to write.
runtime.py is checked as the draft's runtime.py plus runtime-systemexit.patch and
runtime-stage-walls.patch, and prepare_task.py as the draft's plus
prepare-task-agent-timeout.patch, by applying the patches in order to a temporary copy
of the draft and comparing bytes. timing.json is re-measured from the preserved v2
evidence (receipts plus file mtimes), and every wall must be >= 2x its measured need.

  .venv/bin/python -I -B runs/aircraft-harbor-dogfood-local-v4-plan/derive_plan.py
  ... derive_plan.py --local-launch-reviewed   # the reviewer's act; re-package afterwards
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.dont_write_bytecode = True  # Never write __pycache__ into the frozen source tree.

HERE = Path(__file__).resolve().parent
ROOT = Path("/Users/idide/RyanLinEnv")
DRAFT = ROOT / "runs/aircraft-harbor-dogfood-v1-plan"
AUDIT = DRAFT / "claude-audit-v1"
FROZEN = ROOT / "runs/aircraft-objective-ablation-v1-plan/source"
DRAFT_PLAN_SHA = "4e8a7ecbdb1bd7c9971ac1e7e1b2d78e15085b0c8423c759745c16152e72de33"
COPIED = ("aircraft_cpu_loader.py", "profile.json")
# Each patched file is the draft plus these patches, applied in this order.
PATCHED = {
    "runtime.py": ("runtime-systemexit.patch", "runtime-stage-walls.patch"),
    "prepare_task.py": ("prepare-task-agent-timeout.patch",),
}
PATCHES = tuple(patch for patches in PATCHED.values() for patch in patches)
LOCAL_FILES = (*COPIED, *PATCHED, *PATCHES, "local_run.py", "timing.json")
# The failed first local attempt; its plan and output are preserved, never rewritten.
V1_PLAN = ROOT / "runs/aircraft-harbor-dogfood-local-v1-plan/plan.json"
V1_PLAN_SHA = "277bb9b21778a095d0686fce09a76f4549e850cd52280d7de8fe4fac7f4e0f10"
V1_RESULT = ROOT / "runs/aircraft-harbor-dogfood-local-v1/output/dogfood-local-result.json"
V1_PROBE_JOB = (
    ROOT / "runs/aircraft-harbor-dogfood-local-v1/output/jobs/aircraft-harbor-dogfood-local-sft-probe-v1"
)
# The failed second local attempt (wall guard on rl-exposure); preserved, never rewritten.
V2_PLAN = ROOT / "runs/aircraft-harbor-dogfood-local-v2-plan/plan.json"
V2_PLAN_SHA = "a2cd71dfef43f1fee9ec47f985e66ede775f5c1c00a94de3186a12336b0dd48a"
V2_OUTPUT = ROOT / "runs/aircraft-harbor-dogfood-local-v2/output"
V2_RESULT = V2_OUTPUT / "dogfood-local-result.json"
V2_PROBE_JOB = V2_OUTPUT / "jobs/aircraft-harbor-dogfood-local-sft-probe-v2"
V2_MAIN_JOB = V2_OUTPUT / "jobs/aircraft-harbor-dogfood-local-v2"
GIB = 1024**3
SLACK = 300  # seconds an outer wall keeps above the walls it contains (v2 convention)
# In-container stage walls (runtime-stage-walls.patch), verifier wall (draft, unchanged),
# Harbor agent timeout (prepare-task-agent-timeout.patch) and the driver caps.
STAGE_WALLS = {
    "sft-resource-probe": 2000,
    "rl-exposure": 4000,
    "sft-exposure": 2000,
    "sft-time": 3600,
}
DRAFT_STAGE_WALLS = {
    "sft-resource-probe": 1800,
    "rl-exposure": 1800,
    "sft-exposure": 1800,
    "sft-time": 3600,
}
VERIFY_WALL = 1800
AGENT_TIMEOUT = (
    STAGE_WALLS["rl-exposure"] + STAGE_WALLS["sft-exposure"] + STAGE_WALLS["sft-time"] + SLACK
)
PROBE_WALL = STAGE_WALLS["sft-resource-probe"] + SLACK
MAIN_WALL = AGENT_TIMEOUT + VERIFY_WALL + SLACK
OUTER_TIMEOUT = PROBE_WALL + MAIN_WALL + SLACK


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same(observed, expected, what):
    if observed != expected:
        raise ValueError(f"Pin differs, refusing to write plan.json: {what}")


def patched_bytes(name):
    """The draft file with its declared patches applied, in order, in a temporary directory."""
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / name
        shutil.copyfile(DRAFT / name, target)
        for patch in PATCHED[name]:
            subprocess.run(
                ["patch", "-s", str(target), str(HERE / patch)],
                check=True,
                timeout=30,
                capture_output=True,
            )
        return target.read_bytes()


def failed_v1_attempt():
    """Re-read the preserved v1 evidence; refuse if it no longer says what v3 claims."""
    same(sha(V1_PLAN), V1_PLAN_SHA, "v1 local plan.json")
    result = json.loads(V1_RESULT.read_text())
    same(result["plan_sha256"], V1_PLAN_SHA, "v1 result plan sha")
    same(result["pass"], False, "v1 result pass")
    same(result["driver_error"], "ValueError: Scoped loader identity/completion differs", "v1 error")
    (loader_path,) = V1_PROBE_JOB.rglob("runtime/sft-resource-probe/loader.json")
    (guard_path,) = V1_PROBE_JOB.rglob("runtime/sft-resource-probe/guard.json")
    loader, guard = json.loads(loader_path.read_text()), json.loads(guard_path.read_text())
    same((loader["status"], loader.get("error_type")), ("failed", "SystemExit"), "v1 loader")
    same((guard["status"], guard["exit_code"]), ("completed", 0), "v1 guard")
    return {
        "plan": "runs/aircraft-harbor-dogfood-local-v1-plan/plan.json",
        "plan_sha256": V1_PLAN_SHA,
        "result": "runs/aircraft-harbor-dogfood-local-v1/output/dogfood-local-result.json",
        "result_sha256": sha(V1_RESULT),
        "driver_error": result["driver_error"],
        "probe_loader_receipt": {
            "path": str(loader_path.relative_to(ROOT)),
            "sha256": sha(loader_path),
            "status": loader["status"],
            "error_type": loader["error_type"],
        },
        "probe_guard_receipt": {
            "path": str(guard_path.relative_to(ROOT)),
            "sha256": sha(guard_path),
            "status": guard["status"],
            "exit_code": guard["exit_code"],
        },
        "preserved": "evidence only; never edited, re-run or overwritten",
    }


def _one(job, pattern):
    (path,) = job.rglob(pattern)
    return path


def _train_runtime(actor_log):
    """train_runtime from the trainer's final metrics line in actor.log."""
    (value,) = re.findall(r"'train_runtime': '([0-9.]+)'", actor_log.read_text())
    return float(value)


def _stage_timing(job, stage, training, marks):
    """Phase marks in seconds since the stage guard started, from receipts and file mtimes.

    The guard appends one resources.jsonl row per poll with elapsed_seconds, so the stage
    start is mtime(resources.jsonl) - elapsed_seconds of its last row. Every mtime is
    container-written and taken relative to that start, so host/VM clock offset cancels.
    """
    runtime = _one(job, f"runtime/{stage}/resources.jsonl").parent
    rows = (runtime / "resources.jsonl").read_text().splitlines()
    start = (runtime / "resources.jsonl").stat().st_mtime - json.loads(rows[-1])["elapsed_seconds"]
    guard = json.loads((runtime / "guard.json").read_text())
    work = runtime.parent.parent / "training" / training
    at = {name: round((work / rel).stat().st_mtime - start, 2) for name, rel in marks.items()}
    return guard, at, work, runtime


def measure_v2_timing():
    """Per-phase timing of the failed v2 run and the >= 2x wall check, from preserved evidence."""
    marks = {
        "adapter_source_written": "adapter-source.py",
        "before_train_diagnostic_written": "before_train_diagnostic_predictions.json",
        "step_1_checkpoint_committed": "checkpoint-1/active_time.json",
        "after_train_diagnostic_written": "after_train_diagnostic_predictions.json",
        "final_adapter_and_trainer_state_saved": "trainer_state.json",
    }
    probe_guard, probe_at, probe_work, probe_runtime = _stage_timing(
        V2_PROBE_JOB,
        "sft-resource-probe",
        "sft-probe",
        {
            **marks,
            "reload_replay_written": "reloaded_train_diagnostic_predictions.json",
            "endpoint_receipt_written": "endpoint_receipt.json",
        },
    )
    rl_guard, rl_at, rl_work, rl_runtime = _stage_timing(V2_MAIN_JOB, "rl-exposure", "rl", marks)
    (invocation,) = (probe_work / "active_time").glob("invocation-*.json")
    probe_active = json.loads(invocation.read_text())["completed_active_seconds"]
    rl_record = json.loads((rl_work / "local_training.json").read_text())
    rl_active = rl_record["active_time_audit"]["committed_active_seconds"]
    same(rl_record["active_time_audit"]["committed_global_step"], 1, "v2 rl committed step")
    same(rl_record["status"], "interrupted", "v2 rl status")
    probe_train_runtime = _train_runtime(probe_runtime / "actor.log")
    rl_train_runtime = _train_runtime(rl_runtime / "actor.log")
    # The RL child was killed inside the audit-only replay of the after-train diagnostic
    # (_evaluate on the reloaded adapter); the actor.log traceback names that call.
    same(
        "_, reload_predictions = _evaluate(" in (rl_runtime / "actor.log").read_text(),
        True,
        "v2 rl killed during the reload replay",
    )
    same((probe_guard["status"], probe_guard["exit_code"]), ("completed", 0), "v2 probe guard")
    same(
        (rl_guard["status"], rl_guard["exit_code"], rl_guard["stop_reason"]),
        ("failed", -2, "Disk/output/wall resource guard"),
        "v2 rl guard",
    )
    same((probe_guard["wall_seconds"], rl_guard["wall_seconds"]), (1800, 1800), "v2 stage walls")

    def span(at, first, second):
        return round(at[second] - at[first], 2)

    probe_total = round(probe_guard["elapsed_seconds"], 2)
    rl_killed = round(rl_guard["elapsed_seconds"], 2)
    rl_after_eval = span(rl_at, "step_1_checkpoint_committed", "after_train_diagnostic_written")
    rl_need = round(rl_killed + rl_after_eval, 2)
    sft_fixed = round(probe_guard["elapsed_seconds"] - probe_train_runtime, 2)
    # sft-time resumes sft-exposure's checkpoint and stops once cumulative committed active
    # time reaches the RL arm's, so its new steps' active time is below the RL active time.
    sft_time_steps = math.ceil(rl_active / probe_active)
    sft_time_need = round(sft_fixed + rl_active * probe_train_runtime / probe_active, 2)
    verify_bound = probe_at["before_train_diagnostic_written"]
    need = {
        "sft-resource-probe": probe_total,
        "rl-exposure": rl_need,
        "sft-exposure": probe_total,
        "sft-time": sft_time_need,
    }
    agent_need = round(need["rl-exposure"] + need["sft-exposure"] + need["sft-time"], 2)
    walls = {}
    for stage, seconds in need.items():
        if STAGE_WALLS[stage] < 2 * seconds:
            raise ValueError(f"{stage} wall {STAGE_WALLS[stage]} < 2x measured need {seconds}")
        walls[stage] = {
            "draft_wall_seconds": DRAFT_STAGE_WALLS[stage],
            "v3_wall_seconds": STAGE_WALLS[stage],
            "need_seconds": seconds,
            "draft_ratio": round(DRAFT_STAGE_WALLS[stage] / seconds, 3),
            "v3_ratio": round(STAGE_WALLS[stage] / seconds, 3),
        }
    if VERIFY_WALL < 2 * verify_bound:
        raise ValueError("verify wall < 2x its bound")
    if AGENT_TIMEOUT < 2 * agent_need:
        raise ValueError("Harbor agent timeout < 2x the three training stages' need")
    if sft_time_steps > 32:
        raise ValueError("sft-time cannot reach the RL active time within 32 steps")
    return {
        "source": "runs/aircraft-harbor-dogfood-local-v2 (plan a2cd71df), preserved evidence",
        "instrument": (
            "derive_plan.py measure_v2_timing: guard.json and resources.jsonl receipts, "
            "local_training.json active_time_audit, active_time invocation json, actor.log "
            "train_runtime, and training/* file mtimes relative to the guard start "
            "(mtime(resources.jsonl) - last row elapsed_seconds)"
        ),
        "rule": "every wall >= 2x its measured or projected need; outer walls = inner + 300 s",
        "sft-resource-probe": {
            "guard_elapsed_seconds": probe_total,
            "guard_status": probe_guard["status"],
            "peak_tree_rss_bytes": probe_guard["peak_tree_rss_bytes"],
            "committed_active_seconds": round(probe_active, 2),
            "train_runtime_seconds": probe_train_runtime,
            "phase_marks_seconds_since_guard_start": probe_at,
            "phases_seconds": {
                "start_model_load_before_train_diagnostic": probe_at[
                    "before_train_diagnostic_written"
                ],
                "before_gold_nll_and_sft_step_1": span(
                    probe_at, "before_train_diagnostic_written", "step_1_checkpoint_committed"
                ),
                "after_train_diagnostic": span(
                    probe_at, "step_1_checkpoint_committed", "after_train_diagnostic_written"
                ),
                "save_final_adapter": span(
                    probe_at, "after_train_diagnostic_written", "final_adapter_and_trainer_state_saved"
                ),
                "post_train_audit_gold_nll_reload_replay": span(
                    probe_at, "final_adapter_and_trainer_state_saved", "reload_replay_written"
                ),
                "fixed_non_step_seconds": sft_fixed,
            },
        },
        "rl-exposure": {
            "guard_elapsed_seconds": rl_killed,
            "guard_status": rl_guard["status"],
            "guard_exit_code": rl_guard["exit_code"],
            "stop_reason": rl_guard["stop_reason"],
            "peak_tree_rss_bytes": rl_guard["peak_tree_rss_bytes"],
            "committed_active_seconds": round(rl_active, 2),
            "train_runtime_seconds": rl_train_runtime,
            "phase_marks_seconds_since_guard_start": rl_at,
            "phases_seconds": {
                "start_model_load_before_train_diagnostic": rl_at["before_train_diagnostic_written"],
                "before_gold_nll_parity_and_grpo_step_1": span(
                    rl_at, "before_train_diagnostic_written", "step_1_checkpoint_committed"
                ),
                "after_train_diagnostic": rl_after_eval,
                "save_final_adapter": span(
                    rl_at, "after_train_diagnostic_written", "final_adapter_and_trainer_state_saved"
                ),
                "post_train_audit_until_killed": round(
                    rl_guard["elapsed_seconds"] - rl_at["final_adapter_and_trainer_state_saved"], 2
                ),
            },
            "killed_during": "audit-only reload replay (_evaluate on the reloaded adapter), "
            "per the actor.log traceback",
            "projected_need_seconds": rl_need,
            "projection": (
                "elapsed at kill + one after-train-diagnostic duration, because the replay "
                "re-runs that diagnostic and had already started. Lower bound: elapsed at "
                "kill. Unverified assumption: the replay costs no more than the after-train "
                "diagnostic did"
            ),
        },
        "sft-exposure": {
            "need_seconds": probe_total,
            "basis": "same native argv as the probe except --stage-id (1 step, train-only, "
            "same post-train audit), so the v2 probe measures it. It never ran in v2",
        },
        "sft-time": {
            "need_seconds": sft_time_need,
            "target_active_seconds": round(rl_active, 2),
            "steps_to_reach_target_at_probe_step": sft_time_steps,
            "step_cap": 32,
            "basis": "probe fixed non-step seconds + RL committed active seconds x probe "
            "train_runtime/active ratio. It resumes sft-exposure's step, so its new steps' "
            "active time is below the target. It never ran in v2",
        },
        "verify": {
            "wall_seconds": VERIFY_WALL,
            "need_bound_seconds": verify_bound,
            "basis": "unmeasured: never ran in v1 or v2. verify_submission loads the model and "
            "greedily decodes <= 4 images, a subset of the probe's start + model load + "
            "8-row before-train diagnostic",
        },
        "walls": walls,
        "harbor_agent_timeout": {
            "draft_seconds": 7200,
            "v3_seconds": AGENT_TIMEOUT,
            "need_seconds": agent_need,
            "draft_ratio": round(7200 / agent_need, 3),
            "v3_ratio": round(AGENT_TIMEOUT / agent_need, 3),
            "basis": "the agent runs rl-exposure, sft-exposure and sft-time in series; v3 "
            "also encloses the sum of their walls plus 300 s",
        },
        "driver": {
            "probe_wall_seconds": PROBE_WALL,
            "main_wall_seconds": MAIN_WALL,
            "readme_outer_timeout_seconds": OUTER_TIMEOUT,
        },
    }


def failed_v2_attempt():
    """Re-read the preserved v2 evidence; refuse if it no longer says what v3 claims."""
    same(sha(V2_PLAN), V2_PLAN_SHA, "v2 local plan.json")
    result = json.loads(V2_RESULT.read_text())
    same(result["plan_sha256"], V2_PLAN_SHA, "v2 result plan sha")
    same(result["pass"], False, "v2 result pass")
    same(
        result["driver_error"],
        "ValueError: A native model phase failed its actual resource gate",
        "v2 error",
    )
    receipts = {}
    keep = ("error_type", "exit_code", "stop_reason", "wall_seconds", "elapsed_seconds")
    for key, job, stage in (
        ("probe", V2_PROBE_JOB, "sft-resource-probe"),
        ("rl_exposure", V2_MAIN_JOB, "rl-exposure"),
    ):
        for kind in ("loader", "guard"):
            path = _one(job, f"runtime/{stage}/{kind}.json")
            row = json.loads(path.read_text())
            receipts[f"{key}_{kind}_receipt"] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(path),
                "status": row["status"],
                **{name: row[name] for name in keep if name in row},
            }
    same(receipts["probe_loader_receipt"]["status"], "completed", "v2 probe loader")
    same(receipts["probe_guard_receipt"]["status"], "completed", "v2 probe guard")
    same(
        receipts["rl_exposure_guard_receipt"]["stop_reason"],
        "Disk/output/wall resource guard",
        "v2 rl guard",
    )
    return {
        "plan": "runs/aircraft-harbor-dogfood-local-v2-plan/plan.json",
        "plan_sha256": V2_PLAN_SHA,
        "result": "runs/aircraft-harbor-dogfood-local-v2/output/dogfood-local-result.json",
        "result_sha256": sha(V2_RESULT),
        "driver_error": result["driver_error"],
        **receipts,
        "systemexit_fix_validated": "v2 probe loader receipt completed under runtime-systemexit.patch",
        "preserved": "evidence only; never edited, re-run or overwritten",
    }


def verify_draft_pins(draft):
    """Recompute every draft hash that has a local source; return the instrument log."""
    log = {}
    frozen = {p.name: sha(p) for p in (FROZEN / "vision_rl").glob("*.py")}
    same(frozen, draft["source_sha256"], "frozen 37-file source")
    log["source_sha256"] = "37 files recomputed"
    for name, value in draft["host_source_sha256"].items():
        same(sha(ROOT / name), value, name)
    log["host_source_sha256"] = "6 files recomputed"
    same(sha(ROOT / "uv.lock"), draft["host_dependencies"]["uv_lock_sha256"], "uv.lock")
    same(
        sha(ROOT / "docs/aircraft-cpu-experiment-plan.json"),
        draft["first4_descriptor_sha256"],
        "descriptor",
    )
    same(
        sha(ROOT / "runs/aircraft-objective-ablation-v1-plan/plan.json"),
        draft["first4_plan_sha256"],
        "first4 plan",
    )
    same(
        sha(ROOT / "scripts/aircraft_ci_experiment.py"),
        draft["resource_helper_sha256"],
        "resource helper",
    )
    spec = ROOT / "data/fgvc-aircraft-2013b/candidates/fourshot-seed791/data_spec.json"
    same(sha(spec), draft["data_spec_sha256"], "data_spec")
    same(sha(ROOT / "scripts/aircraft_cpu_loader.py"), draft["loader"]["sha256"], "scripts loader")
    descriptor = json.loads((ROOT / "docs/aircraft-cpu-experiment-plan.json").read_text())
    same(descriptor["loader"], draft["loader"], "descriptor loader policy")
    for name, value in descriptor["build_files"].items():
        same(sha(ROOT / name), value, "build file " + name)
    log["build_files"] = "3 files recomputed against docs/aircraft-cpu-experiment-plan.json"
    sys.path.insert(0, str(FROZEN))
    from vision_rl.aircraft_data import aircraft_pixel_sha256

    for row in draft["provenance"]:
        image = Path(row["original_record"]["image"])
        same(row["original_record"]["split"], "train", "split " + image.name)
        same(sha(image), row["encoded_sha256"], "encoded " + image.name)
        same(
            aircraft_pixel_sha256(image), row["model_visible_pixels_sha256"], "pixels " + image.name
        )
    log["provenance"] = (
        "7 JPEG and 7 model-visible pixel hashes recomputed (frozen aircraft_pixel_sha256)"
    )
    for key, local in (
        (
            "public_source_bundle",
            ROOT / "runs/aircraft-ci-preparation/aircraft-cpu7946-inputs-v1.tar",
        ),
        (
            "public_warm_bundle",
            ROOT / "runs/aircraft-warm-release-v1/aircraft-common-warm-f05840be.tar.gz",
        ),
    ):
        same(sha(local), draft[key]["sha256"], key)
        same(local.stat().st_size, draft[key]["bytes"], key + " bytes")
    log["public_bundles"] = "both local release tarballs recomputed (not used by the local driver)"
    warm = ROOT / "runs/aircraft-warmstart-v1"
    files = draft["public_warm_bundle"]["files"]
    for relative, local in (
        (
            "final_adapter/adapter_model.safetensors",
            warm / "final_adapter/adapter_model.safetensors",
        ),
        ("final_adapter/adapter_config.json", warm / "final_adapter/adapter_config.json"),
        ("local_training.json", warm / "local_training.json"),
    ):
        same(sha(local), files["aircraft-common-warm/" + relative], "warm " + relative)
    log["warm_adapter"] = "runs/aircraft-warmstart-v1 adapter, config and record recomputed"
    model = ROOT / "data/models/Qwen2-VL-2B-Instruct-895c3a49"
    log["model_identity"] = {
        "manifest_sha256": sha(model / "model-source.json"),
        "marker_sha256": sha(model / ".pinned-download.json"),
        "revision": json.loads((model / "model-source.json").read_text())["revision"],
    }
    log["loader.source_sha256"] = (
        "installed transformers/safetensors files exist only in the image; checked at run "
        "time by assert_runtime_receipts (runtime_sources == loader.source_sha256)"
    )
    return log


def derive(reviewed):
    draft_path = DRAFT / "plan.json"
    same(sha(draft_path), DRAFT_PLAN_SHA, "draft plan.json")
    draft = json.loads(draft_path.read_text())
    for name in COPIED:
        same(sha(HERE / name), sha(DRAFT / name), "byte-identical copy " + name)
    for name in PATCHED:
        same((HERE / name).read_bytes(), patched_bytes(name), "draft plus patch " + name)
    timing = measure_v2_timing()
    same(json.loads((HERE / "timing.json").read_text()), timing, "timing.json re-measured from v2")
    previous = {"v1": failed_v1_attempt(), "v2": failed_v2_attempt()}
    log = verify_draft_pins(draft)
    identity = log.pop("model_identity")
    same(identity["revision"], "895c3a49bc3fa70a340399125c650a463535e71c", "model revision")
    same(identity["manifest_sha256"][:8], "2c20d30e", "model manifest")
    descriptor = json.loads((ROOT / "docs/aircraft-cpu-experiment-plan.json").read_text())
    plan = {
        key: draft[key]
        for key in (
            "cgroup_policy_reviewed",
            "checker_native_split",
            "checker_role_token",
            "data_spec_sha256",
            "first4_descriptor_sha256",
            "first4_plan_sha256",
            "host_dependencies",
            "host_source_sha256",
            "loader",
            "native_test_used",
            "native_validation_used",
            "profile",
            "provenance",
            "public_source_bundle",
            "public_warm_bundle",
            "repository",
            "resource",
            "resource_helper_sha256",
            "schema_version",
            "selection",
            "sft_precheck",
            "source_commit",
            "source_sha256",
            "training",
        )
    }
    plan["training"] = dict(draft["training"], max_steps=1)
    plan["resource"] = {k: v for k, v in draft["resource"].items() if k != "workflow_minutes"}
    plan.update(
        name="aircraft-harbor-dogfood-local-v4",
        created_at=str(date.today()),
        scope=(
            "local integration-only on Colima vision-harbor; native TRAIN diagnostics; "
            "not held-out, not benchmark, not frontier, not an RL-vs-SFT result"
        ),
        status="prepared locally and unexecuted; local launch "
        + ("reviewed" if reviewed else "unreviewed")
        + "; supersedes failed attempt v2 (plan a2cd71df, rl-exposure killed by the 1800 s "
        "stage wall at 1801.29 s during the post-train reload replay after GRPO update 1 "
        "committed), preserved as evidence in runs/aircraft-harbor-dogfood-local-v2; v2 "
        "superseded failed attempt v1 (plan 277bb9b2, driver_error, loader receipt "
        "error_type SystemExit), preserved as evidence in runs/aircraft-harbor-dogfood-local-v1",
        previous_attempt=previous,
        local_launch_reviewed=bool(reviewed),
        build_files=descriptor["build_files"],
        model_identity={
            **identity,
            "initial_weights_sha256": draft["public_warm_bundle"]["files"][
                "aircraft-common-warm/final_adapter/adapter_model.safetensors"
            ],
        },
        local={
            "platform": "darwin",
            "docker_host": "unix:///Users/idide/.colima/vision-harbor/docker.sock",
            "vm": {"profile": "vision-harbor", "cpu": 4, "memory_gib": 16, "disk_gib": 20},
            "image": "vision-rl:aircraft-cpu7946-continuation",
            "root": "runs/aircraft-harbor-dogfood-local-v4",
            "probe_job": "aircraft-harbor-dogfood-local-sft-probe-v4",
            "main_job": "aircraft-harbor-dogfood-local-v4",
            "stage_wall_seconds": STAGE_WALLS,
            "verify_wall_seconds": VERIFY_WALL,
            "agent_timeout_seconds": AGENT_TIMEOUT,
            "probe_wall_seconds": PROBE_WALL,
            "main_wall_seconds": MAIN_WALL,
            "outer_timeout_seconds": OUTER_TIMEOUT,
            "host_output_cap_bytes": 2 * GIB,
            "host_disk_floor_bytes": GIB,
            "owned_container_label": "vision-rl.harbor-plan=<sha256 of this plan.json>",
            "result": "runs/aircraft-harbor-dogfood-local-v4/output/dogfood-local-result.json",
        },
        commands={
            "harbor": [
                ".venv/bin/harbor",
                "run",
                "-p",
                "TASK",
                "--agent",
                "vision_rl.harbor_agent:DetectionTrainingAgent",
                "--job-name",
                "NAME",
                "--jobs-dir",
                "OUTPUT/jobs",
                "--n-attempts",
                "1",
                "--n-concurrent",
                "1",
                "--max-retries",
                "0",
                "--override-cpus",
                "2",
                "--override-memory-mb",
                "13312",
            ]
        },
        gates={
            "no_automatic_retries": True,
            "same_loader_identity_required": True,
            "sft_one_step_must_pass_before_full_task": True,
            "successful_reviewed_first4_parent_required": False,
            "first4_numerical_throughput_and_resource_gates_required": False,
            "local_launch_requires": [
                "local_launch_reviewed == true and cgroup_policy_reviewed == true",
                "sys.platform == darwin",
                "DOCKER_HOST == unix:///Users/idide/.colima/vision-harbor/docker.sock, DOCKER_CONTEXT unset",
                "cwd == /Users/idide/RyanLinEnv",
                "support/host/build/frozen-source pins match; prepared bundle packaged from this plan sha",
                "runs/aircraft-harbor-dogfood-local-v4/output does not exist (single use)",
                "agent timeout, probe and main driver walls enclose the walls inside them plus 300 s",
            ],
        },
        implementation_status={
            "actual_execution": False,
            "local_driver": "local_run.py; fixture tests only (test_local_plan.py)",
            "runtime_wrapper": "draft runtime.py plus runtime-systemexit.patch and runtime-stage-walls.patch",
            "packager": "draft prepare_task.py plus prepare-task-agent-timeout.patch",
        },
        success_criteria=json.loads((AUDIT / "dogfood-design.json").read_text())[
            "success_criteria"
        ],
        derived_from={
            "draft": "runs/aircraft-harbor-dogfood-v1-plan",
            "draft_plan_sha256": DRAFT_PLAN_SHA,
            "design_sha256": sha(AUDIT / "dogfood-design.json"),
            "audit_sha256": sha(AUDIT / "audit.md"),
            "verification_sha256": sha(AUDIT / "verification/verification.json"),
            "race_patch_sha256": sha(AUDIT / "run-output-size-race.patch"),
            "draft_run_py_sha256": sha(DRAFT / "run.py"),
            "fixed_run_py_sha256": sha(AUDIT / "fixed_run.py"),
        },
        changes=[
            {
                "field": "training.max_steps",
                "from": 4,
                "to": 1,
                "why": "audit F2: 4 GRPO updates project to 2612 s against runtime.py's 1800 s rl-exposure wall (measured.json); 1 update projects to about 1036 s. Local Apple-CPU speed is unmeasured.",
            },
            {
                "field": "launch_reviewed",
                "to": "removed; replaced by local_launch_reviewed=false",
                "why": "the GitHub launch gate does not apply locally; local review is a separate act",
            },
            {
                "field": "gates.successful_reviewed_first4_parent_required",
                "from": True,
                "to": False,
                "why": "dogfood-design: no GitHub API, no parent download; the SFT probe and in-container guards are the resource gates",
            },
            {
                "field": "gates.first4_numerical_throughput_and_resource_gates_required",
                "from": True,
                "to": False,
                "why": "no first4 parent locally; replaced by the same in-container runtime.py guards plus the SFT probe",
            },
            {
                "field": "resource.workflow_minutes",
                "to": "removed",
                "why": "no GitHub workflow; local wall caps are local.probe_wall_seconds and local.main_wall_seconds",
            },
            {
                "field": "support_sha256",
                "why": "run.py and workflow.yml replaced by local_run.py; profile.json now pinned too; v2 also pins runtime-systemexit.patch; v3 also pins runtime-stage-walls.patch, prepare-task-agent-timeout.patch and timing.json",
            },
            {
                "field": "commands.harbor",
                "why": ".venv/bin/harbor (Harbor 0.24.0) instead of uv run; flags unchanged",
            },
            {"field": "status/scope/name/created_at/implementation_status", "why": "local variant"},
            {
                "field": "runtime.py",
                "from": "byte-identical to the draft",
                "to": "draft runtime.py plus runtime-systemexit.patch",
                "why": "local v1 (plan 277bb9b2) failed: child() runs the frozen trainer with "
                "runpy.run_module inside scoped_loader, and the trainer ends with "
                "`raise SystemExit(main())`. A successful SFT probe (guard exit_code 0, "
                "1 committed update) therefore raised SystemExit(0) through the context "
                "manager, the loader receipt recorded status failed / error_type SystemExit, "
                "and assert_probe stopped the run. The patch catches SystemExit around "
                "run_module only: code 0 or None leaves the block normally so the loader "
                "records completed; any other code is re-raised. aircraft_cpu_loader.py is "
                "pinned and shared with CI and is unchanged. verify-child calls "
                "builder.verify_submission(), which returns a report and never raises "
                "SystemExit, so it needs no change.",
            },
            {
                "field": "name/local.root/local.probe_job/local.main_job/local.result",
                "from": "...-v2",
                "to": "...-v3",
                "why": "v1 and v2 outputs are preserved evidence of the failed attempts; v3 is a new single-use run",
            },
            {
                "field": "previous_attempt",
                "to": "{v1, v2}",
                "why": "pins the failed v1 and v2 plans, results and receipts, re-read by derive_plan.py",
            },
            {
                "field": "runtime.py STAGE_WALL_SECONDS[rl-exposure]",
                "from": 1800,
                "to": STAGE_WALLS["rl-exposure"],
                "why": "local v2 (plan a2cd71df): GRPO update 1 committed (752.76 s active, "
                "nonzero gradient) and after_train_diagnostic_predictions.json was written, then "
                "the 1800 s wall killed the stage at 1801.29 s inside the audit-only reload "
                "replay (exit -2). Projected need "
                + str(timing["walls"]["rl-exposure"]["need_seconds"])
                + " s (elapsed at kill + one after-train diagnostic; timing.json). 3600 s would "
                "be below 2x even the 1801.29 s lower bound, so the >= 2x rule gives 4000 s",
            },
            {
                "field": "runtime.py STAGE_WALL_SECONDS[sft-exposure]",
                "from": 1800,
                "to": STAGE_WALLS["sft-exposure"],
                "why": "same argv as the probe, which took "
                + str(timing["walls"]["sft-exposure"]["need_seconds"])
                + " s in v2; 1800 s is "
                + str(timing["walls"]["sft-exposure"]["draft_ratio"])
                + "x, short of the >= 2x rule",
            },
            {
                "field": "runtime.py STAGE_WALL_SECONDS[sft-resource-probe]",
                "from": 1800,
                "to": STAGE_WALLS["sft-resource-probe"],
                "why": "measured "
                + str(timing["walls"]["sft-resource-probe"]["need_seconds"])
                + " s in v2; 1800 s is "
                + str(timing["walls"]["sft-resource-probe"]["draft_ratio"])
                + "x, short of the >= 2x rule",
            },
            {
                "field": "runtime.py STAGE_WALL_SECONDS[sft-time] and the verify wall",
                "from": "3600 and 1800",
                "to": "unchanged",
                "why": "sft-time need "
                + str(timing["walls"]["sft-time"]["need_seconds"])
                + " s (probe fixed cost + RL active time; reaches 752.76 s in "
                + str(timing["sft-time"]["steps_to_reach_target_at_probe_step"])
                + " of 32 steps), so 3600 s is >= 2x. Verify is unmeasured and bounded by the "
                "probe's start + model load + 8-row diagnostic ("
                + str(timing["verify"]["need_bound_seconds"])
                + " s), so 1800 s is >= 2x",
            },
            {
                "field": "prepare_task.py timeout_sec (Harbor task.toml [agent] timeout_sec)",
                "from": 7200,
                "to": AGENT_TIMEOUT,
                "why": "the Harbor agent runs rl-exposure, sft-exposure and sft-time in series, "
                "with measured/projected need "
                + str(timing["harbor_agent_timeout"]["need_seconds"])
                + " s, so 7200 s was below 2x; it must also enclose the new stage walls "
                "(4000 + 2000 + 3600) plus 300 s, or Harbor kills before the stage guard does",
            },
            {
                "field": "local.probe_wall_seconds/local.main_wall_seconds (local_run.py caps) "
                "and the README outer timeout",
                "from": "2100/9300/11700",
                "to": f"{PROBE_WALL}/{MAIN_WALL}/{OUTER_TIMEOUT}",
                "why": "v2's construction (inner + 300 s) over the new inner walls: probe = probe "
                "stage wall + 300; main = agent timeout + verify wall + 300; outer = probe + "
                "main + 300. local_run.py require_wall_nesting refuses a plan that breaks this, "
                "and verify_bundle checks the packaged task.toml agent timeout",
            },
            {
                "field": "timing.json",
                "to": "added",
                "why": "per-phase timing of v2, re-measured by derive_plan.py from preserved "
                "evidence; derivation refuses if any wall is below 2x its need",
            },
            {
                "field": "build_files/model_identity/local/success_criteria/derived_from",
                "to": "added",
                "why": "pins checked by local_run.py and success criteria C1-C4 from dogfood-design.json",
            },
        ],
        copies={
            **{
                name: {"sha256": sha(HERE / name), "identical_to_draft": True, "change": None}
                for name in COPIED
            },
            **{
                name: {
                    "sha256": sha(HERE / name),
                    "identical_to_draft": False,
                    "draft_sha256": sha(DRAFT / name),
                    "change": list(patches),
                    "patch_sha256": {patch: sha(HERE / patch) for patch in patches},
                    "instrument": "patch -s applied in order to a temporary copy of the draft, "
                    "bytes compared",
                }
                for name, patches in PATCHED.items()
            },
        },
        pin_verification={
            "date": str(date.today()),
            "instrument": "derive_plan.py sha256 recomputation",
            **log,
        },
        support_sha256={name: sha(HERE / name) for name in LOCAL_FILES},
    )
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-launch-reviewed", action="store_true")
    args = parser.parse_args()
    timing = json.dumps(measure_v2_timing(), indent=2, sort_keys=True) + "\n"
    (HERE / "timing.json").write_text(timing)
    value = derive(args.local_launch_reviewed)
    (HERE / "plan.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(sha(HERE / "plan.json"))
