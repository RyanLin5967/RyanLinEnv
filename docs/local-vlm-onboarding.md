# Local VLM training: Qwen2-VL, TRL and Harbor

This path trains a pretrained vision-language model with released TRL token GRPO and completion-only SFT. The durable Aircraft recipe is a **TRAIN-only controlled study**, with one common SFT initialization, RL, and three SFT controls. It does not yet implement a complete held-out benchmark study or establish a frontier-model win. The [research and measured evidence](local-vlm-rl-design.md) describe the current limits.

Actual FP32 MPS evidence includes four GRPO updates, three groups with correct-versus-valid-wrong reward variation, 112 changed LoRA tensors, and exact four-step versus fresh-process two-plus-two continuation parity. Both RL and SFT retained 3/4 greedy TRAIN accuracy; SFT had slightly lower final gold-answer NLL. BF16 was 2.044× faster in that diagnostic but failed its frozen log-probability parity threshold: 0.0724754 error versus 0.02 allowed. **FP32 remains selected; no held-out RL improvement is measured.**

## Install an isolated environment

Run from the repository root with Python 3.12 and `uv` installed. Keep this environment separate from running detector jobs. The common dependency lock contains hashes; torch uses the platform-specific wheel source. This adapter supports one CPU or Apple Silicon MPS process. Its older CUDA counterpart is a different, unvalidated training path.

```sh
uv venv --python 3.12 --seed .venv-vlm-local
case "$(uname -s)" in
  Darwin) .venv-vlm-local/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 ;;
  Linux) .venv-vlm-local/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu ;;
esac
.venv-vlm-local/bin/python -m pip install --no-deps --require-hashes -r requirements-vlm-local.lock
.venv-vlm-local/bin/python -m pip install --no-deps --no-build-isolation .
.venv-vlm-local/bin/python -m pip check
```

The following examples use MPS, the backend of the executed Aircraft mechanism study. For CPU, set `VLM_DEVICE=cpu` before creating your study copy below; that is a separately recorded experiment. Preflight checks versions, reviewed TRL source hashes and backend availability without loading model weights.

```sh
export VLM_DEVICE=mps
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_ENABLE_MPS_FALLBACK=0
.venv-vlm-local/bin/python -m vision_rl.vlm_local_training --preflight --device "$VLM_DEVICE"
```

## Download one pinned model copy

The [source manifest](../models/qwen2-vl-2b/source.json) pins `Qwen/Qwen2-VL-2B-Instruct` revision `895c3a49bc3fa70a340399125c650a463535e71c`: 13 files totaling 4,429,621,382 bytes. The downloader verifies sizes and immutable hashes, preserves interrupted downloads and reserves free space. It requires no API key. Keep the source manifest beside the weights: the trainer verifies it and every listed model file before loading, with remote code disabled.

```sh
.venv-vlm-local/bin/python scripts/download_pinned_model.py \
  --manifest models/qwen2-vl-2b/source.json \
  --output data/models/Qwen2-VL-2B-Instruct-895c3a49 \
  --status runs/vlm-download/status.json --min-free-gib 10
cp models/qwen2-vl-2b/source.json data/models/Qwen2-VL-2B-Instruct-895c3a49/model-source.json
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## Prepare the full vocabulary and fixed TRAIN subsets

Aircraft images are for non-commercial research and are not bundled. These commands acquire the already frozen four-per-class selection: 400 native TRAIN images, all 100 variant labels, no validation or test images. The pinned mirror's author-image equivalence and full cross-split pixel audit remain separate requirements before held-out selection; see [data preparation](fgvc-aircraft-image-preparation-2026-10-16.md).

```sh
.venv-vlm-local/bin/python scripts/prepare_aircraft.py --download-metadata \
  --archive runs/aircraft-audit/annotations.tar.gz --output runs/aircraft-audit/audit.json
.venv-vlm-local/bin/python scripts/prepare_aircraft_fourshot.py \
  --metadata-archive runs/aircraft-audit/annotations.tar.gz
.venv-vlm-local/bin/python scripts/fetch_aircraft_fourshot.py \
  --metadata-archive runs/aircraft-audit/annotations.tar.gz
```

Manifests accept **only** `id`, `split`, `image`, `label`; image paths resolve relative to the manifest. Labels must match the complete ordered vocabulary in `data_spec.json`. Its required fields are `schema_version: 1`, `benchmark`, `question`, `labels`, `split_policy`, `preprocessing`, and `sources` with URL/SHA-256 pairs. Aircraft additionally uses `preprocessing_id: "fgvc-aircraft-rgb-remove-bottom20-v1"`. Point to original JPEGs: the shared loader decodes RGB and removes the bottom 20 rows exactly once, before each model's processor. Do not provide pre-cropped images, per-image prompts or a reduced label vocabulary.

Create the prospectively selected 32-image TRAIN assay by joining its IDs to the 400; preserve the original image-path base. Also create a local copy of the published controlled recipe and its exact profile. Use a new output directory for each experiment.

```sh
.venv-vlm-local/bin/python - <<'PY'
import json, os
from pathlib import Path
data = Path("data/fgvc-aircraft-2013b/candidates/fourshot-seed791")
out = Path("runs/aircraft-inputs"); out.mkdir(parents=True, exist_ok=False)
rows = {r["id"]: r for r in map(json.loads, (data / "train400.jsonl").read_text().splitlines())}
selected = json.loads(Path("docs/aircraft-rl-protocol-candidate-v1.json").read_text())["mechanism_gate"]["rows"]
assay = []
for expected in selected:
    row = dict(rows[expected["id"]]); assert row["label"] == expected["label"] and row["split"] == "train"
    row["image"] = str((data / row["image"]).resolve()); assay.append(row)
(out / "train32.jsonl").write_text("".join(json.dumps(r) + "\n" for r in assay))
study = json.loads(Path("recipes/aircraft-controlled400-v1.json").read_text())
study["device"] = os.environ["VLM_DEVICE"]
(out / "study.json").write_text(json.dumps(study, indent=2) + "\n")
(out / "profile.json").write_text(json.dumps(study["profile"], indent=2) + "\n")
PY
```

## Train the common initialization, then run the controlled study

The warm start is a separate SFT phase on the same 400 images. Its gain belongs to SFT. Every subsequent arm and the GRPO KL reference starts from its identical saved weights. The selected profile is answer-only, FP32, language Q/V LoRA rank 8/alpha 16, four completions per sampled image, 50,176–200,704 image pixels and at most 256 completion tokens. Reward is exactly canonical task correctness with EOS termination; there is no format bonus or reasoning claim.

```sh
.venv-vlm-local/bin/python -m vision_rl.vlm_local_training \
  --mode sft --train-only --device "$VLM_DEVICE" \
  --train data/fgvc-aircraft-2013b/candidates/fourshot-seed791/train400.jsonl \
  --data-spec data/fgvc-aircraft-2013b/candidates/fourshot-seed791/data_spec.json \
  --model-directory data/models/Qwen2-VL-2B-Instruct-895c3a49 \
  --output runs/aircraft-common-warm --response-format answer_only --dtype float32 \
  --lora-rank 8 --min-pixels 50176 --max-pixels 200704 --max-prompt-tokens 2048 \
  --max-completion-length 256 --num-generations 4 --temperature 1 \
  --max-steps 400 --max-active-seconds 1800 --learning-rate 0.0001 --save-steps 100 --seed 42
```

Confirm `local_training.json` reports `status: completed` **and** 400 committed steps in `endpoint_receipt.json`; reaching the time cap earlier is not the complete one-pass warm start. Preserve failed or incomplete runs. Hardware can change runtime, so freeze an adequate new budget before a new run instead of extending an existing trajectory's cap.

```sh
.venv-vlm-local/bin/python - <<'PY'
from pathlib import Path
from vision_rl.vlm_local_training import read_endpoint_receipt
assert read_endpoint_receipt(Path("runs/aircraft-common-warm"))["global_step"] == 400
PY
.venv-vlm-local/bin/python -m vision_rl.aircraft_recipe prepare \
  --train data/fgvc-aircraft-2013b/candidates/fourshot-seed791/train400.jsonl \
  --assay runs/aircraft-inputs/train32.jsonl \
  --data-spec data/fgvc-aircraft-2013b/candidates/fourshot-seed791/data_spec.json \
  --initial-adapter runs/aircraft-common-warm/final_adapter \
  --model-directory data/models/Qwen2-VL-2B-Instruct-895c3a49 \
  --study runs/aircraft-inputs/study.json --output runs/aircraft-plan
.venv-vlm-local/bin/python -m vision_rl.aircraft_recipe run \
  --plan runs/aircraft-plan/plan.json --state-dir runs/aircraft-study --background
.venv-vlm-local/bin/python -m vision_rl.aircraft_recipe status --state-dir runs/aircraft-study
# After an interruption, retain the original plan, inputs and installed package:
.venv-vlm-local/bin/python -m vision_rl.aircraft_recipe resume --state-dir runs/aircraft-study --background
```

The example freezes RL at LR 1e-5 and 400 exposures, with three SFT learning rates: 1e-5, 5e-5 and 1e-4. Each SFT arm publishes its equal-exposure endpoint, then resumes its own optimizer trajectory to match RL's committed active time. Original caps remain 400 RL steps, 6,000 SFT steps and 10,800 active seconds per arm. Failure to reach a matched endpoint stays visible. One RL image produces four completions; it does not count as four distinct image exposures. Sampler order can differ after the first epoch.

Active time includes generation, rewards, reference inference, backward and optimizer work; it excludes data preparation, evaluation and checkpoint IO. Boundary overshoot and observed discarded work are reported separately. It is not a cap on all physical computation. Mutable optimizer state lives under `work/<arm>`; immutable exports and checked receipts live under `setup/<arm>-exposure/published` and `setup/<arm>-time/published`. Read `setup/comparison/published/comparison.json`, including every arm, invalid response and cost. `awaiting_final_evaluation` means the diagnostic evidence is complete, not benchmark success: achievement flags stay false and test remains unused. No full native-validation/test execution command exists in this compiler.

Resume verifies model, data, source, environment, optimizer, scheduler, reference, RNG, actual exposure and timing identities. Missing or changed checkpoint files fail explicitly. Use the same installed package; upgrading or reinstalling it during a study can invalidate its source hash. For a directly launched warmup, rerun its exact original command with `--resume-from-checkpoint runs/aircraft-common-warm/checkpoint-N`, naming an existing complete checkpoint. A completed endpoint is reused, not retrained.

## Package the separate CPU Harbor integration task

The Linux arm64 image actually built in **63.293 seconds** and passed network-disabled dependency preflight in **2.954 seconds** (`runs/vlm-docker-build-v1/evidence.json`). That run loaded no model. **Pretrained VLM training and grading inside Harbor remain pending.** This task requests 2 CPUs and 24 GiB RAM; a 4 GiB Docker VM is insufficient, and macOS containers cannot use host MPS.

```sh
docker build -f Dockerfile.vlm -t vision-rl:vlm-local .
docker run --rm --network none vision-rl:vlm-local
UV_PROJECT_ENVIRONMENT=.venv-harbor uv sync --locked --extra harbor
```

Supply `runs/vlm-smoke/train.jsonl` and `runs/vlm-smoke/validation.jsonl`, each with 1–4 strict rows and original image paths. The verification rows must come from separate development validation, with disjoint IDs and model-visible pixels; never relabel TRAIN or use native test. Retain the warmup's exact `data_spec.json` and complete vocabulary. The [data preparation guide](fgvc-aircraft-image-preparation-2026-10-16.md) describes the full integrity gate before native validation manifests become available.

After that gate has produced `development/validation.jsonl`, make the tiny manifests without changing image-path interpretation:

```sh
.venv-vlm-local/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("data/fgvc-aircraft-2013b")
out = Path("runs/vlm-smoke"); out.mkdir(parents=True, exist_ok=False)
for source, name in [(root / "candidates/fourshot-seed791/train400.jsonl", "train"),
                     (root / "development/validation.jsonl", "validation")]:
    rows = [json.loads(line) for line in source.read_text().splitlines()][:4]
    for row in rows:
        row["image"] = str((source.parent / row["image"]).resolve())
    (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
PY
```

```sh
.venv-vlm-local/bin/python -m vision_rl.vlm_local_task_builder build \
  --train runs/vlm-smoke/train.jsonl --verification runs/vlm-smoke/validation.jsonl \
  --data-spec data/fgvc-aircraft-2013b/candidates/fourshot-seed791/data_spec.json \
  --initial-adapter runs/aircraft-common-warm/final_adapter \
  --model-directory data/models/Qwen2-VL-2B-Instruct-895c3a49 \
  --profile runs/aircraft-inputs/profile.json --output runs/vlm-smoke/task
.venv-harbor/bin/harbor run -p "$PWD/runs/vlm-smoke/task" \
  -a vision_rl.harbor_agent:DetectionTrainingAgent \
  --job-name vlm-local-smoke --jobs-dir "$PWD/runs/vlm-smoke/jobs" -n 1
```

The builder binds the host model directory read-only; the Docker daemon must see that path. It packages TRAIN images separately from verification images. This tiny task runs at most four GRPO updates and **one** SFT control at equal exposure and time; the larger native recipe has three controls. Training is followed by a fresh offline verifier accepting only `adapter_model.safetensors`: at most 16 MiB, exactly 112 finite FP32 tensors with fixed rank-8 Q/V shapes. The verifier reconstructs LoRA alpha 16/dropout 0, processor, prompt and generation settings from trusted inputs. Submitted configuration, code, tokenizer and pickle never define grading behavior. The scalar reward is greedy mean-class accuracy on the tiny development sample, not a public benchmark result.

If preflight fails, correct the isolated dependency/backend mismatch before allocating the model. If download verification fails, retain the partial directory and inspect `status.json`/`model-integrity.json`; do not replace the pinned revision. If training shows all-zero or all-correct reward groups, inspect `reward_groups.jsonl` and actual gradients instead of treating completion as learning. Do not repair answers, manufacture reward variation or silently change frozen inputs to make a study pass.
