# Named open-weight baseline on native VAL: Qwen3.8-27B-8bit (pre-registered, unactivated)

The machine-readable plan is [aircraft-named-baseline-plan-v1.json](aircraft-named-baseline-plan-v1.json).
Its status is **UNACTIVATED. It waits for the GPU and must not run.** The native-VAL eight-endpoint
evaluation (`runs/aircraft-native-val-activation-v1`) holds the accelerator, and its guard halts if
any other RyanLinEnv python process appears. The runner
[`scripts/aircraft_named_baseline.py`](../scripts/aircraft_named_baseline.py) refuses the committed
plan. A run needs a root-approved copy with `activated: true` and an `activation` record, committed
before the first model call, and the operator must pass that copy's SHA256 on every invocation.

## What is measured

- **Model:** `mlx-community/Qwen3.8-27B-8bit` at revision `815b83c0…`, affine 8-bit, group 64,
  loaded locally from `data/models/Qwen3.8-27B-8bit` with `.venv-mlx` (mlx-vlm 0.6.8). The runner
  rehashes every snapshot file against `runs/local-qwen8bit/model-integrity.json` before each load.
  Provenance: [local-qwen3.8-27b-8bit-provenance.json](local-qwen3.8-27b-8bit-provenance.json).
- **Profile:** native instruct (`qwen3.8-native-instruct-v1`), using the author-card sampling:
  temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence penalty 1.5, and the native 16,384-token
  ceiling. Thinking is disabled through the template. Each image gets one seeded sample (seed 1729).
  The eight endpoints use greedy decoding. This difference is disclosed rather than equalised,
  because the baseline must use its supported native profile.
- **Prompt, parser and metric are the eight-endpoint evaluator's.** The prompt is
  `prompt_text(spec, "answer_only")` with the complete 100-label vocabulary. Its SHA256 is
  `859dd366…`, the same as the evaluator bundle's `prompt_sha256`. The parser is
  `parse_label(…, allow_reasoning=True)` with source SHA256 `f75d20ee…`, the same in the evaluator's
  frozen copy and on main. It is applied only when `finish_reason == "stop"`. The metric is
  `classification_comparison`'s equal-class accuracy (mean diagonal of the row-normalised
  confusion matrix), and invalid or missing answers stay in their class denominators. Nothing is
  repaired or retried.
- **Images:** the original native-VAL JPEGs with the shared `fgvc-aircraft-rgb-remove-bottom20-v1`
  transform. There is no box crop and no external resize. The runner never touches TEST.

## Stages

1. **Discovery.** This is the protocol-defined sample from `strong_baseline.validation_discovery`.
   For each class, the runner takes the native-VAL image with the lowest SHA256 of
   `["fgvc-aircraft-2013b","fgvc-aircraft-freshpool-screened-grpo-v1","baseline-discovery",791,label,id]`.
   The salt is pinned by D13. That gives 100 images, evaluated in author order. The 100 IDs are
   pinned in the plan with SHA256 `c8d73a4c…`, and the runner recomputes the selection and refuses
   on any difference. The report gives the numerator and denominator, the exact IDs, macro accuracy
   (equal to micro, because the sample is balanced), the Hoeffding 95% interval (±0.1358), invalid
   and unterminated answers, finish reasons, tokens, and seconds. Discovery is a validity and cost
   diagnostic. It is not a benchmark score.
2. **Full VAL.** This stage covers all 3,333 images in author order (ID-list SHA256 `c33a8794…`).
   Discovery outputs are never copied into it. It starts only if the discovery gate, recomputed
   from the discovery journal, says *proceed*:
   - discovery completed, with no unknown or hard-stop outcomes;
   - at least 90% of answers are valid and terminated;
   - the projected active time stays within 172,800 s. The projection is
     3,333 × the discovery mean image latency + 27 invocations × the per-invocation overhead.
     The overhead is the larger of the discovery mean load time and the discovery charged seconds
     not spent generating, per invocation (load, model and input rehash). A measured 0.0 s load
     counts as 0.0 s; a discovery with no recorded load stops the gate.
   - every discovery invocation ran under this activated plan's SHA256 and the same runner
     sources. A discovery made under another plan copy (for example a relaxed gate) is refused.

   The full stage's active-seconds cap is min(172,800, 2 × projection), and it is a hard ceiling:
   an invocation starts only if the charged seconds plus its 18,060 s wall cap stay within it. If the gate says *stop*, the
   named comparator is reported as incomplete. Prompt, profile, output budget and sample stay as
   they are; any change requires a new plan version. Headroom is not gated here, because that
   decision needs the eight-endpoint results and belongs to the root.

## Budgets, resume and journal

| | discovery | full |
|---|---:|---:|
| images per invocation | 50 | 128 |
| invocation wall cap (external supervisor) | 8,700 s | 18,060 s |
| max invocations | 3 | 32 |
| max charged active seconds | 26,100 | from the gate (≤ 172,800) |
| max calendar time from first start | 7 days | 14 days |
| max run output | 1 GiB | 2 GiB |

The invocation wall cap is 600 s for loading, plus 120 s per image, plus 2,100 s. The 2,100 s term
covers one full 16,384-token generation at the slowest measured 8.2 tokens/s, so the wall cap
does not truncate a long native answer. Each invocation runs under `scripts/supervise_local_vlm.py`
with the 36 GiB RSS and 36 GiB Metal limits used in the measured probe.

Inference goes through the validated `vision_rl.local_vlm.run_local`. Before generating, it writes
an fsynced reservation for each image. If an invocation is interrupted, that image's outcome is
recorded as `unknown_after_interruption` and is never regenerated. Completed answers, including
invalid ones, are never repeated. If anything in the run identity changes (model files, prompt,
pixels, seed, profile, runtime versions or execution source), resume is refused before the model
loads.

The runner's own `ledger.jsonl` holds the budget clock. The first start time and the invocation
count survive resume. A finished invocation is charged its measured seconds. A started invocation
with no finish record is charged its full wall cap. The runner refuses a new invocation in any of
these cases:

- a budget is exhausted. For active seconds this means the charged seconds plus the invocation
  wall cap would exceed the cap, so no invocation can carry the total past it while the supervisor
  enforces the wall cap;
- an earlier invocation never recorded its finish;
- the plan or runner sources changed since the stage's first invocation;
- any unknown or hard-stop outcome exists. These need human review and never continue
  automatically.

## Measured cost (the only local evidence)

The instrument is `runs/local-qwen8bit/aircraft-native-instruct-chunk1-review.json` (supervised
run). Two TRAIN images took 47.57 s of supervised wall time, including loading. The per-image
generation latencies were 20.0 s and 8.2 s, each answer was 7 generated tokens, and peak MLX memory
was 36.36 GB (decimal). These are two images, not a forecast. The full-stage cap is derived from the
discovery measurement instead. VAL contains images up to 1.90 MP, against 1.67 MP for the largest
measured image. A memory stop leaves an unknown outcome, and the stage then halts for review.

## Deviations and limits

- `strong_baseline` proposes comparing the v2 thinking request with a direct-answer request on the
  discovery sample. This plan runs only the direct-answer (native instruct) request, which is the
  only profile with a completed measurement. Native xhigh is unmeasured, not defeated. The claim
  label is therefore "Qwen3.8-27B-8bit native-instruct", not "strongest Qwen3.8".
- The quantised weights are not shown to be equivalent to the unquantised author weights.
- A full native-VAL baseline claim requires all 3,333 predictions with zero unknown outcomes.
  Superiority over the frontier still requires the frozen native TEST, which this plan does not
  touch.
- After full VAL, `compare --against <endpoint predictions JSON>` runs the shared paired comparator
  (20,000 draws, seed 2718) against each of the eight endpoints.

## Activation checklist (all required, none done)

1. The native-VAL evaluation has finished or been stopped by its owner, and no RyanLinEnv python
   process or other accelerator owner is live.
2. Free memory leaves room for a 36.4 GB peak on this 48 GB machine. Agent waves are capped so
   `mem-guard` does not fire.
3. Root review approves the plan. An activated copy is committed, and its SHA256 is recorded in the
   decision log.
4. Discovery runs under the supervisor. The gate output is committed before full VAL.

## Commands (prospective; refused until activation)

```bash
.venv-mlx/bin/python -I scripts/supervise_local_vlm.py \
  --status runs/aircraft-named-baseline-v1/stage-discovery/supervision-1.json \
  --log runs/aircraft-named-baseline-v1/stage-discovery/child-1.log \
  --rss-limit-gib 36 --metal-limit-gib 36 \
  --metal-telemetry runs/aircraft-named-baseline-v1/stage-discovery/run/mlx-memory.jsonl \
  --wall-limit-seconds 8700 -- \
  .venv-mlx/bin/python -I scripts/aircraft_named_baseline.py run \
  --plan ACTIVATED-PLAN.json --expected-plan-sha256 SHA --stage discovery \
  --output runs/aircraft-named-baseline-v1
.venv-mlx/bin/python -I scripts/aircraft_named_baseline.py report \
  --plan ACTIVATED-PLAN.json --stage discovery --output runs/aircraft-named-baseline-v1
```
