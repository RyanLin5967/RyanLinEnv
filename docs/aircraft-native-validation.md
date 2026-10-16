# Prospective native validation evaluation

The [evaluation declaration](aircraft-native-validation-plan-v1.json) is **unactivated**.
It is separate from the running TRAIN400 study and from the unactivated full
held-out learning protocol. No native validation inference has been performed by
this evaluator. The test split remains sealed.
The existing automated clearance verifier may rehash retained test JPEG bytes;
this evaluator never decodes, displays or performs inference on test images.

After the complete author archive and cross-split integrity audit pass, evaluate
these eight fixed artifacts: common warm initialization, RL400, and each of the
three declared SFT learning rates at equal exposure and equal committed optimizer
time. The evaluator checks their initialization, training settings and immutable
endpoint receipts. An incomplete time control cannot be relabeled as matched.
The common warmup's400 exposures remain configuration-derived; later controlled
endpoints carry actual recorded prompt IDs.

All arms use the same FP32 Qwen2-VL-2B base, rank-eight language Q/V adapter,
answer-only prompt, complete100-label vocabulary and evaluation device. The
independent loader accepts only112 finite FP32 numeric tensors of fixed shapes,
then constructs its own trusted LoRA configuration. It never loads submitted
Python, tokenizers, optimizer files or pickle checkpoints.

The denominator is all3,333 native VAL images in author order. Inputs are original
JPEGs with the shared RGB/bottom20-row removal, no target box crop. Only `id` and
`image` enter prediction; gold labels are stored separately for offline scoring.
The exact canonical parser permits optional thinking, requires EOS, and never
repairs answers. Raw tokens, raw text, termination, errors and invalid answers are
retained. Greedy pass@1 matches the native trainer's generation call.

The primary reported metric is the author's equal-class mean of the diagonal of
row-normalized confusion. Micro accuracy is supplemental. The shared offline
comparator supplies20,000-draw paired image bootstrap within each class and
class-cluster sensitivity, seed2718. Every learned endpoint is compared with the
common warm initialization; RL is also compared with all six declared SFT
endpoints. The highest SFT macro accuracy is reported as the strongest declared
control, with lexical ID tie-breaking. There are no extra fits or new endpoints
chosen after seeing accuracy. Selection bias and training-seed uncertainty remain
limitations; numerical intervals do not erase them.

The provisional engineering cap is24 hours across all invocations,512 invocations,
128 images/30 minutes per chunk,120 seconds per image and2 GiB output. At roughly
one second/image, eight arms require about7.4 hours of generation, plus load and
verification overhead; that is an estimate, not a measured native-VAL runtime.
The original wall clock and invocation count survive resume. Each image is
reserved durably before generation. An interrupted reservation becomes an unknown
outcome and is never regenerated. Completed invalid answers and generation errors
also stay consumed. Every journal header and event binds the full frozen plan,
arm, numeric weights, source, inputs, profile and runtime. Reservations and outcomes
also bind an immutable receipt from the loaded inference backend. Comparison checks
those identities against the expected arm; copying another arm's journal fails.
A separate durable head rejects deleted or truncated history. Only one fully
fsynced event beyond the last head can be recovered after an interrupted head
commit; a torn or modified journal fails closed for explicit review.
Unknown/errors disqualify a complete-inference claim and remain incorrect in the
fixed denominator. No report claims a frontier win or satisfies the substantial
RL objective; acceptance thresholds require a separate root-approved protocol.

Before activation, freeze the final weights/metadata hashes, complete evaluator
package and clearance scripts, runtime, generation settings and this declaration.
Use an external process-group supervisor for wall deadlines and resource limits,
including model cleanup: the in-process image watchdog does not bound a device
synchronization that hangs during cleanup. Local hashes detect inconsistent files;
they are not a tamper-proof witness against rewriting all trusted state together.
Use the same frozen import paths for preparation and every chunk. Preserve the
original warmup data spec; the newly published VAL manifest supplies membership,
not a replacement task prompt or vocabulary.

```bash
python -m vision_rl.vlm_local_evaluation prepare \
  --validation data/fgvc-aircraft-2013b/development/validation.jsonl \
  --data-spec data/fgvc-aircraft-2013b/candidates/fourshot-seed791/data_spec.json \
  --root data/fgvc-aircraft-2013b \
  --metadata-archive runs/aircraft-audit/reproduced/annotations.tar.gz \
  --probe runs/aircraft-shared-probe16/prepared \
  --model-directory data/models/Qwen2-VL-2B-Instruct-895c3a49 \
  --study ACTIVATED-FINAL-HASHED-DECLARATION.json --output NEW-EVALUATION-BUNDLE

# Each call resumes the next unattempted images; run under external resource guards.
python -m vision_rl.vlm_local_evaluation run-chunk \
  --bundle NEW-EVALUATION-BUNDLE --arm common

# Repeat chunks for each fixed arm; score only after all eight full denominators finish.
python -m vision_rl.vlm_local_evaluation compare --bundle NEW-EVALUATION-BUNDLE
```

These commands are prospective instructions. Preparation currently rejects the
unactivated declaration, missing endpoint hashes and absent/stale author clearance.
No model execution is authorized by documentation alone.
