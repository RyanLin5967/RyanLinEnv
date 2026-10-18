# Named baseline v2 kernel probe: pin cost and host canary (TRAIN only; not a score)

Raw outputs, copied unedited from `~/.claude/uw/results/<uwrun name>/out/`. Every image is TRAIN
(0447760, 1158113, 0090868, 0489934); no VAL or TEST image was opened. Weights:
`Qwen/Qwen3.8-27B@1d4bf0f2` from each host's `/dev/shm`, integrity-checked before load; runtime
from `docs/qwen3.8-27b-hf-cpu-freeze.txt` (each run's log printed `FREEZE_MATCH`). float32. All
hosts were shared with other jobs; iccad10/12/13 carried the exploratory sweep. These are per-host
figures, not dedicated-hardware numbers.

## Instruments

- `probe/kernel_probe.py`, run by `probe/run.sh` (`probe/job-after.sh` queues configs on a host):
  one process per kernel configuration. It runs the v2 `smoke` path and records, per image, the
  answer, drawn token ids, latency, prefill, decode step times, and the SHA256 and top 5 of the
  first-step score row. It then runs a 16-token forced greedy text decode and records s/token,
  CPU-seconds per wall-second ("cores busy") and loadavg.
- The committed `kernel_probe.py` differs from the copy that ran only by two lint-only edits for
  the repo's ruff check (one line wrapped, two imports reordered); its behaviour is unchanged.
- Configurations:

  | name | ATEN_CPU_CAPABILITY | MKL_CBWR | oneDNN | ONEDNN_MAX_CPU_ISA |
  | --- | --- | --- | --- | --- |
  | `pinned` (the previous plan) | default | AVX | off | unset |
  | `unpinned` | unset | unset | on | unset |
  | `nodnn` | unset | unset | off | unset |
  | `cbwr` | default | AVX | on | unset |
  | `dnnavx` (this plan) | default | AVX | on | AVX |

- `fingerprint/fp_probe.py` + `fingerprint/fpjob.sh`: `kernel_fingerprint` (current code,
  integer-valued inputs) at 32, 16 and 32 threads, per configuration, no model. Logs:
  `fingerprint/nbv2-fp-<host>.jsonl`.
- The `kernel_fingerprint` field inside the probe JSONs came from an earlier version that drew its
  inputs with `torch.randn`. ATen's normal sampler picks an AVX2 fill by CPUID whatever the pinned
  capability, so that version differed between Ivy Bridge and AVX2 hosts while the model's score
  rows matched. It was replaced; only `fingerprint/` reflects the committed function.

## Pin vs unpinned, back to back on one host

`nbv2-pinprobe-iccad3/` (image 0447760, 32 threads, run in this order):

| config | s/image | decode s/token | cores busy |
| --- | --- | --- | --- |
| pinned | 462.1 | 61.74 | 25.4 |
| unpinned | 255.0 | 2.12 | 31.5 |
| nodnn | 360.7 | 13.72 | 31.5 |
| cbwr | 243.3 | 2.09 | 31.6 |

The process kept 25-32 cores busy in every configuration, so the slowdown is the kernel path, not
other users' load. oneDNN off costs 6.5x in decode on its own and about 29x together with
`MKL_CBWR`. The previous pin's decode rate varies a lot between draws (iccad2: 13.5 s/token under
it, `nbv2-canary-iccad2/`), but it was the slowest configuration on every host.

## Canary under this plan's path (dnnavx, 4 images, 32 threads)

| host | CPU | mean s/image | decode s/token | first-step score SHA256 (4 images) |
| --- | --- | --- | --- | --- |
| iccad1 | E5-2667 v2 | 258.0 | 2.20 | 4d8a20fd, 9ca6fadd, aa4bde63, 1b223ace |
| iccad2 | E5-2667 v2 | 281.9 | 1.96 | same |
| iccad3 | E5-2667 v2 | 265.0 | 2.08 | same (also at 16 threads: 245.6 s/image) |
| iccad5 | E5-2640 v4 | 503.5 | 6.99 | same |
| iccad10 | Gold 6154 | 558.1 | 12.14 | same |
| iccad12 | Gold 6154 | 973.0 | 18.64 | same |
| iccad13 | Gold 6154 | 546.6 | 7.96 | same |

Answers everywhere: Fokker 50, Fokker 70, DC-9-30, ATR-42. Unpinned, iccad5's score rows differ
from iccad3's (0447760: `34b56fcb…` vs `43f80dae…`), so the pin is needed across CPU families.
Fully unpinned, iccad5 ran 230.1 s/image, 2.2x faster than under dnnavx. That compares all three
settings at once; no `cbwr`-only config ran on iccad5, so how the cost splits between them is
unmeasured. No unpinned model run finished on a Skylake host.

The previous pin also gave a single score row on every host it ran on (0447760: `1941b42d…` on
iccad1, 2, 3, 5, 10 and 12).

iccad10/12/13 probe directories hold only the dnnavx JSON (`nbv2-dnnavx-iccad10/12/13` were stopped
before their unpinned config, to give the sweep its cores back). iccad9 and iccad11 runs were stopped
before finishing any canary image and are not recorded here; their fingerprint (`nbv2-fp-iccad9`)
matches.

## Fingerprint (`fingerprint/`)

Measured on iccad3, iccad5, iccad6, iccad9 and iccad12 (`nbv2-fp-*`), and later, with the same
function, on iccad1, iccad10 and iccad13 (`nbv2-fp2-*`, engine file SHA256 `1320a1cb…`):

| config | iccad1, iccad3 (DEFAULT) | iccad5, iccad6 (AVX2) | iccad9, iccad10, iccad12, iccad13 (AVX-512) |
| --- | --- | --- | --- |
| pinned | f39d5c3c | f39d5c3c | f39d5c3c |
| dnnavx | 9d149dab | 9d149dab | 9d149dab |
| unpinned | 34d1388d | a9270444 | 3ff3c85f |

Identical at 16 and 32 threads on every host. Unpinned, the fingerprint differs per ISA, so the
check fires on a real kernel difference.

## The documented invocation, end to end

`nbv2-clismoke-iccad3c/` (committed tree 879ab74, `smoke.sh`): a venv built only from the lock
(no project install, no PYTHONPATH or kernel variables set), then
`<venv>/bin/python -I scripts/aircraft_named_baseline_v2.py smoke …` on TRAIN 0447760 with the
real `/dev/shm` snapshot, including its `.cache/huggingface` bookkeeping. It passed the runtime
lock and the snapshot check. The engine reported the plan's kernel path (DEFAULT, `MKL_CBWR=AVX`,
oneDNN on, `ONEDNN_MAX_CPU_ISA=AVX`) and `kernel_fingerprint` 9d149dab…, the same as
`fingerprint/`. It answered Fokker 50 in 223.5 s.

## uwrun names

`nbv2-pinprobe-iccad3`, `nbv2-pinprobe-iccad5` (stopped after its pinned config),
`nbv2-canary-iccad{1,2,10,12}` (pinned), `nbv2-dnnavx-iccad{1,2,3,5,10,12,13}`,
`nbv2-fp-iccad{3,5,6,9,12}`. Stopped without a canary result: `nbv2-canary-iccad{9,11,13}`,
`nbv2-dnnavx-iccad{9,11}`.
