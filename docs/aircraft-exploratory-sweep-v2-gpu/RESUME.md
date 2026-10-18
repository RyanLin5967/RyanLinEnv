# GPU lane resume state (aircraft-exploratory-sweep-v2-gpu)

Branch `exploratory-sweep-v2-gpu.noindex`, worktree /Users/idide/wt/RyanLinEnv-exploratory-sweep-v2-gpu.noindex.
Never write the CPU sweep's /Users/idide/RyanLinEnv/runs/aircraft-exploratory-sweep-v2/state.json; never stop its runs.
dev800 only; never native VAL or sealed TEST.

- Code: CPU sweep branch + sweep-feat-cuda merged (64bc4c5), lane changes e25f050, driver bcfb5d5.
- Payload: runs/aircraft-exploratory-sweep-v2-gpu/payload-e25f050 (sha256 ac9ce689...).
- Host root: /var/tmp/r226lin-rlesweep-gpu on ecetesla2, ecetesla4 (bf16), eceubuntu2 (fp32). Receipts ~/rlesweep-gpu/receipts.
- Queue per host: ROOT/gpuq/{pending,running,done,failed}; worker = scripts/exploratory_sweep/gpu_worker.sh via `gpu_lane.py worker`.
- Driver state: runs/aircraft-exploratory-sweep-v2-gpu/state.json (workers, queued).

## Procedure
1. `python3 -I scripts/exploratory_sweep/gpu_lane.py queue --host H` (each host) - see what runs.
2. `gpu_lane.py fetch` then `gpu_lane.py report`.
3. When the first fp32 GRPO arm on eceubuntu2 finishes: `gpu_lane.py control --of <arm> --host eceubuntu2`.
4. Commit raw results (copy summaries/assays into docs/aircraft-exploratory-sweep-v2-gpu/raw/) before interpreting.

## Log
- 2026-10-18T17:55Z bootstrap OK ecetesla2; ecetesla4/eceubuntu2 bootstrapping.
- ~18:10Z all three bootstrapped. Workers: rlegpu-worker-ecetesla2-1792345416, rlegpu-worker-ecetesla4-1792345781,
  rlegpu-worker-eceubuntu2-1792345442. Queued: evals W/I per host, 16 bf16 GRPO-W G8 arms (8 per 3070),
  4 fp32 GRPO-W G8 arms on eceubuntu2. First fetched: I on ecetesla2 bf16 = 0.7750 (CPU fp32 0.775).
  G8 bf16 fits (5951 MiB nvidia-smi), ~6 s/update; eval 315 s (3070), 1094 s (2080 Ti fp32).
- ~18:40Z (Mac clock) UW SSH port 22 unreachable from the Mac for ALL hosts (https to uwaterloo.ca fine).
  Workers run in tmux and continue. NEXT: when reachable -> fetch, raw, commit; then
  `control --of gpu-rtx2080ti-grpo-W-lr1e-05-g8-b0-t1-n --host eceubuntu2` and
  `control --of gpu-rtx3070-grpo-W-lr1e-05-g8-b0-t1-n --host ecetesla2` once those arms are done.
