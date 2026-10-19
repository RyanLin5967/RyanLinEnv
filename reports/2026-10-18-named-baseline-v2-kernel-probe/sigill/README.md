# MKL VML SIGILL on pool CPUs, and the warm-up that prevents it (TRAIN only)

**Symptom.** Under torch 2.14.1+cpu, a process sometimes dies with SIGILL (rc 132) on a
float32 elementwise op. Every crash examined faults at the same place:
- The address is libtorch_cpu.so offset 0xb724bf3 (`iccad6-dmesg-invalid-opcode.txt`).
- `objdump` shows an AVX-512 `vmovups %zmm26` there, inside `mkl_vml_kernel_sCos_Z0HAynn`: MKL
  VML's AVX-512 float cosine kernel.
- It happens on Ivy Bridge (iccad1/3) and on Broadwell (iccad6, which has AVX2), so it is
  neither a bad core nor an AVX2 problem.
- The likely mechanism is that MKL's lazy dispatch initialisation races when several OpenMP
  workers make the first VML call at once. That is inferred, not proven.

**Workaround in the engine.** `pin_cpu_numerics` does one single-threaded pass of cos, sin,
exp, tanh and log over 2^20 values right after torch is imported, then restores the thread
count. It records `vml_single_thread_warmup: true` in the observed numerics, which enter every
shard's run identity. The plan pins the flag.

## Measurements (plan's kernel pin, 32 threads, one subprocess per run)

Cold, before the warm-up existed, with `ew_probe.py` via `run.sh` / `ew2.sh`:

| host | runs | SIGILL |
| --- | --- | --- |
| iccad3 (`nbv2-sigill-iccad3.log`) | 200 | 5 |
| iccad1 (`nbv2-sigill-iccad1.log`) | 200 | 1 |
| iccad6 (`nbv2-sigill-iccad6-ew.log`) | 200 | 3 |
| iccad6 matrix, seeds 1-100 | 100 | 1 |
| iccad6 matrix, seed 4 repeated | 100 | 0 (not seed-determined) |
| iccad6 matrix, 1 thread | 100 | 0 |
| iccad6 matrix, MKL_NUM_THREADS=1 | 100 | 2 (no fix) |
| iccad6 matrix, single-threaded warm-up | 100 | 0 |

With the warm-up inside the engine's own pin (`ew_engine_probe.py` via `run_warm.sh`; logs
`nbv2-sigillwarm-<host>.log`, which `uwrun logs` re-snapshots while the runs continue):

| host | elementwise runs, SIGILL | real-engine forwards so far | forwards with `warmup: true` and a matching fingerprint | forward SIGILL |
| --- | --- | --- | --- | --- |
| iccad1 | 200, 0 | 21 | 21 | 0 |
| iccad3 | 200, 0 | 27 | 27 | 0 |
| iccad5 | 200, 0 | 16 | 16 | 0 |
| iccad6 (matrix, `ew_probe.py warm`) | 100, 0 | none | none | none |

- **What a forward does** (`model_probe.py`): the plan pin with the warm-up, model load, the
  fingerprint check, then one TRAIN image through the vision tower (including rotary
  embeddings) and full prefill to the first token.
- **Counts:** come from the log lines `MODEL run <n> ok {...}`; a crash would print
  `MODEL run <n> rc=<code>`.
- **Host load:** the hosts were loaded by other sessions' jobs during these runs, so the
  forward timings are not rates.

The cold model-path runs (`nbv2-sigill-iccad1/3`) were stopped before their first forward
finished, when the warm-up superseded them.
