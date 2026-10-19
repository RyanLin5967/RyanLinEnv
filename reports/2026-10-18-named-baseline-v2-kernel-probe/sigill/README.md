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

With the warm-up inside the engine's own pin (`ew_engine_probe.py` via `run_warm.sh`): see
`nbv2-sigillwarm-*.log`, which also holds the real-engine forwards (`model_probe.py`: pin,
load, fingerprint check, one TRAIN image through the vision tower and full prefill to the first
token).

The cold model-path runs (`nbv2-sigill-iccad1/3`) were stopped before their first forward
finished, when the warm-up superseded them.
