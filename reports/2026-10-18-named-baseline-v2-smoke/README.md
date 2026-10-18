# Named baseline v2 smoke: 4 TRAIN images (sizing only; not a score)

**Superseded kernel path.** The plan no longer runs the "pinned" configuration below (oneDNN off).
It now keeps oneDNN on, capped at the AVX ISA, because oneDNN off was measured to cause the slow
decode. See `reports/2026-10-18-named-baseline-v2-kernel-probe`. These files stay as the record of
the earlier runs.

The **pinned** runs used the configuration plan v2 registered at the time:
- ATen `default` kernels, `MKL_CBWR=AVX` and oneDNN off;
- an mlx-vlm presence history that includes the prompt tail.

The unpinned runs came earlier. They used default kernels with oneDNN on, and a presence history
of generated tokens only.

Raw outputs of `scripts/aircraft_named_baseline_v2.py smoke`. The run used the author
`Qwen/Qwen3.8-27B@1d4bf0f2` weights, with the snapshot integrity-checked before load, on
transformers 5.20.0 and torch 2.14.1+cpu. The images were TRAIN 0447760, 1158113, 0090868 and
0489934, and no VAL or TEST image was used. Both hosts were shared with other jobs while the smoke
ran, so the timings below are per host and do not stand in for dedicated hardware.

| uwrun name | host | dtype | threads | per-image seconds | mean |
| --- | --- | --- | --- | --- | --- |
| qwen38hf-smoke-pinned-iccad3 | iccad3 (E5-2667 v2) | float32, **pinned** | 32 | 333.1, 550.9, 257.5, 330.1 | 367.9 |
| qwen38hf-smoke-pinned-iccad5 | iccad5 (E5-2640 v4) | float32, **pinned** | 32 | 912.0, 1249.4, 1004.9, 915.2 | 1020.4 |
| qwen38hf-smoke-fp32-iccad1-r2 | iccad1 | float32 | 32 | 859.7, 831.2, 177.1, 266.6 | 533.7 |
| qwen38hf-smoke-fp32-iccad3 | iccad3 | float32 | 32 | 504.6, 596.6, 207.9, 283.2 | 398.1 |
| qwen38hf-smoke-bf16-iccad3-r3 | iccad3 | bfloat16 | 32 | image 1 unfinished after about 3,120 s, then stopped | none |

- **Prefill.** Prefill covers vision plus prompt, up to the first token.
  - Unpinned iccad3: 468, 570, 188 and 272 s; decode about 2.9 to 6.1 s per token.
  - Pinned iccad3: 254, 381, 178 and 225 s; decode 13 to 28 s per token.
  - Pinned iccad5: 477, 766, 491 and 564 s; decode 73 to 88 s per token.

  Each answer was 5 to 7 tokens.
- **Inputs match v1.** All 16 float32 images reproduce v1 MLX's templated-prompt sha `c4409cac…`.
  On 0447760 and 1158113, the two images v1's MLX probe also ran, `input_ids_shape`,
  `pixel_values_shape` and `image_grid_thw` are identical to v1's.
- **Outputs match across hosts.** All four float32 runs produced the same four answers from the
  same seeds: Fokker 50, Fokker 70, DC-9-30 and ATR-42. That includes pinned Ivy Bridge (iccad3)
  against pinned Broadwell (iccad5). A 5-7-token canary is weak evidence, not proof, that the
  numerics are equal.
- **float32 peak memory.** Max RSS was 164.7 GB per process.

Code: iccad1 ran the tree at 5e6ff2d, where the engine was still `src/vision_rl/local_hf.py` and
prefill time was not yet recorded. Unpinned iccad3 float32 ran the engine as committed in c0a4d60. The pinned runs used f20910c.
