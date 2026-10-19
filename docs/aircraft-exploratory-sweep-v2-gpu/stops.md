# GPU lane: arms stopped on purpose (not failures)

Each row: the arm, where it stopped, its committed endpoints, and why. Raw results for every
endpoint stay under raw/. The host keeps the spec under gpuq/stopped-by-root/ or gpuq/retired/.

| when (UTC) | arm | stopped at | endpoints kept | reason |
|---|---|---|---|---|
| 2026-10-19 ~06:50 | gpu-rtx3070-xbf16-mixedv3-lr2e-05-g8-b0-t1.3-ds8 (P1) | after step 300 | 150 0.7462, 300 0.7262 | root re-plan: pool-only GRPO forgets |
| 2026-10-19 ~07:00 | gpu-rtx2080ti-cgrpo-I-lr2e-05-g8-b0-t1.3-ds8 | in stage 600 | 150 0.7725, 300 0.7538 | root: declining; fp32 T1.0 twin first |
| 2026-10-19 ~08:30 | gpu-rtx3070-xbf16-hardv3-lr2e-05-g8-b0-t1.3-ds8 (P3) | step 84 | none | root: pool-only at T1.3 forgets |
| 2026-10-19 ~10:10 | gpu-rtx3070-xbf16-R3t1-mixedv3strain-lr2e-05-g8-b0-t1-ds8 | in stage 600 | 150 0.7575, 300 0.7575 | **amendment 19, stopped by root: low value** |
| 2026-10-19 ~10:10 | gpu-rtx3070-xbf16-cgrpo-I-lr2e-05-g8-b0-t1-ds8 @c600 | in stage 1200 | 600 0.7837, 900 0.7650 | **amendment 19, stopped by root: low value** (peak was noise) |

Retired before running: P2 (mixed-v3 lr5e-5), P5 at T1.3, the seed-314 replicate of xbf16 lr2e-5 T1.0,
the exploratory SFT-from-W control, the pools-v2 P0 diagnostic.
