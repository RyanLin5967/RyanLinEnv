# GPU lane resume state (aircraft-exploratory-sweep-v2-gpu) — read top to bottom

Branch `exploratory-sweep-v2-gpu.noindex` (on origin; becomes a PR). Worktree
/Users/idide/wt/RyanLinEnv-exploratory-sweep-v2-gpu.noindex. dev800/TRAIN only; never native VAL or sealed TEST.
Root corrections: /Users/idide/RyanLinEnv/runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md. GATE 0 PASSED (D37).
Pools v2 SUPERSEDED (v3 at rl-campaign-pools ad54cfb); nothing here trains on any pool: all arms train on s_train scans.

## Tools (all in scripts/exploratory_sweep/)
- gpu_lane.py: specs | enqueue --host H --ids A,B --prio NN | worker --host H [--need-cores -64] [--slot N] |
  queue | fetch | raw | report | calib | placebo | p0sum | control2 --of ARM --host H | continue
- gpu_ops/running.sh  (what is ACTUALLY running: read from each host's gpuq/running*; never report from queue order)
- gpu_ops/next.sh N   (block until a new arm rc line)   gpu_ops/argv_audit.sh (actual trainer argv per host)
- gpu_ops/boot_delta.sh OLD NEW HOSTS (env OLDPAY/ROOTNAME/SCRATCH for CPU hosts)   gpu_ops/lanetest.sh (pytest on UW)
- Roots: GPU hosts /var/tmp/r226lin-rlesweep-gpu; CPU hosts /dev/shm/r226lin-rlesweep-gpulane. Payload 18edbc9
  (eceubuntu2 766d3cb: + v3 mixed manifest; its receipt is stale because the CUDA kernel check OOMed under our own arm).

## Done (committed under docs/aircraft-exploratory-sweep-v2-gpu/)
- Device control; E0 (resume inside floor); dev800 calibration (Spearman 0.929, RL400-SFT sign reproduced);
  P0-I clean pool (p0-summary.json; T1.3 > T1.0); SFT-from-I curves (best 0.785); placebo s42/s314/s7 (max |d| 0.25 pp so far).
- M9 bf16 s42 done: 0.7700/0.7750/0.7712 (flat vs I 0.775).

## In flight (2026-10-19 ~04:30Z)
- ecetesla2: xbf16 lr2e-5 T1.3 -> xbf16 lr5e-5 T1.3 -> M9 s42 control2 x3 (prio 30-32).
- ecetesla4: M9 bf16 s314 -> xbf16 lr2e-5 T1.0 -> lr5e-5 T1.0 -> placebo s11, s2718.
- eceubuntu2 (fp32; co-tenant x2524wan blocks at times): cgrpo lr2e-5 T1.3 -> P0 mixed v3 -> lr5e-5/lr1e-5/lr3e-6 T1.3 -> lr2e-5 T1.0 -> M9 fp32.
- CPU fp32 (16 thr each): iccad0 lr2e-5 T1.0 + lr5e-5 T1.3 s314; iccad13 lr5e-5 T1.0 + lr2e-5 T1.3 s2718 + lr3e-6 T1.3
  (lr3e-6 T1.0 queued); iccad8 M9 fp32 s42+s314; iccad10 lr1e-5 T1.3 + lr2e-5 T1.0 s314 + lr2e-5 T1.3 s7;
  iccad11 lr5e-5 T1.3; iccad12 lr2e-5 T1.3 s314; iccad7 lr1e-5 T1.0 + lr2e-5 T1.3 s11.

## Rules from root
- Any xbf16 arm >= 0.795 dev800 -> fp32 replicate jumps the 2080 Ti queue; its bf16 control2 follows.
- Per RL arm on finish: control2 (SFT-from-I LR 5e-6/1e-5/2e-5, scanned ids, time-matched each checkpoint) on the
  SAME device class (CPU arms -> CPU fp32, same host preferred).
- At step 150 of CPU arms: report draws/update from dynamic_sampling.jsonl.
- Report every endpoint: vs I, vs best SFT-from-I (0.785), vs matched control, paired CIs (`gpu_lane.py report`).

## OVERNIGHT (2026-10-19 night; /Users/idide/RyanLinEnv/runs/session-handoff-2026-10-16/OVERNIGHT.md)
- 3070 queues (amendment 14): ecetesla2 P1 mixedv3 lr2e-5 T1.3, P3 hardv3, P5 s_train lr1e-4, P7 G16, then M9 s42 ctl;
  ecetesla4 P2 mixedv3 lr5e-5, P4 s_train lr5e-5 T1.3, P6 T1.5, then lr5e-5 T1.0, placebo s2718/s11.
- If a 3070 queue empties: seeds of the current best recipe (gpu_lane promotion_specs style), never idle.
- DONE ~09:10Z: 3070s on main trainer (M3), payload 4113fc1 (scripts; running workers keep the old worker
  script until their next restart; the new one retries only co-tenant OOMs). P=4 OOMs on a 3070; P=2 fits.
- ecetesla4: P2 lr2e-5 (running) -> P2 lr1e-5 -> lr5e-5 T1.0 (P=1) -> placebo s2718/s11. T>=1.3 retired (amendment 20).
- ecetesla2: R2t1 (running) -> P2 lr5e-5 -> P5t1 -> M9 s42 ctl -> G16.
- TODO when both step150 exist (P2 lr2e-5 on ecetesla4, P2 lr5e-5 on ecetesla2): add P2 seed-314 of the better LR
  (spec via promotion_specs-style seed copy, s_train_scan_seed314) at the front of whichever 3070 frees first.
- TODO when R2t1 finishes (600): `gpu_lane.py control2 --of gpu-rtx3070-xbf16-R2t1-mixedv3-lr5e-06-g8-b0.04-t1-ds8 --host ecetesla4 --prio 1401` then rename keys on host to 1401z1..3 (after P2 lr2e-5, before P2 lr1e-5? root: after the P=2 lr2e-5 arm).
- Amendment 21 queued: ecetesla2 122a R2t1 s314, 122b lr1e-5 b0.04, 122c lr5e-6 b0.1; eceubuntu2 002 R2t1 fp32 twin.
- eceubuntu2 (fp32 T1.0 twin) and CPU arms stay on their old trainer until their arms finish.
- Promoted arm finishes -> `gpu_lane.py control2 --of ARM --host ecetesla2|4 --prio 06` (bf16 controls).

## Next action (loop)
`bash scripts/exploratory_sweep/gpu_ops/next.sh 300; bash scripts/exploratory_sweep/gpu_ops/tick.sh`
-> SendMessage main one line per new endpoint (tick prints them); check `gpu_ops/running.sh` hourly.
