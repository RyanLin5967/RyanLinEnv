# RL campaign I: TRAIN-only RL pools from I (branch rl-campaign-pools.noindex)

Rules: runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md (C9, C10, M8). P0-I rollouts moved to the GPU lane
(coordinator, 2026-10-18); this branch builds pools only.

Scripts: scripts/rl_campaign_pools/{setup_host.sh, score_worker.py, build_pools.py, make_fixture.py, exclusions.json}
- eligible = all_train (2,329) - train400 (400) - 477 VAL/TEST pHash near-duplicates = 1,452 rows, 46 units of 32
- p_gold = trie-summed P(gold+EOS) at T 1.0 (label-trie 656dc02f, from commit 552d490); each unit cross-checks
  one row against the trainer's own _batch_gold_probability (canonical and all alternatives), refuses > 1e-4
- smoke (ecetesla1, row 0038598): trainer vs cached-trie |diff| 9.5e-8 (T 1.0), 1.2e-7 (T 1.3)
- builder self-test on fixture: 30/30 planted overlaps refused (id, content, pool-level for 10 reference sets)

UW state: ~/rlpools/v2/state/{claims,results}; workers rlpools2-score-{ecetesla0-a,ecetesla1-a,iccad6-a,
iccad7-a,iccad7-b,iccad8-c,iccad0-a}. VOID: ~/rlpools/v1/state-canonical-only-void (pre-correction run).
Next action: when all 46 results exist, run build_pools.py on a UW host (see fixture.sh pattern), fetch out/,
commit into docs/aircraft-rl-campaign-I/pools/, push.
