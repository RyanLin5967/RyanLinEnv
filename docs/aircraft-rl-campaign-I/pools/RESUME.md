# CURRENT: pools v3 (docs/aircraft-rl-campaign-I/pools/v3/). v2 is SUPERSEDED: do not train on it.

v2 pooled 0447760 (mid, mixed), a pHash near-duplicate of native VAL 0098491 (Hamming 4) and sealed
TEST 1290785 (Hamming 6). Found by a fresh review of PR #41 (2026-10-18): the original audit
(runs/aircraft-freshpool-g0-v1/manifests/phash_audit.json) hashed only its 2,919-image pool, never
train400 or the 15 probe16-outside-train400 rows, all 15 of which are scored here; v2's builder only
removed ids that audit listed, so an unaudited row passed as clean.

v3 fix (scripts/rl_campaign_pools/phash_eligible.py, run rlpools3-phash-1 on iccad7):
- pHash all 1,452 scored rows with the producer's own phash64/near_pairs/HAMMING_MAX (imported from
  scripts/prepare_aircraft_freshpool.py, origin/main sha256 019a57d4...) against native VAL (3,333,
  from images), sealed TEST (3,333, from runs/aircraft-freshpool-g0-v1/test-phashes.json, sha256
  edbf711c..., never images) and train400 (400). Output v3/eligible_phash_audit.json sha256 83270d7d...
- Method parity with the original audit on the 1,437 rows it did hash: 0 pHash mismatches, 13/13
  near-duplicate pairs identical. Forced fire: planted VAL copy flagged (Hamming 0); producer
  forced_fire passed on the eligible rows.
- The 15 unaudited rows: 0447760 (VAL+TEST), 1446341 (VAL), 2171061 (TEST) flagged; the other 12,
  including 0090868, 1548951, 2045491, are clean at Hamming <= 6.
- Also dropped: 13 near-duplicates of train400 (CORRECTIONS keeps train400 out of RL pools),
  including 1377655, 1921783, 2026755. 16 rows dropped in all (v3/pools/phash_excluded.jsonl).
- build_pools.py refuses when any scored row has no pHash entry (self-test plants: missing entry,
  null entry, pair over threshold, unknown split, changed threshold; all refused, 33/33 plants).
- v3 = v2 minus the flagged ids exactly; kept rows' scores byte-identical to v2 (no rescoring).
  hard 207 (-1921783), mid 396 (-0447760, 1377655, 2026755), mixed 454, unreachable 129 (-2171061),
  eligible 1,436. Built by rlpools3-build-1 on iccad7 from /dev/shm/r226lin-rlesweep/payload
  (code_head c44802e, all_train 9bed5893...); ~/rlpools/payload-c44802e no longer exists on NFS.

# RL campaign I: TRAIN-only RL pools from I (branch rl-campaign-pools.noindex)

Rules: runs/rl-lessons-audit-2026-10-18/CORRECTIONS.md (C9, C10, M8). P0-I rollouts moved to the GPU lane
(coordinator, 2026-10-18); this branch builds pools only.

Scripts: scripts/rl_campaign_pools/{setup_host.sh, score_worker.py, build_pools.py, make_fixture.py, exclusions.json}
- eligible = all_train (2,329) - train400 (400) - 477 VAL/TEST pHash near-duplicates = 1,452 rows, 46 units of 32
- p_gold = trie-summed P(gold+EOS) at T 1.0 (label-trie 656dc02f, from commit 552d490); each unit cross-checks
  one row against the trainer's own _batch_gold_probability (canonical and all alternatives), refuses > 1e-4
- smoke (ecetesla1, row 0038598): trainer vs cached-trie |diff| 9.5e-8 (T 1.0), 1.2e-7 (T 1.3)
- builder self-test: 25/25 plants refused (id, content, pool-level for 8 reference sets + forged TRAIN bytes),
  on the fixture and again on the real scores (v2/self_test.json). Corrected from an earlier "30/30, 10 sets".

UW state: ~/rlpools/v2/state/{claims,results}; workers rlpools2-score-{ecetesla0-a,ecetesla1-a,iccad6-a,
iccad7-a,iccad7-b,iccad8-c,iccad0-a}. VOID: ~/rlpools/v1/state-canonical-only-void (pre-correction run).
DONE 2026-10-18: all 46 units scored; built on iccad7 (rlpools2-build-1, rebuilt identically as -2).
Outputs: v2/pools/{manifests/*.jsonl, POOLS.json, coverage.{json,md}, scores.jsonl}, v2/raw (46 unit results).
Counts: eligible 1452; hard 208; mid 399; mixed 458; unreachable 130. Pool-scoring seconds 44527.1.
Next action: none on this branch; the GPU lane consumes the manifests for P0-I.

Lint pass before the PR (2026-10-18): build_pools.py, make_fixture.py and score_worker.py were reformatted for
`ruff check .` / `ruff format`. Each is AST-identical to the version that produced v2 (only loop-variable renames
`l`->`lab`/`line` and docstring wrapping differ). score_worker.py as run: sha256 49721fad37dffa7475af4643a804e8808b242795af012e36e8566835d861755f
(commit 0ae9a30); its local import order is pinned with `# noqa: I001` to match that run.
