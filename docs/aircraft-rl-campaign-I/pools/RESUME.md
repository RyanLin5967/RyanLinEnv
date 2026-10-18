# RL campaign I: TRAIN-only hard-example pools (branch rl-campaign-pools.noindex)

Worktree: /Users/idide/wt/RyanLinEnv-rl-campaign-pools.noindex
Scripts: scripts/rl_campaign_pools/{setup_host.sh,score_worker.py,build_pools.py,make_fixture.py}
UW state (NFS): ~/rlpools/v1/bin (staged scripts), ~/rlpools/v1/state/{claims,results} (shard-NNN.json)
Hosts: own env at /var/tmp/r226lin-pools on ecetesla1, ecetesla0, iccad6 (setup runs rlpools-setup-H-N).
Pool scored: payload-9d9a09f data/manifests/all_train.jsonl (2,329 rows, sha 9bed5893...), 73 shards of 32.

Done: scripts written; setup launched.
Next action: when setups print POOLS SETUP OK, launch workers
  uwrun run --host H --threads T --mem 40 --disk 5 --timeout 43200 --name rlpools-score-H-N -- \
    /var/tmp/r226lin-pools/venv/bin/python -I ~/rlpools/v1/bin/score_worker.py \
    --root /var/tmp/r226lin-pools --state ~/rlpools/v1/state --threads T
then, when all 73 results exist, run build_pools.py --self-test and the build on a UW host,
fetch the out dir, commit manifests + receipts to docs/aircraft-rl-campaign-I/pools/, push.
