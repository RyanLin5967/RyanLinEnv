"""Restart a running lane arm IN PLACE (same host, same root) with its regenerated spec.

  python3 -I scripts/sft_ceiling/restart_inplace.py ARM HOST OLDRUN N

Why (2026-10-18 ~23:05Z): specs carried temperature None; arm_exec.assay builds Profile from the
spec and validate() raises TypeError on None (forced-fire probe:
scripts/sft_ceiling/profile_probe.py
on iccad6: "FAIL None TypeError", "OK 1.0"). A running arm_exec holds the spec in memory, so the
fix (temperature 1.0 in the spec, as eval-I's spec had; the assay is greedy, temperature unused;
SFT training never passes --temperature) needs a restart.

Steps: `uwrun stop OLDRUN`; find the latest COMPLETE checkpoint in ROOT/arms/ARM/train on HOST;
write launch/ARM-rN/arm.json = current launch/ARM/arm.json + resume_stop (if a checkpoint exists);
with no checkpoint the old arm dir is moved aside and the arm starts fresh. Same host + same root
keeps the trainer's resume identity; active seconds continue from the checkpoint.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
LANE = WT / "runs/aircraft-sft-ceiling-v1"
UW = str(Path.home() / ".claude/bin/uwrun")
FILES = ("trainer_state.json optimizer.pt scheduler.pt rng_state.pth "
         "adapter_model.safetensors active_time.json")


def uw(*args, timeout=900):
    p = subprocess.run([UW, *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise SystemExit(f"REFUSE: uwrun {' '.join(args[:3])} rc={p.returncode}: {p.stderr[-800:]}")
    return p.stdout


def main():
    arm, host, oldrun, n = sys.argv[1:5]
    spec = json.loads((LANE / "launch" / arm / "arm.json").read_text())
    if spec.get("temperature") is None:
        raise SystemExit("REFUSE: regenerated spec still has temperature None")
    uw("stop", oldrun, timeout=300)
    probe = ("for R in /dev/shm/r226lin-sftceil /var/tmp/r226lin-sftceil; do "
             f"[ -d $R/arms/{arm} ] && break; done; "
             "echo ROOT=$R; best=0; "
             f"for c in $(ls -d $R/arms/{arm}/train/checkpoint-* 2>/dev/null); do ok=1; "
             f"for f in {FILES}; do [ -f $c/$f ] || ok=0; done; s=${{c##*-}}; "
             f"[ $ok = 1 ] && [ $s -gt $best ] && best=$s; done; echo BEST=$best; "
             f"pgrep -u $(id -u) -f arms/{arm}/train >/dev/null && echo LIVE=1 || echo LIVE=0")
    out = uw("ssh", host, probe, timeout=120)
    vals = dict(line.split("=", 1) for line in out.split() if "=" in line)
    if vals.get("LIVE") != "0":
        raise SystemExit(f"REFUSE: a trainer for {arm} is still live on {host}")
    root, best = vals["ROOT"], int(vals["BEST"])
    new = dict(spec, restarted={"from_run": oldrun, "reason": "assay Profile temperature None",
                                "checkpoint": best})
    if best:
        new.update(resume_stop=best, stops=[s for s in spec["stops"] if s > best])
    else:
        uw("ssh", host, f"mv {root}/arms/{arm} {root}/arms/{arm}.pre-restart-$(date +%s)",
           timeout=120)
    dest = LANE / "launch" / f"{arm}-r{n}"
    if dest.exists():
        raise SystemExit(f"REFUSE: {dest} exists")
    dest.mkdir(parents=True)
    (dest / "arm.json").write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
    shutil.copy2(LANE / "launch" / arm / "run.sh", dest / "run.sh")
    scratch = "shm" if host.startswith("iccad") else "disk"
    out = uw("run", "--host", host, "--threads", "16", "--mem", "17", "--timeout", "259200",
             "--disk", "3", "--scratch", scratch, "--sync", str(dest), "--name", f"{arm}-{n}",
             "--", "bash", "run.sh")
    print(f"{arm}: root {root}, resumed from checkpoint {best or 'none (fresh)'} -> run {arm}-{n}")


main()
