"""Move a running lane arm to a freer host, resuming from its latest complete checkpoint.

  python3 -I scripts/sft_ceiling/move_arm.py ARM OLDRUN OLDHOST NEWHOST N [--dry-run]

Steps (each refuses on failure, nothing is guessed):
  1. read the old run's summary (endpoints so far); refuse while it is mid-assay, i.e. the
     trainer's last checkpoint is at or past a stop whose endpoint is not in the summary yet
  2. `uwrun stop OLDRUN`; pack on OLDHOST (pack_arm.sh, --wait); fetch the pack
  3. write launch/ARM-mvN/: arm.json = original spec + resume_stop (packed checkpoint step),
     the remaining stops, prior_endpoints; pack/; run.sh = run_resume.sh
  4. `uwrun run --host NEWHOST ... --name ARM-N -- bash run.sh`
Active seconds continue from the checkpoint's active_time.json (the trainer charges only
active seconds; the stop/pack/move wall time is not active time).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
LANE = WT / "runs/aircraft-sft-ceiling-v1"
UW = str(Path.home() / ".claude/bin/uwrun")


def uw(*args, timeout=900, check=True):
    p = subprocess.run([UW, *args], capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise SystemExit(f"REFUSE: uwrun {' '.join(args[:3])} rc={p.returncode}: {p.stderr[-800:]}")
    return p.stdout


def remote_json(host, path):
    out = uw("ssh", host, f"cat {path} 2>/dev/null || echo null", timeout=120)
    return json.loads(out)


def main():
    arm, oldrun, oldhost, newhost, n = sys.argv[1:6]
    dry = "--dry-run" in sys.argv
    spec = json.loads((LANE / "launch" / arm / "arm.json").read_text())
    summary = remote_json(oldhost, f"~/runs/{oldrun}/work/results/summary.json") or {"endpoints": []}
    done = {e["steps"] for e in summary["endpoints"]}
    root_glob = f"/dev/shm/r226lin-sftceil/arms/{arm}/train /var/tmp/r226lin-sftceil/arms/{arm}/train"
    last = uw("ssh", oldhost, f"for d in {root_glob}; do [ -d $d ] && ls $d | grep -o 'checkpoint-[0-9]*' "
              f"| cut -d- -f2 | sort -n | tail -1; done; true", timeout=120).split()
    step = int(last[-1]) if last else 0
    pending = [s for s in spec["stops"] if s <= step and s not in done]
    if pending:
        raise SystemExit(f"REFUSE: stops {pending} reached but not assayed yet (mid-assay); move later")
    print(f"{arm}: last checkpoint {step}, endpoints {sorted(done)}")
    if dry:
        return
    uw("stop", oldrun, timeout=300)
    packdir = LANE / "pack-src"
    packdir.mkdir(exist_ok=True)
    shutil.copy2(WT / "scripts/sft_ceiling/pack_arm.sh", packdir / "pack_arm.sh")
    packrun = f"mvpack-{arm}-{n}"
    uw("run", "--host", oldhost, "--threads", "1", "--mem", "2", "--timeout", "1800", "--disk", "1",
       "--scratch", "shm" if oldhost.startswith("iccad") else "disk", "--sync", str(packdir),
       "--name", packrun, "--wait", "--", "bash", "pack_arm.sh", arm, oldrun, timeout=2400)
    dest = LANE / "launch" / f"{arm}-mv{n}"
    if dest.exists():
        raise SystemExit(f"REFUSE: {dest} exists")
    fetched = LANE / "fetch" / packrun
    if fetched.exists():
        shutil.rmtree(fetched)
    uw("fetch", packrun, "work/pack/", str(fetched), timeout=1800)
    pack = json.loads((fetched / "PACK.json").read_text())
    resumed = pack["checkpoint_step"]
    dest.mkdir(parents=True)
    shutil.copytree(fetched, dest / "pack")
    new = dict(spec, resume_stop=resumed, stops=[s for s in spec["stops"] if s > resumed],
               prior_endpoints=summary["endpoints"],
               moved={"from_run": oldrun, "from_host": oldhost, "checkpoint": resumed})
    if not new["stops"]:
        raise SystemExit("REFUSE: no stops remain")
    (dest / "arm.json").write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
    shutil.copy2(WT / "scripts/sft_ceiling/run_resume.sh", dest / "run.sh")
    print(uw("run", "--host", newhost, "--threads", "16", "--mem", "17", "--timeout", "259200",
             "--disk", "3", "--scratch", "shm" if newhost.startswith("iccad") else "disk",
             "--sync", str(dest), "--name", f"{arm}-{n}", "--", "bash", "run.sh"))


main()
