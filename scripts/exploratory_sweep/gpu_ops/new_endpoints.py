"""Print one line per dev800 endpoint in runs/<study>/report.json not printed before
(seen list: runs/<study>/endpoints-reported.txt). Run `gpu_lane.py report` first."""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[3] / "runs" / "aircraft-exploratory-sweep-v2-gpu"
seen_path = D / "endpoints-reported.txt"
seen = set(seen_path.read_text().splitlines()) if seen_path.is_file() else set()
rep = json.loads((D / "report.json").read_text())
new = []
for e in rep["arms"]:
    key = f"{e['arm']}|{e['label']}"
    if key in seen:
        continue
    c = e["vs_I"]
    line = (f"{e['arm']} {e['label']} acc {e['accuracy']:.4f} vs I {c['delta']*100:+.2f} pp "
            f"[{c['ci95'][0]*100:+.2f},{c['ci95'][1]*100:+.2f}]")
    b = e.get("vs_best_sft_I")
    if b:
        line += (f"; vs best SFT-I {b['sft_accuracy']:.4f} {b['delta']*100:+.2f} pp "
                 f"[{b['ci95'][0]*100:+.2f},{b['ci95'][1]*100:+.2f}]")
    for lr, k in sorted((e.get("vs_ctl2") or {}).items()):
        line += (f"; vs ctl {lr} {k['control_accuracy']:.4f} {k['delta']*100:+.2f} pp "
                 f"[{k['ci95'][0]*100:+.2f},{k['ci95'][1]*100:+.2f}]")
    new.append(line)
    seen.add(key)
seen_path.write_text("\n".join(sorted(seen)) + "\n")
print("\n".join(new) if new else "no new endpoints")
