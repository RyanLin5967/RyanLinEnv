"""Copy into SHIP every file of payload NEW that is absent from, or differs in, payload OLD
(by PAYLOAD.json sha256), plus NEW's PAYLOAD.json. Mac-side helper of boot_delta.sh."""
import json
import shutil
import sys
from pathlib import Path

old_dir, new_dir, ship = (Path(x) for x in sys.argv[1:4])
old = json.loads((old_dir / "PAYLOAD.json").read_text())["files"]
new = json.loads((new_dir / "PAYLOAD.json").read_text())["files"]
if json.loads((new_dir / "PAYLOAD.json").read_text()).get("code_dirty"):
    raise SystemExit("REFUSE: payload NEW was staged from a dirty tree")
n = 0
for rel, digest in new.items():
    if old.get(rel) != digest:
        dst = ship / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(new_dir / rel, dst)
        n += 1
shutil.copy2(new_dir / "PAYLOAD.json", ship / "PAYLOAD.json")
print("delta files", n, "of", len(new))
