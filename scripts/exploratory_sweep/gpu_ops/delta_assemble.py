"""Host-side helper of boot_delta.sh: build PAY from SHIP (changed files, copied) and the
host's current payload OLD (unchanged files, hardlinked), listing exactly NEW's files."""
import json
import os
import shutil
import sys
from pathlib import Path

old, ship, pay = (Path(x) for x in sys.argv[1:4])
files = json.loads((ship / "PAYLOAD.json").read_text())["files"]
for rel in files:
    dst = pay / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if (ship / rel).is_file():
        shutil.copy2(ship / rel, dst)
    else:
        try:
            os.link(old / rel, dst)
        except OSError:  # another filesystem (e.g. /var/tmp -> /dev/shm): copy
            shutil.copy2(old / rel, dst)
shutil.copy2(ship / "PAYLOAD.json", pay / "PAYLOAD.json")
print("assembled", len(files))
