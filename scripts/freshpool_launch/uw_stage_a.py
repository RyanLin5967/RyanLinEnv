"""Run independent signal-limit Stage A stages concurrently on UWaterloo hosts through uwrun.

Used by signal_limit_launch.py when the plan says ``"stage_a_execution": "uwaterloo"``: the
Stage A arms (and their H-select assays) are independent of each other, so they run at the
same time, one per host slot, instead of 13 x ~1 h serially on the Mac. Everything that is not
Stage A still runs on MPS. Every stage goes through the SAME Launcher.prepare / start_instance /
finish as a local stage, so the guards and refusals are the MPS launcher's:

  before a stage: plan pins, local disk reserve and headroom (the fetched results land here),
    the void of an interrupted instance charged to its arm, the 1.25x arm cap, fresh instance
    directories, every template placeholder resolved;
  while it runs: the arm cap on launcher wall time (uwrun stop at the cap; the remote command
    is also bounded by ``uwrun --timeout`` at the remaining cap); host loss (LOST, or
    unreachable longer than ``unreachable_seconds``) is an interrupt: every other UW stage is
    stopped and the launch ends resumable, and on resume each such instance is voided and
    re-run from scratch, never resumed or quoted;
  after it: rc, cap, the fetched fragment validated by the stage's own validator, plus a
    transfer check (the fetched adapter's weights hash to the fragment's adapter_sha256).

Remote side (one uwrun run per stage instance, its own name, uwrun's host-local
/var/tmp/r226lin-runs/NAME run directory, never the NFS home): the bundle holds the plan's
pinned repository files and the whole import tree (``repo/``), the stage's inputs (manifests
rewritten to bundled images, init adapter, label trie) and ``pins.json`` (the SHA256 of every
bundled file). The command checks every pin (``remote-check``, stdlib only) and writes
``out/hardware.json`` (hostname, CPU model, cores, memory, Python, threads) before
``phase_worker.py --remote-cpu`` runs the stage entry with ACCELERATE_TORCH_DEVICE=cpu, the
plan's frozen environment, the bundled import tree on PYTHONPATH and VISION_RL_TORCH_THREADS =
uwrun --threads. Only ``work/out/`` is fetched back. Every stage is journalled with its uwrun
name, host and that hardware record; a finished run's remote directory is removed (uwrun rm).

Host choice is an allowlist: only the plan's ``uw.hosts``, each probed directly (``uwrun ssh``,
so hosts outside uwrun's pools, ecetesla0/1, work), at most ``uw.per_host`` uwrun runs per
host (anyone's ``uw-`` tmux sessions: the remote analogue of the second-owner refusal), with
at least ``uw.min_free_ram_gb`` free RAM (the RSS analogue), ``uw.min_local_gb`` (>= 5, the
disk reserve) free in /var/tmp, and the plan's ``uw.python`` and ``uw.model_directory``
present on the host, and AVX2 in its CPU flags. Both paths must be absolute host-local
paths under /var/tmp.

Blind spots, stated: the remote stage's RSS and the remote disk are checked only at launch
(the phase worker's peak RSS is journalled after); an owner on the remote host that is not a
uwrun run is not seen; the remote environment (venv, model files) is the plan's ``uw.python``
and ``uw.model_directory`` as provisioned on each host, verified by the trainer's own model
directory and package checks, not by this file; PYTHONPATH entries outside the repository are
not shipped (journalled); CPU FP32 numerics differ from MPS FP32, which is why every Stage A
arm runs on the same executor (the plan picks one for all of Stage A) and the hardware of each
is journalled.
"""

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import string
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = re.compile(r"[^A-Za-z0-9_-]+")
UW_KEYS = ("uwrun", "hosts", "per_host", "max_concurrent", "threads", "python",
           "model_directory", "min_local_gb", "min_free_ram_gb", "poll_seconds",
           "unreachable_seconds", "forecast_seconds")
UW_FORECASTS = ("assay_h_select", "stage_a_sft", "stage_a_egsft", "stage_a_hn_sft",
                "stage_a_hls_sft", "stage_a_cce_sft")
# Substitutions that are never local paths to ship (identities, numbers, remote names).
NOT_PATHS = ("python", "stage_dir", "arm_dir", "fragment", "model_directory", "device",
             "stage_id", "arm", "seed", "run_id", "lr", "split", "a_floor", "init_role",
             "label_trie_sha256", "training_group", "arm_label", "active_cap", "arm_cap",
             "t_rl", "group")
# One line per host: load1 nproc availGB uwruns localGB provisioned avx2 (uwrun's own probe,
# plus whether the plan's remote python and model directory exist there, plus AVX2: the torch
# CPU wheel dies with SIGILL rc=132 on hosts without it, iccad1/2/3, measured 2026-10-18 in
# ~/uwaterloo-access/FINDINGS.md [rlenv-sweep-v2]).
PROBE = (r'l=$(cut -d" " -f1 /proc/loadavg); n=$(nproc); '
         r'm=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); '
         r'r=$(tmux ls 2>/dev/null | grep -c "^uw-"); '
         r'd=$(df -Pk /var/tmp | awk "NR==2{print int(\$4/1048576)}"); '
         r'p=0; [ -x {python} ] && [ -d {model} ] && p=1; '
         r'v=0; grep -q -w avx2 /proc/cpuinfo && v=1; '
         r'echo "uwprobe $l $n $m $r $d $p $v"')
IMPORTABLE_SKIP = ("__pycache__",)


def _launch():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import launch

    return launch


class UWInterrupted(KeyboardInterrupt):
    """A UW stage was lost: the launch ends resumable, its instance is voided on resume."""


def verify_uw_block(plan):
    """Refuse a plan whose ``uw`` block could run anything but the allowlisted shape."""
    launch = _launch()
    Stop = launch.Stop
    uw = plan.get("uw")
    if not isinstance(uw, dict) or set(uw) - {*UW_KEYS, "setup"} or not set(UW_KEYS) <= set(uw):
        raise Stop("uw_block_invalid", keys=sorted(uw) if isinstance(uw, dict) else None)
    hosts = uw["hosts"]
    if (not isinstance(hosts, list) or not hosts or len(set(hosts)) != len(hosts)
            or not all(isinstance(h, str) and re.fullmatch(r"[a-z0-9]+", h) for h in hosts)):
        raise Stop("uw_hosts_invalid", hosts=hosts)
    for key, low, high in (("per_host", 1, 8), ("max_concurrent", 1, 64), ("threads", 1, 256),
                           ("min_free_ram_gb", 8, 4096)):
        if type(uw[key]) is not int or not low <= uw[key] <= high:
            raise Stop("uw_limit_invalid", limit=key, value=uw[key])
    if type(uw["min_local_gb"]) is not int or uw["min_local_gb"] < 5:
        raise Stop("uw_disk_reserve_below_5_GiB", min_local_gb=uw["min_local_gb"])
    for key in ("poll_seconds", "unreachable_seconds"):
        if type(uw[key]) not in (int, float) or not uw[key] > 0:
            raise Stop("uw_limit_invalid", limit=key, value=uw[key])
    for key in ("python", "model_directory"):
        value = uw[key]
        if (not isinstance(value, str) or not value.startswith("/var/tmp/")
                or ".." in value.split("/") or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)):
            # ~/uwaterloo-access/FINDINGS.md 2026-10-18: the NFS home quota is ~16 GB for
            # every session combined, so venvs and weights live on the host's /var/tmp.
            raise Stop("uw_path_not_host_local", field=key, value=value)
    if not Path(uw["uwrun"]).is_file():
        raise Stop("uwrun_missing", uwrun=uw["uwrun"])
    if "setup" in uw and not isinstance(uw["setup"], str):
        raise Stop("uw_setup_not_a_string")
    forecasts = uw["forecast_seconds"]
    if not isinstance(forecasts, dict) or any(
            type(forecasts.get(k)) not in (int, float) or not forecasts[k] > 0
            for k in UW_FORECASTS):
        raise Stop("uw_forecast_missing_or_invalid")
    if not any(launch.resolve(p) == Path(__file__).resolve() for p in plan["file_sha256"]):
        raise Stop("plan_does_not_pin", file=str(Path(__file__).resolve()))
    return uw


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


class UWRunner:
    def __init__(self, launcher):
        self.launcher, self.plan = launcher, launcher.plan
        self.launch = _launch()
        self.uw = verify_uw_block(self.plan)
        self.ledger, self.state = launcher.ledger, launcher.state
        self.state.setdefault("uw_running", {})

    # -- uwrun
    def uwrun(self, *args, timeout=600):
        try:
            result = subprocess.run([self.uw["uwrun"], *args], capture_output=True, text=True,
                                    timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return 124, "", "uwrun timed out"
        return result.returncode, result.stdout, result.stderr

    def hosts(self):
        """{host: (load1, nproc, avail_gb, uwruns, local_gb, ready)} for every reachable
        allowlisted host, each probed directly and in parallel; ready = the plan's python and
        model directory exist there and the CPU has AVX2."""
        probe = PROBE.replace("{python}", shlex.quote(self.uw["python"])).replace(
            "{model}", shlex.quote(self.uw["model_directory"]))
        running = {host: subprocess.Popen([self.uw["uwrun"], "ssh", host, probe],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          stdin=subprocess.DEVNULL, text=True)
                   for host in self.uw["hosts"]}
        found = {}
        for host, process in running.items():
            try:
                out, _ = process.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                continue
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 8 and parts[0] == "uwprobe":
                    try:
                        found[host] = (float(parts[1]), int(parts[2]), int(parts[3]),
                                       int(parts[4]), int(parts[5]),
                                       parts[6] == "1" and parts[7] == "1")
                    except ValueError:
                        pass
        return found

    def status(self, name):
        """('RUNNING'|'DONE'|'LOST'|'UNREACHABLE'|'UNKNOWN', rc, flags, host), from
        ``uwrun status NAME``: ``NAME HOST DONE rc=N [timedout] [stopped]`` and the like."""
        code, out, _ = self.uwrun("status", name, timeout=120)
        parts = out.split()
        if code != 0 or len(parts) < 3 or parts[0] != name:
            return "UNREACHABLE", None, set(), None
        host, word = parts[1], parts[2]
        if word == "DONE":
            match = re.fullmatch(r"rc=(\d+)", parts[3] if len(parts) > 3 else "")
            return "DONE", int(match.group(1)) if match else None, set(parts[4:]), host
        return word, None, set(), host

    # -- one stage
    def pick_host(self, running):
        ours = {}
        for entry in running.values():
            ours[entry["host"]] = ours.get(entry["host"], 0) + 1
        best = None
        for host, (load, nproc, avail, uwruns, local, ready) in self.hosts().items():
            if (not ready or avail < self.uw["min_free_ram_gb"]
                    or local < self.uw["min_local_gb"]
                    or max(uwruns, ours.get(host, 0)) >= self.uw["per_host"]):
                continue
            key = (uwruns, load / max(nproc, 1), host)
            if best is None or key < best[0]:
                best = (key, host)
        return None if best is None else best[1]

    def ship_repository(self, directory):
        """repo/: every pinned file under the repository and the whole import tree (the
        trainer reads package data next to its modules). pins.json: every shipped file."""
        launch = self.launch
        root = launch.ROOT.resolve()
        files = {}
        for name in self.plan["file_sha256"]:
            path = launch.resolve(name).resolve()
            if path.is_relative_to(root):
                files[path] = path.relative_to(root)
        if self.plan.get("import_root") is not None:
            tree = launch.resolve(self.plan["import_root"]).resolve()
            if not tree.is_relative_to(root):
                raise launch.Stop("uw_import_root_outside_repository", import_root=str(tree))
            for path in tree.rglob("*"):
                if path.is_file() and not any(p in IMPORTABLE_SKIP for p in path.parts):
                    files[path] = path.relative_to(root)
        pins = {}
        for path, relative in sorted(files.items()):
            link_or_copy(path, directory / "repo" / relative)
            pins[f"repo/{relative}"] = file_sha256(path)
        for name, expected in self.plan["file_sha256"].items():
            path = launch.resolve(name).resolve()
            if path in files and pins[f"repo/{files[path]}"] != expected:
                raise launch.Stop("pinned_file_changed", file=name)
        return pins

    def bundle(self, context, directory):
        """Copy the repository files and the stage's inputs; return the remote argv."""
        launch = self.launch
        pins = self.ship_repository(directory)
        mapped = {str((launch.ROOT / Path(name[len("repo/"):])).resolve()): name
                  for name in pins}
        substitutions = dict(context["substitutions"])
        used = {field for part in context["template"]["argv"]
                for _, field, _, _ in string.Formatter().parse(part) if field}
        remote = {"python": self.uw["python"], "stage_dir": "out/stage", "arm_dir": "out/arm",
                  "fragment": "out/stage/fragment.json", "device": "cpu",
                  "model_directory": self.uw["model_directory"]}
        for key, value in substitutions.items():
            if (key not in used or key in NOT_PATHS or key in remote
                    or not isinstance(value, str)):
                continue
            path = Path(value)
            if not path.is_absolute() or not path.exists():
                continue
            resolved = str(path.resolve())
            if resolved in mapped:
                remote[key] = mapped[resolved]
            elif path.is_dir():
                target = directory / "inputs" / key
                shutil.copytree(path, target, copy_function=shutil.copy2, symlinks=False)
                remote[key] = f"inputs/{key}"
            elif path.suffix == ".jsonl":
                remote[key] = self.bundle_manifest(path, directory, key)
            else:
                target = directory / "inputs" / key / path.name
                link_or_copy(path, target)
                remote[key] = f"inputs/{key}/{path.name}"
        substitutions.update(remote)
        try:
            argv = [part.format_map(substitutions) for part in context["template"]["argv"]]
        except (KeyError, ValueError, IndexError) as exc:
            raise launch.Stop("template_placeholder_unresolved", stage=context["stage_id"],
                              error=str(exc)) from None
        allowed = (self.uw["model_directory"], self.uw["python"])
        local = [a for a in argv if a.startswith("/") and not a.startswith(allowed)]
        if local:
            # Allowlist: a remote command may name only bundle-relative paths and the plan's
            # host-local python and model directory.
            raise launch.Stop("uw_argv_names_local_path", stage=context["stage_id"],
                              paths=local[:5])
        (directory / "pins.json").write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")
        return argv

    @staticmethod
    def bundle_manifest(path, directory, key):
        rows, target = [], directory / "inputs" / key
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and isinstance(row.get("image"), str):
                image = (path.parent / row["image"]).resolve()
                name = f"{len(rows):06d}{image.suffix}"
                link_or_copy(image, target / "images" / name)
                row = {**row, "image": f"images/{name}"}
            rows.append(row)
        target.mkdir(parents=True, exist_ok=True)
        (target / path.name).write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                                                for r in rows))
        return f"inputs/{key}/{path.name}"

    def remote_pythonpath(self):
        """The plan's PYTHONPATH entries that lie in the repository, as bundle paths; others
        are not shipped (returned second, for the journal)."""
        root = self.launch.ROOT.resolve()
        shipped, dropped = [], []
        for entry in self.launch.pythonpath(self.plan):
            path = Path(entry).resolve()
            if path.is_relative_to(root):
                shipped.append(f"$PWD/repo/{path.relative_to(root)}".rstrip("/."))
            else:
                dropped.append(entry)
        if "$PWD/repo/src" not in shipped:
            shipped.append("$PWD/repo/src")
        return shipped, dropped

    def script(self, context, argv):
        environment = dict(self.plan.get("environment") or {})
        environment.pop("PYTHONPATH", None)
        environment.update(ACCELERATE_TORCH_DEVICE="cpu")
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(environment.items()))
        python = shlex.quote(self.uw["python"])
        template = context["template"]
        worker = [self.uw["python"], "-B", "repo/scripts/freshpool_launch/phase_worker.py",
                  "--remote-cpu", "--guard-dir", "out/guards", "--stage", context["stage_id"],
                  "--kind", template["kind"], "--entry", template["entry"],
                  "--entry-sha256", template["entry_sha256"], "--", *argv]
        shipped, _ = self.remote_pythonpath()
        lines = [
            "set -u",
            # uwrun's cwd is ~/runs/NAME/work, a symlink to /var/tmp: O_NOFOLLOW path opens
            # fail under it (FINDINGS.md 2026-10-18 [sweep-feat-format]), so run from the real
            # path, which is also what remote_root.txt records.
            'cd "$(pwd -P)" || exit 97',
            "mkdir -p out/stage out/arm out/guards",
            "pwd -P > out/remote_root.txt",
            f"export {exports}" if exports else ":",
            f'export PYTHONPATH="{":".join(shipped)}" VISION_RL_TORCH_THREADS="$UWRUN_THREADS"',
        ]
        if self.uw.get("setup"):
            lines.append(f"( {self.uw['setup']} ) || exit 96")
        lines += [
            f"{python} -I -B repo/scripts/freshpool_launch/uw_stage_a.py remote-check "
            "pins.json out/hardware.json || exit 95",
            "exec " + " ".join(shlex.quote(part) for part in worker),
        ]
        return "\n".join(lines) + "\n"

    def launch_stage(self, job, host, running):
        launcher, launch = self.launcher, self.launch
        forecast = self.uw["forecast_seconds"][job["forecast_key"]]
        context = launcher.prepare(job["stage_id"], job["template"], job["arm"], forecast,
                                   job["values"], owner_scan=False)
        name = NAME.sub("-", f"rle-{launcher.plan_sha[:8]}-{job['stage_id']}"
                             f"-i{context['instance']}")
        directory = launcher.ledger.directory / "uw" / "bundles" / name
        directory.mkdir(parents=True, exist_ok=False)
        argv = self.bundle(context, directory)
        _, dropped = self.remote_pythonpath()
        remaining = max(1, math.floor(context["cap"] - context["used"]))
        command = ["run", "--host", host, "--sync", str(directory), "--name", name,
                   "--threads", str(self.uw["threads"]), "--timeout", str(remaining),
                   "--min-local", str(self.uw["min_local_gb"]), "--", "bash", "-c",
                   self.script(context, argv)]
        launcher.start_instance(context, command=[self.uw["uwrun"], *command[:-1], "<script>"],
                                executor="uwaterloo", uw_name=name, host=host,
                                remote_argv=argv, pythonpath_not_shipped=dropped)
        self.state["uw_running"][name] = {"stage": job["stage_id"], "host": host,
                                          "instance": context["instance"]}
        self.ledger.save()
        code, out, err = self.uwrun(*command, timeout=4000)
        if code != 0 or (out.strip().splitlines() or [""])[-1] != name:
            self.ledger.append("uw_launch_failed", stage=job["stage_id"], uw_name=name,
                               host=host, exit_code=code, stderr=err[-2000:])
            raise launch.Pause("uw_launch_failed", stage=job["stage_id"], uw_name=name)
        shutil.rmtree(directory / "repo", ignore_errors=True)  # synced; keep inputs + pins
        running[name] = {"job": job, "context": context, "host": host,
                         "started": time.monotonic(), "unreachable_since": None}

    def finish_stage(self, name, entry, rc, flags, host):
        launcher, launch = self.launcher, self.launch
        context = entry["context"]
        elapsed = time.monotonic() - entry["started"]
        stage_id = context["stage_id"]
        self.state["arm_wall_seconds"][stage_id] = context["used"] + elapsed
        destination = context["stage_dir"] / "remote"
        # Trailing slash: rsync copies the CONTENT of work/out into the destination.
        code, _, err = self.uwrun("fetch", name, "work/out/", str(destination), timeout=4000)
        if code != 0:
            self.ledger.append("uw_fetch_failed", stage=stage_id, uw_name=name, stderr=err[-2000:])
            raise UWInterrupted()
        hardware = launch.read_json(destination / "hardware.json")
        guards = [launch.read_json(p) for p in sorted((destination / "guards").glob("*.json"))]
        peak = max((g.get("peak_rss_bytes") or 0 for g in guards if isinstance(g, dict)),
                   default=None)
        self.ledger.append("uw_stage_hardware", stage=stage_id, uw_name=name, host=host,
                           hardware=hardware, remote_peak_rss_bytes=peak)
        reason = ("arm_cap" if "timedout" in flags
                  else "uw_run_stopped_externally" if "stopped" in flags else None)
        if reason is None and rc == 0:
            reason = self.import_fragment(destination, context["fragment"])
        self.state["uw_running"].pop(name, None)
        self.ledger.save()
        outcome = {"exit_code": rc, "reason": reason, "elapsed_seconds": elapsed,
                   "arm_wall_seconds": context["used"] + elapsed,
                   "peak_group_rss_bytes": peak, "demoted": None, "executor": "uwaterloo",
                   "uw_name": name, "host": host}
        launcher.finish(context, outcome, entry["job"]["validate"])
        code, out, err = self.uwrun("rm", name, timeout=600)
        self.ledger.append("uw_run_removed", uw_name=name, exit_code=code,
                           output=(out + err)[-300:])

    @staticmethod
    def import_fragment(destination, local_fragment):
        """Translate the fetched fragment's remote paths to the fetched copy; check the
        adapter bytes survived the transfer. A reason string on failure, else None."""
        root = destination / "remote_root.txt"
        source = destination / "stage" / "fragment.json"
        if not root.is_file() or not source.is_file():
            return None  # finish() reports stage_fragment_missing
        prefix = root.read_text().strip() + "/out"

        def translate(value):
            if isinstance(value, str) and (value == prefix or value.startswith(prefix + "/")):
                return str(destination) + value[len(prefix):]
            if isinstance(value, list):
                return [translate(v) for v in value]
            if isinstance(value, dict):
                return {k: translate(v) for k, v in value.items()}
            return value

        fragment = translate(json.loads(source.read_text()))
        adapter = fragment.get("adapter_path") if isinstance(fragment, dict) else None
        if adapter is not None:
            weights = Path(adapter) / "adapter_model.safetensors"
            if not weights.is_file() or file_sha256(weights) != fragment.get("adapter_sha256"):
                return "uw_fetched_adapter_differs"
        Path(local_fragment).write_text(json.dumps(fragment, indent=2, allow_nan=False) + "\n")
        return None

    def stop(self, name):
        code, out, err = self.uwrun("stop", name, timeout=300)
        self.ledger.append("uw_stop", uw_name=name, exit_code=code, output=(out + err)[-500:])
        return code == 0 and "WARNING" not in out

    def reap(self):
        """Stop what a previous launcher left running before anything is relaunched: their
        instances are voided by prepare() and re-run from scratch, never resumed or quoted."""
        for name in list(self.state["uw_running"]):
            if not self.stop(name):
                raise self.launch.Pause("uw_orphan_run_not_stopped", uw_name=name)
            self.ledger.append("uw_orphan_stopped", uw_name=name,
                               **self.state["uw_running"].pop(name))
            self.ledger.save()

    # -- a batch
    def run_batch(self, jobs):
        self.reap()
        pending = [j for j in jobs if j["stage_id"] not in self.state["completed"]]
        running, waited = {}, None
        try:
            while pending or running:
                launched = False
                if pending and len(running) < self.uw["max_concurrent"]:
                    host = self.pick_host(running)
                    if host is not None:
                        self.launch_stage(pending.pop(0), host, running)
                        launched, waited = True, None
                    elif not running:
                        waited = waited or time.monotonic()
                        if time.monotonic() - waited > self.uw["unreachable_seconds"]:
                            raise self.launch.Pause("uw_no_host_qualifies",
                                                    stage=pending[0]["stage_id"])
                self.poll(running)
                if not launched and (pending or running):
                    time.sleep(self.uw["poll_seconds"])
        except BaseException:
            for name in list(running):
                if self.stop(name):
                    self.state["uw_running"].pop(name, None)
            self.ledger.save()
            raise

    def poll(self, running):
        for name in list(running):
            entry = running[name]
            word, rc, flags, host = self.status(name)
            context = entry["context"]
            elapsed = time.monotonic() - entry["started"]
            self.state["arm_wall_seconds"][context["stage_id"]] = context["used"] + elapsed
            if word == "DONE":
                del running[name]
                self.finish_stage(name, entry, rc, flags, host or entry["host"])
                continue
            if word == "UNREACHABLE":
                entry["unreachable_since"] = entry["unreachable_since"] or time.monotonic()
                lost = time.monotonic() - entry["unreachable_since"] > self.uw[
                    "unreachable_seconds"]
            else:
                entry["unreachable_since"] = None
                lost = word != "RUNNING"
            if lost:
                self.ledger.append("uw_run_lost", stage=context["stage_id"], uw_name=name,
                                   status=word, host=entry["host"])
                raise UWInterrupted()
            if context["used"] + elapsed >= context["cap"]:
                del running[name]
                self.stop(name)
                self.state["uw_running"].pop(name, None)
                self.ledger.save()
                self.launcher.finish(context, {
                    "exit_code": None, "reason": "arm_cap", "elapsed_seconds": elapsed,
                    "arm_wall_seconds": context["used"] + elapsed,
                    "peak_group_rss_bytes": None, "demoted": None, "executor": "uwaterloo",
                    "uw_name": name, "host": entry["host"]}, entry["job"]["validate"])


# ---------------------------------------------------------------- remote side (stdlib only)


def remote_check(pins_path, hardware_path):
    """On the host: every pinned file's bytes, then the hardware record. Exit 1 on a pin."""
    pins = json.loads(Path(pins_path).read_text())
    bad = [name for name, expected in pins.items()
           if not Path(name).is_file() or file_sha256(name) != expected]
    hardware = {"hostname": socket.gethostname(), "cpu_count": os.cpu_count(),
                "python": sys.version.split()[0], "threads": os.environ.get("UWRUN_THREADS"),
                "pins_checked": len(pins), "pins_failed": bad[:10]}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                hardware["cpu_model"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                hardware["mem_total_kib"] = int(line.split()[1])
                break
    except OSError:
        pass
    Path(hardware_path).write_text(json.dumps(hardware, indent=2) + "\n")
    return 1 if bad or not pins else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "remote-check":
        sys.exit(remote_check(sys.argv[2], sys.argv[3]))
    sys.exit("usage: uw_stage_a.py remote-check PINS.json HARDWARE.json "
             "(the runner itself is used through signal_limit_launch.py)")
