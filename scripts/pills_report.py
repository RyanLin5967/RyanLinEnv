"""Publish measured detection evidence from an explicit index; never run inference.

Usage: python scripts/pills_report.py --index runs/report-index.json --output reports/pills

Paths in the index are relative to its optional workspace (default: current directory).
Evaluations declare id, label, phase (validation/test), role, path, and optional groups_path
(JSON id->group or prepared manifest JSONL with image_id/capture_hour) and usage_path.
Histories declare label/path/x/y plus optional x_offset. Comparisons declare a/b IDs and
optional groups_path. Measurements declare label/path/fields (display label -> dotted key).
Supplementary evidence declares id/path/description. final_gate declares specialist and
frontiers IDs. No test gate exists until those explicit, matching test reports are present.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vision_rl.detection_eval import (
    PROTOCOL,
    _cached_ap,
    _evaluation_error_count,
    _prepared_cache,
    compare_detections,
)

SECRET_KEYS = {"api_key", "apikey", "authorization", "access_token", "secret", "password"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
        raise ValueError("evidence IDs must contain letters, numbers, hyphens or underscores")
    return value


def _sanitize(value, workspace: Path):
    if isinstance(value, dict):
        return {
            _sanitize(k, workspace): _sanitize(v, workspace)
            for k, v in value.items()
            if k.casefold() not in SECRET_KEYS
        }
    if isinstance(value, list):
        return [_sanitize(v, workspace) for v in value]
    if isinstance(value, str):
        value = value.replace(str(workspace), "$WORKSPACE")
        if value.startswith(("/Users/", "/home/")):
            return "$LOCAL_PATH/" + Path(value).name
    return value


def _json(path: Path):
    return json.loads(path.read_text())


def load_groups(path: Path, image_ids: list[int]) -> dict[int, str | int]:
    """Read capture groups, conservatively pooling images with unknown capture."""
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        mapping = {}
        for row in rows:
            iid = int(row["image_id"])
            group = row.get("capture_hour")
            mapping[iid] = group if group not in (None, "", "unknown") else "unknown-capture"
    else:
        mapping = {int(k): v for k, v in _json(path).items()}
    if not set(image_ids) <= set(mapping):
        raise ValueError("group evidence must cover every evaluated image")
    return {iid: mapping[iid] for iid in image_ids}


def ap_interval(report: dict, groups: dict | None, samples: int, seed: int) -> dict:
    """Percentile interval over cached native AP; report resampling assumptions."""
    prepared = _prepared_cache(report["match_cache"])
    point = _cached_ap(prepared, report["image_ids"])
    if point is None or not np.isclose(point, report["AP"], atol=1e-12):
        raise ValueError("reported AP does not match cached matching data")
    units = {}
    for iid in report["image_ids"]:
        key = groups[iid] if groups is not None else iid
        units.setdefault(key, []).append(iid)
    members = list(units.values())
    if len(members) < 2:
        return {
            "ci95": None,
            "units": len(members),
            "undefined_samples": samples,
            "method": "insufficient independent resampling units",
        }
    rng, scores, undefined = np.random.default_rng(seed), [], 0
    for _ in range(samples):
        ids = [iid for i in rng.integers(0, len(members), len(members)) for iid in members[i]]
        value = _cached_ap(prepared, ids)
        if value is None:
            undefined += 1
        else:
            scores.append(value)
    return {
        "ci95": np.quantile(scores, [0.025, 0.975]).tolist() if scores else None,
        "units": len(members),
        "undefined_samples": undefined,
        "method": "capture-group bootstrap" if groups is not None else "image bootstrap",
    }


def _rows(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    value = _json(path)
    if not isinstance(value, list):
        raise ValueError("history JSON must contain a list of measured rows")
    return value


def _numeric(value):
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _csv(rows: list[dict], fields: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _markdown_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _percent(value) -> str:
    return "not available" if value is None else f"{value * 100:.2f}%"


def _table(headers: list[str], rows: list[list]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(_markdown_cell(v) for v in row) + " |" for row in rows],
        ]
    )


def _plot(summary: dict, target: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "pills-report-v1",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    colors = {"frontier": "#177e89", "specialist": "#3155a5", "ablation": "#9070b3"}
    for ax in axes.flat:
        ax.set_facecolor("#f8fafc")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.16)
        ax.set_axisbelow(True)

    def bars(ax, rows, title):
        ax.set_title(title, loc="left", fontweight="bold", pad=14)
        if not rows:
            ax.text(
                0.5,
                0.5,
                "No measured evidence supplied",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#64748b",
            )
            ax.set_axis_off()
            return
        labels = [f"{r['label']}\n(n={r['n']}; {r['errors']} failed)" for r in rows]
        values = [r["AP"] for r in rows]
        ax.barh(range(len(rows)), values, color=[colors[r["role"]] for r in rows], height=0.56)
        for i, row in enumerate(rows):
            ci = row["AP_interval"]["ci95"]
            if ci:
                ax.plot(ci, [i, i], color="#1e293b", linewidth=1.2)
                ax.plot(ci, [i, i], "|", color="#1e293b", markersize=7)
            ax.text(
                min(max(row["AP"], ci[1] if ci else row["AP"]) + 0.015, 1.01),
                i,
                f"{row['AP']:.1%}",
                va="center",
                fontsize=9,
            )
            if row["errors"]:
                ax.text(0.99, i, "INVALID", ha="right", va="center", color="#a33c20", fontsize=8)
        ax.set_yticks(range(len(rows)), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Native COCO AP50:95 · 95% bootstrap interval")

    rows = summary["evaluations"]
    bars(
        axes[0, 0],
        [r for r in rows if r["phase"] == "validation" and r["role"] == "frontier"],
        "A  Frontier prompt search · VALIDATION ONLY",
    )
    curves = summary["histories"]
    ax = axes[0, 1]
    ax.set_title(
        "B  Specialist learning curves · VALIDATION ONLY", loc="left", fontweight="bold", pad=14
    )
    for history in curves:
        ax.plot(
            [p["x"] for p in history["points"]],
            [p["AP"] for p in history["points"]],
            marker=".",
            markersize=4,
            label=history["label"],
        )
    if curves:
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Completed epoch / recorded step (see exported history)")
        ax.set_ylabel("Recorded validation AP50:95")
    else:
        ax.text(0.5, 0.5, "No learning history supplied", transform=ax.transAxes, ha="center")
        ax.set_axis_off()
    bars(
        axes[1, 0],
        [r for r in rows if r["phase"] == "validation" and r["role"] != "frontier"],
        "C  Training ablations · VALIDATION ONLY",
    )
    tests = [r for r in rows if r["phase"] == "test"]
    bars(axes[1, 1], tests, "D  Frozen held-out test · named models only")
    if not tests:
        axes[1, 1].text(
            0.5,
            0.3,
            "Validation results establish no final frontier win.",
            transform=axes[1, 1].transAxes,
            ha="center",
            color="#a33c20",
            fontsize=10,
        )
    fig.suptitle(
        summary["title"] + "\nMeasured evidence; selection and held-out evaluation separated",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(target, metadata={"Date": None}, bbox_inches="tight")
    target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
    fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(index_path: Path, output: Path, *, bootstrap_samples: int = 2000, seed: int = 42):
    """Create portable compact evidence, prediction CSVs, figure, and reviewable gate."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    index = _json(index_path)
    workspace = Path(index.get("workspace", Path.cwd())).expanduser().resolve()
    output = output.resolve()
    sources, written = (
        {"$INDEX": {"sha256": _sha(index_path), "bytes": index_path.stat().st_size}},
        set(),
    )

    def source(value: str) -> Path:
        path = Path(value).expanduser()
        path = path if path.is_absolute() else workspace / path
        if not path.is_file():
            raise ValueError(f"evidence file does not exist: {value}")
        key = str(path.relative_to(workspace)) if path.is_relative_to(workspace) else path.name
        sources[key] = {"sha256": _sha(path), "bytes": path.stat().st_size}
        return path

    def write(name: str, value, *, raw: bool = False, compress: bool = False):
        if compress:
            name += ".gz"
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            value
            if raw
            else json.dumps(_sanitize(value, workspace), indent=2, allow_nan=False) + "\n"
        )
        if compress:
            path.write_bytes(gzip.compress(payload.encode(), mtime=0))
        else:
            path.write_text(payload)
        written.add(name)

    records, evaluations, histories = {}, [], []
    for spec in index.get("evaluations", []):
        eid = _slug(spec["id"])
        if eid in records or spec["phase"] not in ("validation", "test"):
            raise ValueError("evaluation IDs must be unique and phase explicit validation/test")
        if spec["role"] not in ("frontier", "specialist", "ablation"):
            raise ValueError("evaluation role must be frontier, specialist or ablation")
        report = _json(source(spec["path"]))
        if report.get("protocol") != PROTOCOL or _numeric(report.get("AP")) is None:
            raise ValueError("evaluation must contain measured native COCO AP")
        groups = (
            load_groups(source(spec["groups_path"]), report["image_ids"])
            if spec.get("groups_path")
            else None
        )
        interval = ap_interval(report, groups, bootstrap_samples, seed)
        row = {
            "id": eid,
            "label": spec["label"],
            "phase": spec["phase"],
            "role": spec["role"],
            "n": report["n"],
            "AP": report["AP"],
            "AP50": report.get("AP50"),
            "AP75": report.get("AP75"),
            "AP_interval": interval,
            "errors": _evaluation_error_count(report),
            "model": report.get("model", {}),
            "dataset_fingerprint": report["dataset_fingerprint"],
        }
        evaluations.append(row)
        records[eid] = {"report": report, "spec": spec, "row": row, "groups": groups}
        compact = {k: v for k, v in report.items() if k not in ("match_cache", "items")}
        compact["items"] = [
            {k: v for k, v in item.items() if k != "predictions"}
            for item in report.get("items", [])
        ]
        write(f"evidence/{eid}.json", compact)
        predictions = []
        for item in report.get("items", []):
            for pred in item.get("predictions", []):
                predictions.append(
                    {
                        "image_id": pred["image_id"],
                        "category_id": pred["category_id"],
                        "x": pred["bbox"][0],
                        "y": pred["bbox"][1],
                        "width": pred["bbox"][2],
                        "height": pred["bbox"][3],
                        "score": pred["score"],
                    }
                )
        write(
            f"predictions/{eid}.csv",
            _csv(predictions, ["image_id", "category_id", "x", "y", "width", "height", "score"]),
            raw=True,
            compress=index.get("compress_predictions", False),
        )
        if spec.get("usage_path"):
            write(f"evidence/{eid}-usage.json", _json(source(spec["usage_path"])))
        if groups is not None:
            write(f"evidence/{eid}-groups.json", {str(k): v for k, v in groups.items()})
    comparisons = []
    requests = list(index.get("comparisons", []))
    gate = index.get("final_gate")
    if gate:
        for frontier in gate["frontiers"]:
            if not any(c["a"] == gate["specialist"] and c["b"] == frontier for c in requests):
                requests.append({"a": gate["specialist"], "b": frontier})
    for spec in requests:
        first, second = records[spec["a"]], records[spec["b"]]
        if first["spec"]["phase"] != second["spec"]["phase"]:
            raise ValueError("comparisons cannot mix validation and test")
        groups = (
            load_groups(source(spec["groups_path"]), first["report"]["image_ids"])
            if spec.get("groups_path")
            else first["groups"]
        )
        comparison = compare_detections(
            first["report"],
            second["report"],
            groups,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        comparison.update({"a": spec["a"], "b": spec["b"], "phase": first["spec"]["phase"]})
        comparisons.append(comparison)
        write(f"comparisons/{spec['a']}-vs-{spec['b']}.json", comparison)
    final_status = {
        "status": "not_evaluated",
        "requirements": {"AP": 0.80, "margin": 0.20, "ci_lower": 0.10},
        "scope": "Named baselines on this benchmark only; no claim about all frontier models.",
    }
    if gate:
        specialist = records[gate["specialist"]]
        for eid in [gate["specialist"], *gate["frontiers"]]:
            candidate = records[eid]["report"]
            if (
                candidate.get("synthetic_oracle_bound")
                or candidate.get("evaluation_kind") == "annotation_oracle_upper_bound"
            ):
                raise ValueError("synthetic annotation oracle bounds cannot enter the final gate")
        if (
            not gate["frontiers"]
            or specialist["spec"]["phase"] != "test"
            or specialist["spec"]["role"] != "specialist"
        ):
            raise ValueError("final gate requires a test specialist and at least one frontier")
        for eid in gate["frontiers"]:
            if (
                records[eid]["spec"]["phase"] != "test"
                or records[eid]["spec"]["role"] != "frontier"
            ):
                raise ValueError(
                    "final gate baselines must be explicitly declared frontier test reports"
                )
        relevant = [
            c for c in comparisons if c["a"] == gate["specialist"] and c["b"] in gate["frontiers"]
        ]
        passed = specialist["report"]["AP"] >= 0.80 and all(
            c["margin_target_met"] for c in relevant
        )
        final_status.update(
            {
                "status": "met" if passed else "not_met",
                "specialist": gate["specialist"],
                "frontiers": gate["frontiers"],
            }
        )
    for spec in index.get("histories", []):
        points = []
        for row in _rows(source(spec["path"])):
            x, y = _numeric(row.get(spec.get("x", "epoch"))), _numeric(row.get(spec["y"]))
            if x is None or y is None:
                continue
            if not 0 <= y <= 1:
                raise ValueError("history AP must use fractions in 0..1")
            points.append(
                {
                    "x": x + spec.get("x_offset", 0),
                    "AP": y,
                    "training_phase": row.get("phase", "supervised"),
                }
            )
        histories.append(
            {
                "label": spec["label"],
                "phase": "validation",
                "points": points,
                "metric_field": spec["y"],
            }
        )
    measurements = []
    for spec in index.get("measurements", []):
        data = _json(source(spec["path"]))
        values = {}
        for label, dotted in spec["fields"].items():
            value = data
            for key in dotted.split("."):
                value = value[int(key)] if isinstance(value, list) else value[key]
            if isinstance(value, (dict, list)):
                raise ValueError("measurement fields must select scalar values")
            values[label] = value
        measurements.append(
            {
                "label": spec["label"],
                "values": values,
                "scope": spec.get("scope", "As measured by the source; scopes may differ."),
            }
        )
    for spec in index.get("supplementary_evidence", []):
        path = source(spec["path"])
        name = f"evidence/{_slug(spec['id'])}{path.suffix}"
        if name in written:
            raise ValueError("supplementary evidence name collides with generated evidence")
        if path.suffix == ".json":
            write(name, _json(path))
        elif path.suffix == ".jsonl":
            lines = [
                json.dumps(_sanitize(json.loads(line), workspace), allow_nan=False)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            write(name, "\n".join(lines) + "\n", raw=True, compress=spec.get("compress", False))
        elif path.suffix in (".csv", ".log", ".txt"):
            write(
                name,
                _sanitize(path.read_text(), workspace),
                raw=True,
                compress=spec.get("compress", False),
            )
        else:
            raise ValueError("supplementary evidence must be JSON, JSONL, CSV, LOG or TXT")
    summary = {
        "schema_version": 1,
        "title": index.get("title", "Pill detection study"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "dataset": index.get("dataset", {}),
        "evaluations": evaluations,
        "comparisons": comparisons,
        "histories": histories,
        "measurements": measurements,
        "final_gate": final_status,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "overview": index.get("overview", []),
        "prediction_compression": "gzip" if index.get("compress_predictions") else "none",
    }
    write("summary.json", summary)
    write("histories.json", histories)
    fields = [
        "id",
        "label",
        "phase",
        "role",
        "n",
        "AP",
        "AP50",
        "AP75",
        "errors",
        "AP_ci95_low",
        "AP_ci95_high",
        "bootstrap_units",
        "dataset_fingerprint",
    ]
    csv_rows = []
    for row in evaluations:
        ci = row["AP_interval"]["ci95"] or [None, None]
        csv_rows.append(
            {
                **{k: row[k] for k in fields if k in row},
                "AP_ci95_low": ci[0],
                "AP_ci95_high": ci[1],
                "bootstrap_units": row["AP_interval"]["units"],
            }
        )
    write("metrics.csv", _csv(csv_rows, fields), raw=True)
    _plot(summary, output / "measured-results.svg")
    written.update({"measured-results.svg", "measured-results.png"})
    lines = [
        f"# {summary['title']}",
        "",
        *[part for paragraph in summary["overview"] for part in (paragraph, "")],
        "![Measured study results](measured-results.svg)",
        "",
        "Only measured persisted evidence appears below. Validation data guided training "
        "and prompt selection; "
        "it does not establish a held-out frontier win.",
        "",
        f"**Frozen test gate: {final_status['status'].replace('_', ' ')}.** "
        "Requires specialist AP ≥ 80%, advantage ≥ 20 percentage points, "
        "paired 95% CI lower bound ≥ 10 points, "
        "and zero transport/parse failures against every named baseline.",
        "",
        "Native metric: COCO AP50:95, 101 recall points, maxDets=500. "
        "Intervals are percentile bootstrap "
        "intervals conditional on this sample; capture groups are used where supplied. "
        "Neither an interval nor a passing gate establishes annotation quality "
        "or unseen-session generalization.",
        "",
    ]
    for phase in ("validation", "test"):
        rows = [r for r in evaluations if r["phase"] == phase]
        heading = "Validation — selection only" if phase == "validation" else "Frozen held-out test"
        lines += [
            f"## {heading}",
            "",
        ]
        if not rows:
            lines += ["No measured reports supplied.", ""]
            continue
        cells = []
        for row in rows:
            ci = row["AP_interval"]["ci95"]
            interval = "–" if ci is None else f"{ci[0]:.1%}–{ci[1]:.1%}"
            cells.append(
                [
                    row["label"],
                    row["n"],
                    _percent(row["AP"]),
                    interval,
                    row["errors"],
                    row["AP_interval"]["method"],
                ]
            )
        lines += [
            _table(
                ["Model / configuration", "Images", "AP50:95", "95% CI", "Failures", "Resampling"],
                cells,
            ),
            "",
            "Rows with different dataset fingerprints or image counts are not paired comparisons. "
            "Failed requests remain in the denominator and invalidate a performance-win claim.",
            "",
        ]
    if comparisons:
        lines += [
            "## Paired differences",
            "",
            _table(
                [
                    "Phase",
                    "Specialist / baseline",
                    "AP difference",
                    "95% CI",
                    "Valid evaluation",
                    "Margin met",
                ],
                [
                    [
                        c["phase"],
                        f"{c['a']} / {c['b']}",
                        f"{c['AP_difference'] * 100:+.2f} pp",
                        f"[{c['AP_difference_ci95'][0] * 100:+.2f}, "
                        f"{c['AP_difference_ci95'][1] * 100:+.2f}] pp",
                        c["comparison_valid"],
                        c["margin_target_met"],
                    ]
                    for c in comparisons
                ],
            ),
            "",
        ]
    if measurements:
        lines += ["## Runtime and resource measurements", ""]
        for measurement in measurements:
            lines += [
                f"**{measurement['label']}** — {measurement['scope']}",
                "",
                _table(["Measurement", "Recorded value"], list(measurement["values"].items())),
                "",
            ]
        lines += [
            "Latency scopes, hardware and warmup conditions must match before computing speedups. "
            "Provider-reported tokens are usage evidence; billed money or credits are unknown "
            "unless a billing source supplies them.",
            "",
        ]
    limitations = index.get("dataset", {}).get("limitations", [])
    if limitations:
        lines += ["## Scope and limitations", "", *[f"- {line}" for line in limitations], ""]
    lines += [
        "## Reproduction evidence",
        "",
        "Compact evaluation JSON preserves model identity, source fingerprints "
        "and per-image counts. "
        "Prediction CSV files (gzip-compressed when configured) preserve every scored box "
        "and confidence. "
        "Matching caches are omitted from this bundle; "
        "Recompute them from these predictions and the source COCO annotations "
        "using `evaluate_coco`. "
        "`sources.json` hashes original inputs; `checksums.json` hashes published files.",
        "",
        "Machine-readable exports: [metrics.csv](metrics.csv), [summary.json](summary.json), "
        "[histories.json](histories.json), [sources.json](sources.json).",
        "",
    ]
    write("README.md", "\n".join(lines), raw=True)
    write("index.json", index)
    sources["scripts/pills_report.py"] = {
        "sha256": _sha(Path(__file__)),
        "bytes": Path(__file__).stat().st_size,
    }
    write("sources.json", sources)
    checksums = {name: _sha(output / name) for name in sorted(written)}
    write("checksums.json", checksums)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", default=2000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    summary = build_report(
        args.index, args.output, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evaluations": len(summary["evaluations"]),
                "final_gate": summary["final_gate"]["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
