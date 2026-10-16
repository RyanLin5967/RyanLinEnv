"""A local review artifact for checking model errors against image evidence."""

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

from vision_rl.data import fingerprint, load_manifest


def make_review(manifest: Path, reports: list[Path], output: Path) -> dict:
    rows = load_manifest(manifest)
    identity = fingerprint(rows)
    evaluations = [json.loads(p.read_text()) for p in reports]
    if not evaluations or any(e["dataset_fingerprint"] != identity for e in evaluations):
        raise ValueError("review reports must match the manifest fingerprint")
    indices = [{item["id"]: item for item in e["items"]} for e in evaluations]
    ids = {row["id"] for row in rows}
    if any(set(index) != ids for index in indices):
        raise ValueError("review reports must contain every manifest example")
    cards = []
    for row in rows:
        if all(index[row["id"]]["correct"] for index in indices):
            continue
        with Image.open(row["image"]) as source:
            picture = source.convert("RGB")
            picture.thumbnail((640, 640))
            buffer = io.BytesIO()
            picture.save(buffer, format="PNG")
        image = base64.b64encode(buffer.getvalue()).decode()
        answers = []
        for path, evaluation, index in zip(reports, evaluations, indices, strict=True):
            item = index[row["id"]]
            letter = item["answer"]
            value = row["choices"][ord(letter) - 65] if letter else "Invalid / missing"
            name = f"{path.stem} · {evaluation['model']['name']}"
            answers.append(
                f"<tr><td>{html.escape(name)}</td><td>{html.escape(value)}</td>"
                f"<td>{'Matches label' if item['correct'] else 'Disagrees'}</td>"
                f"<td><code>{html.escape(item.get('response') or '')}</code></td></tr>"
            )
        label = row["choices"][ord(row["answer"]) - 65]
        cards.append(
            f'<article><img src="data:image/png;base64,{image}" alt="Evaluation image">'
            f'<div><p class="id">{html.escape(row["id"])}</p>'
            f"<h2>{html.escape(row['prompt'])}</h2>"
            f"<p>Imported label: <strong>{html.escape(label)}</strong></p>"
            "<table><tr><th>Run / model</th><th>Answer</th><th>Label agreement</th>"
            f"<th>Raw response</th></tr>{''.join(answers)}</table>"
            '<p class="check">Inspect object scope, occlusion, missing labels, and ambiguous '
            "instructions before calling a disagreement a model failure.</p></div></article>"
        )
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vision RL · Disagreement review</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;background:#f4f6f9;color:#152238;margin:0;padding:32px}
main{max-width:1250px;margin:auto}h1{font-size:32px;margin-bottom:8px}h2{font-size:19px}
article{display:grid;grid-template-columns:minmax(240px,480px) 1fr;gap:24px;
background:white;border:1px solid #dce3ec;border-radius:12px;padding:20px;margin:24px 0}
img{width:100%;border-radius:6px}table{border-collapse:collapse;width:100%;font-size:14px}
td,th{text-align:left;border-bottom:1px solid #e1e6ee;padding:8px;overflow-wrap:anywhere}
.id{font:11px monospace;overflow-wrap:anywhere;color:#52627b}.check{color:#775600}
@media(max-width:850px){article{grid-template-columns:1fr}body{padding:16px}}
</style><main><h1>Disagreement review</h1>
<p>Local, self-contained image evidence. Label agreement is not proof of visual correctness.
This file includes dataset images; keep it with the private experiment artifacts.</p>
"""
    document += f"<p>{len(cards)} examples with at least one disagreement · {len(rows)} total</p>"
    document += "".join(cards) + "</main></html>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    return {"report": str(output.resolve()), "disagreements": len(cards), "total": len(rows)}
