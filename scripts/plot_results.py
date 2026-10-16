"""Render measured results with uncertainty; no generated or placeholder scores."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/2026-10-16-pawn-counting"
names = ["majority", "initial", "rl", "sft", "argon"]
labels = [
    "Train-majority baseline",
    "Untrained policy",
    "Supervised + RL",
    "Supervised specialist",
    "Gemini 4 Argon",
]
rows = [json.loads((SOURCE / f"{name}-evaluation.json").read_text()) for name in names]
values = np.array([r["accuracy"] * 100 for r in rows])
low = np.array([r["accuracy_ci95"][0] * 100 for r in rows])
high = np.array([r["accuracy_ci95"][1] * 100 for r in rows])
fig, ax = plt.subplots(figsize=(11, 5.8))
fig.patch.set_facecolor("#f7f9fc")
ax.set_facecolor("#f7f9fc")
colors = ["#a0aaba", "#c1c9d4", "#1967d2", "#5d90df", "#178575"]
ax.barh(labels, values, color=colors, height=0.58)
ax.errorbar(
    values,
    range(len(rows)),
    xerr=[values - low, high - values],
    fmt="none",
    ecolor="#27354a",
    capsize=4,
    linewidth=1.4,
)
for i, r in enumerate(rows):
    ax.text(104, i, f"{r['correct']}/{r['n']}  ({r['accuracy']:.1%})", va="center", fontsize=11)
ax.set_xlim(0, 130)
ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Agreement with imported labels · 95% Wilson intervals")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", alpha=0.15)
ax.set_axisbelow(True)
fig.suptitle("A real run exposed benchmark defects", fontsize=19, weight="bold", x=0.04, ha="left")
fig.text(
    0.04,
    0.89,
    "30 held-out images · fixed test and model choices · native provider responses retained",
    fontsize=11,
    color="#52627b",
)
fig.text(
    0.04,
    0.025,
    "Two confirmed question/label scope mismatches; one suspected omitted annotation.\n"
    "These scores are not verified visual accuracy. No frontier win or RL benefit is established.",
    fontsize=10,
    color="#8a4c00",
)
fig.subplots_adjust(left=0.24, right=0.96, top=0.81, bottom=0.20)
target = ROOT / "docs/assets"
target.mkdir(parents=True, exist_ok=True)
svg = target / "measured-results.svg"
fig.savefig(svg, bbox_inches="tight")
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
fig.savefig(target / "measured-results.png", dpi=160, bbox_inches="tight")
print("Wrote docs/assets/measured-results.svg and .png")
