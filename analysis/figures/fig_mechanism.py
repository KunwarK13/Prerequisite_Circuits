#!/usr/bin/env python3
"""Observed persistence and a targeted causal test of retained computation.

All curves read completed original records. Panel A shows three original
initializations in two absence conditions; panel B shows three interventions
on one previously studied state, not three independent states.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPOSITORY = Path(__file__).resolve().parents[2]
ROOT = REPOSITORY / "artifacts/paper-inputs/controlled"
OUTPUT = REPOSITORY / "outputs/figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE = "#0072B2", "#D55E00"
SEEDS = (531, 532, 533)
HORIZON = 600_000

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
    }
)


def read(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.21, top=0.88, wspace=0.32)
    a, b = axes
    for seed in SEEDS:
        ready = read(f"results/g1_BA_s{seed}.json")
        records = ready["records"]
        segment = max(int(r["segment"]) for r in records if r.get("source") == "A2")
        start = [
            r for r in records if int(r["segment"]) == segment - 1 and r["label"] == "boundary"
        ][-1]["step"]
        pts = [
            (r["step"] - start, r["fuel_A2_uptake"])
            for r in records
            if int(r["segment"]) == segment
        ]
        assert pts[0][0] == 500 and pts[-1][0] == 90_000
        a.plot(*zip(*pts), color=BLUE, lw=0.8, alpha=0.8)
        tf = read(f"long_horizon/results/h1_TF_s{seed}.json")
        prelude = tf["meta"]["config"]["prelude_steps"]
        pts = [
            (r["step"] - prelude, r["fuel_A2_uptake"])
            for r in tf["records"]
            if r.get("source") == "A2" and r["step"] > prelude
        ]
        assert pts[-1][0] == HORIZON and max(y for _, y in pts) < 0.03
        a.plot(*zip(*pts), color=ORANGE, lw=0.8, alpha=0.8)
        ur = read(f"long_horizon/results/h1_UR_s{seed}.json")
        original = ur["original_records"]
        start = original[0]["step"]
        pts = [(r["step"] - start, r["fuel_A2_uptake"]) for r in original]
        assert pts[0][0] == 0 and pts[-1][0] == 90_000
        pts += [
            (r["block_step"], r["fuel_A2_uptake"])
            for r in ur["records"]
            if r["label"] != "continued"
        ]
        pts = [p for p in pts if p[0] > 0]
        assert pts[-1][0] == HORIZON and max(y for _, y in pts) < 0.03
        a.plot(*zip(*pts), color=ORANGE, lw=0.8, alpha=0.8, ls=(0, (3, 1.5)))
    a.set_xscale("log")
    a.set_xlim(450, HORIZON * 1.13)
    a.set_xticks([500, 5_000, 50_000, 600_000], ["500", "5k", "50k", "600k"])
    a.set_title("(A) No learning in 600k updates", loc="left", fontsize=7.4)
    a.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=1.2, label="prerequisite formed first"),
            Line2D([], [], color=ORANGE, lw=1.2, label="target first"),
            Line2D(
                [], [], color=ORANGE, lw=1.2, ls=(0, (3, 1.5)), label="previous-token head reset"
            ),
        ],
        frameon=False,
        fontsize=6.5,
        loc="center right",
        bbox_to_anchor=(1, 0.55),
        handlelength=1.8,
        borderaxespad=0.2,
        labelspacing=0.4,
    )

    rc = read("retained_computation/result.json")
    assert rc["status"] == "COMPLETE" and rc["attempt"] == 2
    specs = (
        ("UPX", BLUE, "-", "original + control reset"),
        ("PT", ORANGE, "-", "retained head reset"),
        ("UP2", ORANGE, "--", "original + retained head reset"),
    )
    for arm, color, linestyle, label in specs:
        result = rc["arms"][arm]
        assert result["complete"] and result["tape_matches_archived"]
        points = [(r["update"], r["fuel_A2_uptake"]) for r in result["records"]]
        assert points[-1][0] == 10_000
        b.plot(*zip(*points), color=color, lw=1.4, ls=linestyle, label=label)
    b.set_xlim(-120, 10_250)
    b.set_xticks([0, 3_000, 6_000, 10_000], ["0", "3k", "6k", "10k"])
    b.set_title("(B) Test of the retained head", loc="left", fontsize=7.4)
    b.legend(
        frameon=False,
        fontsize=6.3,
        loc="center right",
        bbox_to_anchor=(1, 0.24),
        handlelength=1.7,
        borderaxespad=0.2,
        labelspacing=0.4,
    )
    for ax in axes:
        ax.axhline(1, color=".55", lw=0.6, ls=":")
        ax.set_ylim(-0.15, 2.9)
        ax.set_yticks([0, 1, 2])
        ax.set_xlabel("target updates", fontsize=7)
        ax.set_ylabel("copy gain on target (nats)", fontsize=7)
        ax.tick_params(labelsize=7, length=2)
        ax.grid(axis="y", alpha=0.15, lw=0.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        for label in [
            ax._left_title,
            ax.xaxis.label,
            ax.yaxis.label,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
        ]:
            extent = label.get_window_extent(renderer)
            if (
                extent.x0 < 0
                or extent.x1 > fig.bbox.width
                or extent.y0 < 0
                or extent.y1 > fig.bbox.height
            ):
                raise ValueError(f"axis text clipped by figure canvas: {label.get_text()}")
    out = OUTPUT / "fig_mechanism.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}: six completed continuations, one targeted three-arm comparison")


if __name__ == "__main__":
    main()
