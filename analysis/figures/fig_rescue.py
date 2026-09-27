#!/usr/bin/env python3
"""Reversible training-only suppression and activation rescue.

All curves come from the same three previously studied states. Evaluation
restores the original head in every condition and disables activation supply.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

REPOSITORY = Path(__file__).resolve().parents[2]
ROOT = REPOSITORY / "artifacts/paper-inputs/controlled"
OUTPUT = REPOSITORY / "outputs/figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#666666"
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
    }
)


def load(path):
    return json.loads((ROOT / path).read_text())


def draw(ax, curves, color, label, linestyle="-"):
    x = curves[0][0]
    assert all(np.array_equal(x, c[0]) for c in curves)
    for xs, ys in curves:
        ax.plot(xs, ys, color=color, lw=0.65, alpha=0.4, ls=linestyle)
    ax.plot(
        x,
        np.mean([c[1] for c in curves], axis=0),
        color=color,
        label=label,
        lw=1.7,
        ls=linestyle,
        marker="o",
        ms=2,
    )


def schematic(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.set_title(
        "(A) Change the head during training; restore it for evaluation",
        loc="left",
        fontsize=8.2,
        pad=4,
    )
    boxes = [
        (0.01, 0.25, "Same starting model", "Same checkpoint\nSame target batches"),
        (
            0.355,
            0.27,
            "During target training",
            "Keep, suppress or supply the\nprevious-token head output",
        ),
        (0.72, 0.27, "Every evaluation", "Original head enabled\nNothing supplied"),
    ]
    for x, w, title, content in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.08),
                w,
                0.78,
                boxstyle="round,pad=.007",
                edgecolor=".65",
                facecolor="#F5F6F7",
                lw=0.7,
            )
        )
        ax.text(x + w / 2, 0.66, title, ha="center", va="center", fontsize=7.1, fontweight="bold")
        ax.text(x + w / 2, 0.32, content, ha="center", va="center", fontsize=7, linespacing=1.25)
    for start, end in [(0.275, 0.343), (0.64, 0.706)]:
        ax.annotate(
            "",
            xy=(end, 0.47),
            xytext=(start, 0.47),
            arrowprops={"arrowstyle": "->", "color": ".3", "lw": 0.9},
        )


def main():
    fig = plt.figure(figsize=(5.5, 3.15))
    design = fig.add_axes([0.035, 0.745, 0.95, 0.205])
    schematic(design)
    axes = [fig.add_axes([0.09, 0.285, 0.38, 0.34]), fig.add_axes([0.59, 0.285, 0.38, 0.34])]
    suppression = load("stable_input_reversible/result.json")
    supply = load("activation_rescue/result.json")
    assert supply["status"] == "COMPLETE"
    for arm, label, color, ls in [
        ("OPEN", "no suppression", BLUE, "-"),
        ("CONTROL", "control head suppressed", GRAY, "--"),
        ("CLOSED", "previous-token head suppressed", ORANGE, "-"),
    ]:
        curves = []
        for seed in (531, 532, 533):
            rows = suppression["seeds"][str(seed)]["phases"][arm]["records"]
            curves.append(([v["step"] for v in rows], [v["fuel_A2_uptake"] for v in rows]))
        draw(axes[0], curves, color, label, ls)
    for arm, label, color, ls in [
        ("SUPPLY", "correct supply", BLUE, "-"),
        ("NONE", "no supply", ORANGE, "-"),
        ("WRONG", "wrong-head supply", GRAY, "--"),
    ]:
        curves = []
        for seed in (531, 532, 533):
            rows = supply["seeds"][str(seed)][arm]["records"]
            curves.append(([v["step"] for v in rows], [v["fuel_A2_uptake"] for v in rows]))
        draw(axes[1], curves, color, label, ls)
    for ax, title in zip(
        axes, ["(B) Training-only suppression", "(C) Head suppressed, output supplied"]
    ):
        ax.set_title(title, loc="left", fontsize=8.2, pad=6)
        ax.set_ylim(-0.15, 2.9)
        ax.set_xlim(-20, 1020)
        ax.set_yticks([0, 1, 2])
        ax.set_xticks([0, 500, 1000], ["0", "500", "1,000"])
        ax.axhline(1, color=".6", lw=0.6, ls=":")
        ax.grid(axis="y", alpha=0.15)
        ax.tick_params(labelsize=7.5, length=2)
        ax.set_xlabel("target updates", fontsize=8, labelpad=3)
        ax.set_ylabel("copy gain on target (nats)", fontsize=8, labelpad=4)
        ax.legend(
            frameon=False,
            fontsize=7.1,
            loc="upper left",
            bbox_to_anchor=(0, -0.38),
            borderaxespad=0,
            handlelength=1.6,
            labelspacing=0.25,
        )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in [design, *axes]:
        artists = [ax._left_title, *ax.texts]
        if ax in axes:
            artists += [
                ax.xaxis.label,
                ax.yaxis.label,
                *ax.get_xticklabels(),
                *ax.get_yticklabels(),
                ax.get_legend(),
            ]
        for artist in artists:
            extent = artist.get_window_extent(renderer)
            if (
                extent.x0 < 0
                or extent.x1 > fig.bbox.width
                or extent.y0 < 0
                or extent.y1 > fig.bbox.height
            ):
                raise ValueError(f"Text or legend clipped: {artist}")
    fig.savefig(OUTPUT / "fig_rescue.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
