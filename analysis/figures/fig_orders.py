#!/usr/bin/env python3
"""Main order figure: the identical target tape at two arrival states.

Only the three confirmatory seeds and the two causal comparison schedules are
shown in the main figure. Pilot and interleaved arms belong in the appendix.
Thin curves are seeds; the heavy curve is their pointwise mean.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[2]
ROOT = REPOSITORY / "artifacts/paper-inputs/controlled"
OUTPUT = REPOSITORY / "outputs/figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results"
SEEDS = (531, 532, 533)
ARMS = {
    "BA": {"label": "prerequisite present", "color": "#0072B2"},
    "AB": {"label": "prerequisite absent", "color": "#D55E00"},
}

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)


def load(tag: str, seed: int) -> dict:
    path = RES / f"g1_{tag}_s{seed}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def target_block(run: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return evaluations aligned to the start of the first target block."""
    target_segments = sorted(
        {
            int(row["segment"])
            for row in run["records"]
            if row["source"] == "A2" and int(row["segment"]) >= 1
        }
    )
    if not target_segments:
        raise ValueError("run has no target block")
    segment = target_segments[0]
    starts = [
        row
        for row in run["records"]
        if int(row["segment"]) == segment - 1 and row["label"] == "boundary"
    ]
    if not starts:
        raise ValueError("target block has no preceding boundary")
    start = starts[-1]
    rows = [start] + sorted(
        (row for row in run["records"] if int(row["segment"]) == segment),
        key=lambda row: int(row["step"]),
    )
    x = np.asarray([(int(row["step"]) - int(start["step"])) / 1000 for row in rows])
    return x, {
        key: np.asarray([float(row[key]) for row in rows])
        for key in ("key4_d64_acc", "fuel_A2_uptake", "key4_d24_acc")
    }


def pointwise_mean(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    grid = curves[0][0]
    if any(not np.array_equal(grid, x) for x, _ in curves[1:]):
        raise ValueError("confirmatory evaluations do not share a target-step grid")
    return grid, np.mean(np.stack([y for _, y in curves]), axis=0)


def main() -> None:
    panels = (
        ("key4_d64_acc", "(A) Target accuracy", "distance-64 accuracy", (-0.025, 0.82)),
        ("fuel_A2_uptake", "(B) Target copy gain", "copy gain (nats)", (-0.15, 2.9)),
    )
    data: dict[str, dict[int, tuple[np.ndarray, dict[str, np.ndarray]]]] = {
        tag: {seed: target_block(load(tag, seed)) for seed in SEEDS} for tag in ARMS
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(5.5, 2.20),
        sharex=False,
        gridspec_kw={"left": 0.085, "right": 0.99, "bottom": 0.25, "top": 0.78, "wspace": 0.39},
    )
    for ax, (metric, title, ylabel, ylim) in zip(axes, panels):
        for tag, style in ARMS.items():
            curves = []
            for seed in SEEDS:
                x, values = data[tag][seed]
                y = values[metric]
                curves.append((x, y))
                ax.plot(x, y, color=style["color"], lw=0.55, alpha=0.30, zorder=1)
            x, mean = pointwise_mean(curves)
            ax.plot(x, mean, color=style["color"], lw=1.8, solid_capstyle="round", zorder=3)
        ax.set_title(title, fontsize=7.4, loc="left", pad=4)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xlabel("target updates (thousands)", fontsize=7, labelpad=2)
        ax.set_xlim(-1, 92)
        ax.set_ylim(*ylim)
        ax.set_xticks((0, 30, 60, 90))
        ax.tick_params(labelsize=6.5, length=2.5)
        ax.grid(axis="y", color="0.88", lw=0.5, zorder=0)
        ax.axvspan(0, 10, color="0.5", alpha=0.055, lw=0, zorder=0)

    axes[1].axhline(1.0, color="0.35", lw=0.6, ls=(0, (2, 2)), zorder=0)
    axes[1].text(
        88, 1.04, "learning criterion", fontsize=5.5, color="0.35", ha="right", va="bottom"
    )

    handles = [
        Line2D([0], [0], color=style["color"], lw=1.8, label=style["label"])
        for style in ARMS.values()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        fontsize=7.2,
        handlelength=2.2,
        columnspacing=2.5,
    )

    out = OUTPUT / "fig_orders.pdf"
    ax = axes[2]
    for seed in SEEDS:
        run = json.loads((RES / f"e1_REEXPOSE_s{seed}.json").read_text())
        segments = sorted({int(r["segment"]) for r in run["records"] if r.get("source") == "A2"})
        segment = segments[-1]
        start = [
            r for r in run["records"] if r["segment"] == segment - 1 and r["label"] == "boundary"
        ][-1]
        rows = [start] + [
            r
            for r in run["records"]
            if r["segment"] == segment and r["step"] - start["step"] <= 10000
        ]
        ax.plot(
            [(r["step"] - start["step"]) / 1000 for r in rows],
            [r["fuel_A2_uptake"] for r in rows],
            color="#666666" if seed == 532 else "#0072B2",
            lw=1.1,
            ls="--" if seed == 532 else "-",
        )
    ax.set_title("(C) Replay", loc="left", fontsize=7.4)
    ax.set_xlabel("replay updates (thousands)", fontsize=7)
    ax.set_ylabel("copy gain (nats)", fontsize=7)
    ax.set_xlim(-0.1, 10.4)
    ax.set_xticks([0, 5, 10])
    ax.set_ylim(-0.15, 2.9)
    ax.axhline(1, color=".6", lw=0.6, ls=":")
    ax.text(
        0.97,
        0.38,
        "prerequisite\nnot formed",
        transform=ax.transAxes,
        ha="right",
        fontsize=6.5,
        color=".4",
    )
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
            box = label.get_window_extent(renderer)
            if box.x0 < 0 or box.x1 > fig.bbox.width or box.y0 < 0:
                raise ValueError(f"axis text clipped: {label.get_text()}")
    fig.savefig(out)
    plt.close(fig)

    print("wrote fig_orders.pdf")


if __name__ == "__main__":
    main()
