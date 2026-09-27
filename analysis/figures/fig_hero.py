#!/usr/bin/env python3
"""Supporting reset comparison in all original and prospective states.

Blue marks arms in which the prerequisite computation stays available while the
target tape trains (downstream heads reset); orange marks arms in which it does
not (upstream head reset). Every plotted point is an actual evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[2]
ROOT = REPOSITORY / "artifacts/paper-inputs/controlled"
OUTPUT = REPOSITORY / "outputs/figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE = "#0072B2", "#D55E00"
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


def curves(fresh: bool, arm: str) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for seed in (1541, 1542, 1543) if fresh else (531, 532, 533):
        if fresh:
            data = json.loads(
                (ROOT / f"matched_failure_confirmation/s{seed}/{arm}.json").read_text()
            )
            records = data["records"]
            x = np.array([v["update"] for v in records])
        else:
            tag = {"UP": "ABL1H", "DOWN": "ABL2"}[arm]
            data = json.loads((ROOT / f"results/c1_{tag}_s{seed}.json").read_text())
            records = data["records"]
            x = np.array([v["step"] - data["meta"]["start_step"] for v in records])
        assert x[0] == 0 and x[-1] == 90_000
        out.append((x, np.array([v["fuel_A2_uptake"] for v in records])))
    return out


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.32, top=0.87, wspace=0.33)
    for ax, horizon, title in zip(
        axes, (1_000, 90_000), ("(A) First 1,000 updates", "(B) Full 90,000-update block")
    ):
        scale = 1 if horizon == 1_000 else 1_000
        for fresh in (False, True):
            for arm, color in (("UP", ORANGE), ("DOWN", BLUE)):
                ys = []
                for x, y in curves(fresh, arm):
                    take = x <= horizon
                    xs, y = x[take] / scale, y[take]
                    ys.append(y)
                    ax.plot(
                        xs,
                        y,
                        color=color,
                        alpha=0.35,
                        lw=0.6,
                        ls="--" if fresh else "-",
                        marker="o" if horizon == 1_000 else None,
                        ms=1.8,
                        mew=0,
                    )
                ax.plot(
                    xs,
                    np.mean(ys, axis=0),
                    color=color,
                    lw=1.6,
                    ls="--" if fresh else "-",
                    marker="o" if horizon == 1_000 else None,
                    ms=2.4,
                    mew=0,
                )
        ax.axhline(1, color=".6", lw=0.6, ls=":")
        ax.set_ylim(-0.15, 2.85)
        ax.set_ylabel("copy gain on target (nats)", labelpad=2, fontsize=8)
        ax.set_title(title, loc="left", fontsize=8.2)
        ax.grid(axis="y", alpha=0.15)
        ax.tick_params(labelsize=7.5, length=2)
        if horizon == 1_000:
            ax.set_xscale("symlog", linthresh=25, linscale=0.35)
            ax.set_xlim(0, 1_150)
            ax.set_xticks([0, 25, 100, 250, 1_000])
            ax.set_xticklabels(["0", "25", "100", "250", "1,000"])
            ax.set_xticks([], minor=True)
            ax.set_xlabel("target updates", fontsize=8)
        else:
            ax.set_xticks([0, 30, 60, 90])
            ax.set_xlabel("target updates (thousands)", fontsize=8)
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, label="induction heads reset"),
            Line2D([], [], color=ORANGE, label="previous-token head reset"),
            Line2D([], [], color=".3", label="original models"),
            Line2D([], [], color=".3", ls="--", label="fresh models"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.52, 0.0),
        ncol=2,
        frameon=False,
        fontsize=7.1,
        handlelength=1.6,
        columnspacing=1.2,
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
            extent = label.get_window_extent(renderer)
            if extent.x0 < 0 or extent.x1 > fig.bbox.width or extent.y0 < 0:
                raise ValueError(f"axis text clipped by figure canvas: {label.get_text()}")
    out = OUTPUT / "fig_hero.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
