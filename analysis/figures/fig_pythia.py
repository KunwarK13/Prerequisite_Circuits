#!/usr/bin/env python3
"""Pythia order, exact replay, causal clamp, and post-hoc key-dependence checks.

The visual unit is a public pretraining run. Each thin curve is
the mean of its two paired target tapes; the heavy curve is the mean across
the three original runs in panels A/B and all sixteen runs in panel C.
Some additional runs share initialization or data order; no standard error
is inferred from these structured variants or from the paired tapes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
    }
)

REPOSITORY = Path(__file__).resolve().parents[2]
ROOT = REPOSITORY / "artifacts/paper-inputs/controlled"
OUTPUT = REPOSITORY / "outputs/figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results"
PRESENT = "#0072B2"
ABSENT = "#D55E00"
RUNS = (
    ("main", "r3"),
    ("seed1", "r4_seed1"),
    ("seed2", "r4_seed2"),
)
TAPES = (8, 9)
# Presentation decision is explicit; completion alone must not alter the main figure.
INCLUDE_GRADIENT_PANEL = False


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def target_segment(run: dict, block: int) -> dict[int, float]:
    """Return mapped-copy accuracy indexed by updates within one target block."""
    meta = next(item for item in run["blocks"] if int(item["block"]) == block)
    start = int(meta["start_step"])
    prior = [record for record in run["records"] if int(record["step"]) <= start]
    if not prior:
        raise ValueError(f"no boundary reading at target block {block}")
    values = {0: float(prior[-1]["mc_acc"])}
    for record in run["records"]:
        if int(record.get("block", -99)) == block:
            values[int(record["block_step"])] = float(record["mc_acc"])
    return values


def mean_curves(curves: list[dict[int, float]]) -> tuple[list[int], list[float]]:
    xs = sorted(set.intersection(*(set(curve) for curve in curves)))
    return xs, [sum(curve[x] for curve in curves) / len(curves) for x in xs]


def order_curves() -> tuple[
    list[tuple[list[int], list[float]]],
    list[tuple[list[int], list[float]]],
    list[tuple[list[int], list[float]]],
]:
    ready, first, replay = [], [], []
    for _, prefix in RUNS:
        ba = [target_segment(load(f"{prefix}_BA_s{seed}.json"), 1) for seed in TAPES]
        ab_first = [target_segment(load(f"{prefix}_AB_s{seed}.json"), 0) for seed in TAPES]
        ab_replay = [target_segment(load(f"{prefix}_AB_s{seed}.json"), 2) for seed in TAPES]
        ready.append(mean_curves(ba))
        first.append(mean_curves(ab_first))
        replay.append(mean_curves(ab_replay))
    return ready, first, replay


def r5_complete() -> bool:
    return all(
        (RES / f"r5_{tag}_{arm}_s{seed}.json").exists()
        for tag, _ in RUNS
        for seed in TAPES
        for arm in ("CLAMP", "CLAMPCTL")
    )


def r5_curves(arm: str) -> list[tuple[list[int], list[float]]]:
    answer = []
    for tag, _ in RUNS:
        curves = []
        for seed in TAPES:
            run = load(f"r5_{tag}_{arm}_s{seed}.json")
            curves.append(
                {int(record["step"]): float(record["mc_acc"]) for record in run["records"]}
            )
        answer.append(mean_curves(curves))
    return answer


def polypythia_curves(arm: str) -> list[tuple[list[int], list[float]]]:
    """Additional PolyPythias runs (polypythia_clamp/), one curve per run with both tapes complete."""
    directory = ROOT / "polypythia_clamp" / "results"
    answer = []
    tags = sorted({path.name.split("_")[1] for path in directory.glob("p1_*_CLAMP_s8.json")})
    for tag in tags:
        paths = [directory / f"p1_{tag}_{arm}_s{seed}.json" for seed in TAPES]
        if not all(path.exists() for path in paths):
            continue
        curves = []
        for path in paths:
            run = json.loads(path.read_text())
            curves.append(
                {int(record["step"]): float(record["mc_acc"]) for record in run["records"]}
            )
        answer.append(mean_curves(curves))
    return answer


def key_checks() -> dict[str, tuple[float, float]]:
    """Summarize the existing R6 checks, never as new independent replications."""
    values = {"heuristic": [], "relocation": [], "deletion": []}
    for tag, _ in RUNS:
        for seed in TAPES:
            run = load(f"r6_{tag}_CLAMPCTL_s{seed}.json")
            controls = run["endpoint_key_controls"]
            paired = controls["paired_original"]["correct"]
            relocated = controls["relocated"]["correct"]
            if len(paired) != len(relocated) or not paired:
                raise ValueError("key relocation must compare paired examples")
            both = sum(bool(a) and bool(b) for a, b in zip(paired, relocated)) / len(paired)
            if abs(both - controls["pair_both_correct"]) > 1e-12:
                raise ValueError("paired key-relocation statistic disagrees with raw predictions")
            values["heuristic"].append(run["positional_baseline"]["nearest_marker_accuracy"])
            values["relocation"].append(both)
            values["deletion"].append(controls["broken"]["accuracy"])
    return {key: (min(items), max(items)) for key, items in values.items()}


def draw_family(ax, curves, color: str, label: str) -> None:
    for xs, ys in curves:
        ax.plot(xs, ys, color=color, lw=0.75, alpha=0.38)
    grand = []
    for xs, ys in curves:
        grand.append(dict(zip(xs, ys)))
    xs, ys = mean_curves(grand)
    ax.plot(
        xs,
        ys,
        color=color,
        lw=1.8,
        marker="o",
        ms=2.0,
        markevery=[0, 2, 4, 6, 8, 10],
        label=label,
        zorder=3,
    )


def gradient_curves():
    path = ROOT / "pythia_gradient_block/result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result["status"] != "COMPLETE":
        return None
    if len(result["cells"]) != 6:
        raise ValueError("gradient block must include all six planned cells")
    answer = []
    for tag, _ in RUNS:
        cells = [c for c in result["cells"] if c["model_tag"] == tag]
        assert sorted(c["seed"] for c in cells) == [8, 9]
        answer.append(mean_curves([{r["step"]: r["mc_acc"] for r in c["records"]} for c in cells]))
    return answer


def main() -> None:
    ready, first, replay = order_curves()
    causal = r5_complete()
    checks = key_checks() if causal else None
    gradient = gradient_curves() if INCLUDE_GRADIENT_PANEL else None
    ncols = 2 if gradient else 3 if causal else 2
    fig, axes = plt.subplots(
        2 if gradient else 1,
        ncols,
        figsize=(5.5, 3.65) if gradient else (5.5, 2.4),
        sharex=True,
        sharey=True,
    )
    axes = list(axes.flat)

    draw_family(axes[0], ready, PRESENT, "after induction forms")
    draw_family(axes[0], first, ABSENT, "before induction forms")
    axes[0].set_title("(A) Order", fontsize=7.4, loc="left")

    draw_family(axes[1], replay, PRESENT, "replay")
    draw_family(axes[1], first, ABSENT, "first exposure")
    axes[1].set_title("(B) Replay", fontsize=7.4, loc="left")

    if causal:
        controls = r5_curves("CLAMPCTL") + polypythia_curves("CLAMPCTL")
        clamps = r5_curves("CLAMP") + polypythia_curves("CLAMP")
        if len(controls) != 16 or len(clamps) != 16:
            raise ValueError("the final clamp panel requires all sixteen planned runs")
        draw_family(axes[2], controls, PRESENT, "matched control")
        draw_family(axes[2], clamps, ABSENT, "induction heads")
        axes[2].set_title("(C) Training-only suppression", fontsize=7.4, loc="left")
        print("panel C includes all sixteen runs, two tapes averaged within each")

    if gradient:
        draw_family(axes[3], gradient, "#7B3294", "target gradients blocked")
        draw_family(axes[3], r5_curves("CLAMPCTL"), PRESENT, "matched clamp control")
        axes[3].set_title("(D) Forward retained", fontsize=7.4, loc="left")

    for i, ax in enumerate(axes):
        ax.axhline(1 / 32, color="#555555", lw=0.55, ls=":", zorder=0)
        ax.axhline(0.5, color="#777777", lw=0.6, ls=(0, (3, 2)), zorder=0)
        if checks:
            # This narrow band is an observed range for a specified heuristic,
            # not a theoretical limit on all positional strategies.
            ax.axhspan(*checks["heuristic"], color="#777777", alpha=0.40, lw=0.6, zorder=0)
        # Keep the endpoint tick labels inside the native-width PDF canvas.
        ax.set_xlim(-10, 520)
        ax.set_ylim(-0.025, 1.025)
        ax.set_xticks([0, 100, 300, 500])
        ax.set_yticks([0, 0.5, 1])
        ax.grid(axis="y", alpha=0.18, lw=0.4)
        ax.set_xlabel("target updates", fontsize=7.0)
        ax.tick_params(labelsize=7.0, length=2)
        ax.legend(
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(1.0, 0.12),
            fontsize=6.5,
            handlelength=1.4,
            borderaxespad=0.15,
            labelspacing=0.25,
        )
        if i == 0 or gradient and i == 2:
            ax.set_ylabel("mapped-copy accuracy", fontsize=7.0)

    legend = [
        Line2D([], [], color="#555555", lw=0.7, ls=":", label="uniform 1/32 guess"),
        Line2D([], [], color="#777777", lw=0.7, ls=(0, (3, 2)), label="learning criterion (0.50)"),
    ]
    if checks:
        low, high = checks["heuristic"]
        legend.append(
            Patch(
                facecolor="#777777",
                alpha=0.4,
                label=f"nearest-marker heuristic: {low:.3f}–{high:.3f}",
            )
        )
    fig.legend(
        handles=legend,
        loc="center",
        bbox_to_anchor=(0.52, 0.13 if gradient else 0.065),
        ncol=2,
        frameon=False,
        fontsize=6.0,
        handlelength=1.5,
        columnspacing=1.0,
        handletextpad=0.5,
        labelspacing=0.3,
    )
    # Key-relocation and key-deletion outcomes are reported in Table r6, not in a figure footer.
    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.93 if gradient else 0.89,
        bottom=0.23 if gradient else 0.31,
        wspace=0.20,
        hspace=0.65 if gradient else 0.20,
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
    out = OUTPUT / "fig_pythia.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name} ({len(axes)} panels)")


if __name__ == "__main__":
    main()
