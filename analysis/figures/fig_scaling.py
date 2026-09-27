"""Scaling figure: training-only suppression across Pythia sizes, the substitute route,
adaptive suppression, and endpoint localization. Reads raw records only."""

import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    }
)
CONTROL, SUPP, IND_ADAPT, KEY_ADAPT, GRAY = "#0072B2", "#D55E00", "#CC79A7", "#009E73", "#777777"
R = Path(__file__).resolve().parents[2] / "artifacts/paper-inputs/scaling"
RES = R / "results/extensions/pythia_scaling"
EVID = RES / "followup-v2-rebuild-evidence-v1/runs/scaling"
ADA = EVID / "adaptive_suppression_v1"
ART28 = RES / "complete-confirmation-2p8b-v1/runs/scaling"
ART69 = RES / "complete-maturity-distance-raw-v1/runs/scaling"


def horizon_tsv(size):
    f = (
        RES
        / f"complete-training-horizon-{size}-figures-v1/runs/scaling/figures/training_horizon_{size}_v1.tsv"
    )
    rows = list(csv.DictReader(open(f), delimiter="\t"))
    out = {}
    for r in rows:
        key = (r["tape"], r["arm"])
        out.setdefault(key, []).append(
            (
                int(r["step"]),
                float(r["target_accuracy"]),
                float(r["forced_induction_accuracy"] or "nan"),
            )
        )
    return {k: np.array(sorted(v)) for k, v in out.items()}


def from_adaptive(path):
    d = json.load(open(path))
    # column 3: number of heads masked during the updates that ended at this step
    return np.array(
        [
            (
                r["step"],
                r["mc_acc"],
                r["forced_original"]["ind"]["accuracy"],
                len(r["heads"]),
                r["primary_keys"]["pair_both_correct"],
            )
            for r in d["records"]
        ]
    )


def from_result(path):
    d = json.load(open(path))
    diag = {x["step"]: x for x in d.get("diagnostic_records", [])}
    rows = []
    for r in d["records"]:
        ind = diag.get(r["step"], {}).get("ind", {}).get("accuracy", np.nan)
        rows.append((r["step"], r["mc_acc"], ind))
    return np.array(rows)


def curves():
    """Per size: dict with 'supp' list of arrays (step, target, masked induction) and 'ctrl' list."""
    data = {}
    for size in ("160m", "410m", "1b"):
        h = horizon_tsv(size)
        tapes = sorted({t for t, _ in h})
        data[size] = {
            "supp": [h[(t, "circuit")] for t in tapes],
            "ctrl": [h[(t, a)] for t in tapes for a in ("control", "intact")],
        }
    long160 = sorted(glob.glob(str(ADA / "160m_tape*_circuit_trace10k.json")))
    if long160:
        data["160m"]["supp_long"] = [from_adaptive(p)[:, :3] for p in long160]
    tr28 = sorted(glob.glob(str(ADA / "2.8b_tape*_circuit_trace.json")))
    ctrl28 = [
        from_result(p)
        for p in sorted(
            glob.glob(str(ART28 / "confirmation_2.8b_lr3e-05/*/tape*_control_fp32/result.json"))
        )
        + sorted(
            glob.glob(str(ART28 / "confirmation_2.8b_lr3e-05/*/tape*_intact_fp32/result.json"))
        )
    ]
    supp28 = [from_adaptive(p)[:, :3] for p in tr28]
    data["2.8b"] = {"supp": supp28, "ctrl": ctrl28}
    base69 = ART69 / "confirmation_maturity_6.9b_step2000_lr3e-05/6.9b_step2000"
    data["6.9b"] = {
        "supp": [
            from_result(p)
            for p in sorted(glob.glob(str(base69 / "tape*_circuit_fp32_fsdp/result.json")))
        ],
        "ctrl": [
            from_result(p)
            for p in sorted(glob.glob(str(base69 / "tape*_control_fp32_fsdp/result.json")))
            + sorted(glob.glob(str(base69 / "tape*_intact_fp32_fsdp/result.json")))
        ],
    }
    return data


def panel_size(ax, d, title, xmax):
    for c in d["ctrl"]:
        ax.plot(c[:, 0], c[:, 1], color=CONTROL, lw=0.8, alpha=0.55)
    series = d.get("supp_long") or d["supp"]
    for s in series:
        ax.plot(s[:, 0], s[:, 1], color=SUPP, lw=1.3)
        ax.plot(s[:, 0], s[:, 2], color=SUPP, lw=0.9, ls=(0, (2.2, 1.4)))
    ax.set_title(title, fontsize=7.2, loc="left")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.03, 1.04)
    ax.set_yticks([0, 0.5, 1])


def main():
    data = curves()
    fig = plt.figure(figsize=(5.5, 4.1))
    gs = fig.add_gridspec(
        2,
        5,
        height_ratios=[1, 1.1],
        hspace=0.75,
        wspace=0.42,
        left=0.075,
        right=0.985,
        top=0.86,
        bottom=0.165,
    )
    sizes = [
        ("160m", "160M (4 heads)"),
        ("410m", "410M (8)"),
        ("1b", "1B (8)"),
        ("2.8b", "2.8B (4)"),
        ("6.9b", "6.9B (24)"),
    ]
    for i, (key, title) in enumerate(sizes):
        ax = fig.add_subplot(gs[0, i])
        long = key == "160m" and "supp_long" in data[key]
        xmax = 500 if key == "6.9b" else (10000 if long else 2000)
        panel_size(ax, data[key], ("(A) " if i == 0 else "") + title, xmax)
        if long:
            ax.set_xticks([0, 5000, 10000])
            ax.set_xticklabels(["0", "5k", "10k"])
        elif key == "6.9b":
            ax.set_xticks([0, 250, 500])
            ax.set_xticklabels(["0", "250", "500 "])
        else:
            ax.set_xticks([0, 1000, 2000])
            ax.set_xticklabels(["0", "1k", "2k"])
        if i == 0:
            ax.set_ylabel("accuracy", fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("updates", fontsize=6.5, labelpad=1)
    fig.legend(
        handles=[
            Line2D([], [], color=CONTROL, lw=0.9, alpha=0.7, label="target, controls"),
            Line2D([], [], color=SUPP, lw=1.3, label="target, induction heads suppressed"),
            Line2D(
                [],
                [],
                color=SUPP,
                lw=0.9,
                ls=(0, (2.2, 1.4)),
                label="masked induction (suppressed runs)",
            ),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=6.3,
        bbox_to_anchor=(0.53, 1.0),
        handlelength=2.2,
        columnspacing=1.4,
    )

    # (B) adaptive suppression at 410M
    axb = fig.add_subplot(gs[1, 0:2])
    fixed = horizon_tsv("410m")
    for t in ("38", "39"):
        axb.plot(fixed[(t, "circuit")][:, 0], fixed[(t, "circuit")][:, 1], color=SUPP, lw=1.1)
        axb.plot(
            fixed[(t, "control")][:, 0],
            fixed[(t, "control")][:, 1],
            color=CONTROL,
            lw=0.8,
            alpha=0.55,
        )
    handles = [
        Line2D([], [], color=CONTROL, lw=0.9, alpha=0.7, label="control"),
        Line2D([], [], color=SUPP, lw=1.1, label="fixed 8-head mask"),
    ]
    for tag, color, ls, label in (
        ("", IND_ADAPT, (0, (4, 1.5)), "re-mask on induction"),
        ("_keyroute", KEY_ADAPT, "-", "re-mask on key retrieval"),
    ):
        paths = sorted(glob.glob(str(ADA / f"410m_tape*_circuit{tag}.json")))
        for p in paths:
            a = from_adaptive(p)
            axb.plot(a[:, 0], a[:, 1], color=color, lw=1.1, ls=ls)
        if paths:
            handles.append(Line2D([], [], color=color, lw=1.1, ls=ls, label=label))
    yoked = sorted(glob.glob(str(ADA / "410m_tape*_yoked_keyroute.json")))
    for p in yoked:
        a = from_adaptive(p)
        axb.plot(a[:, 0], a[:, 1], color=CONTROL, lw=1.0, ls=(0, (1, 1)))
    if yoked:
        handles.append(Line2D([], [], color=CONTROL, lw=1.0, ls=(0, (1, 1)), label="yoked control"))
    static = sorted(glob.glob(str(ADA / "410m_tape*_circuit_keyroutestatic.json")))
    for p in static:
        a = from_adaptive(p)
        axb.plot(a[:, 0], a[:, 1], color=CONTROL, lw=1.0, ls=(0, (4, 1, 1, 1)))
    if static:
        handles.append(
            Line2D(
                [],
                [],
                color=CONTROL,
                lw=1.0,
                ls=(0, (4, 1, 1, 1)),
                label="static control (22/16 heads)",
            )
        )
    axb.set_title("(B) 410M, adaptive re-masking", fontsize=7.2, loc="left")
    axb.set_xlim(0, 2000)
    axb.set_ylim(-0.03, 1.04)
    axb.set_yticks([0, 0.5, 1])
    axb.set_xlabel("updates", fontsize=6.5, labelpad=1)
    axb.set_ylabel("target accuracy", fontsize=7)
    axb.legend(
        handles=handles,
        frameon=False,
        fontsize=5.8,
        loc="upper left",
        handlelength=2.0,
        bbox_to_anchor=(-0.02, -0.25),
        ncol=3,
        columnspacing=1.0,
    )

    # (C) number of masked heads for adaptive runs
    axc = fig.add_subplot(gs[1, 2])
    for tag, color, ls in (("", IND_ADAPT, (0, (4, 1.5))), ("_keyroute", KEY_ADAPT, "-")):
        for p in sorted(glob.glob(str(ADA / f"410m_tape*_circuit{tag}.json"))):
            a = from_adaptive(p)
            axc.step(a[:, 0], a[:, 3], where="pre", color=color, lw=1.1, ls=ls)
    axc.set_title("(C) heads masked", fontsize=7.2, loc="left")
    axc.set_ylim(6, None)
    axc.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axc.set_xlim(0, 2000)
    axc.set_xticks([0, 1000, 2000])
    axc.set_xticklabels(["0", "1k", "2k"])
    axc.set_xlabel("updates", fontsize=6.5, labelpad=1)

    # (D) endpoint ablation of recruited heads
    axd = fig.add_subplot(gs[1, 3:5])
    for size, marker in (("410m", "o"), ("1b", "s")):
        f = EVID / f"regrowth_heads_v1/regrowth_{size}.json"
        if not f.exists():
            continue
        rep = json.load(open(f))
        for key, rows in rep["ablation"].items():
            for kind, color in (("regrown", SUPP), ("matched_low", CONTROL)):
                pts = sorted(
                    (r["n_added"], r["target"], r["heldout_induction"])
                    for r in rows
                    if r["added"] == kind or r["n_added"] == 0
                )
                pts = np.array(pts)
                pts[:, 0] = [[0, 2, 4, 8, 16].index(int(v)) for v in pts[:, 0]]
                axd.plot(pts[:, 0], pts[:, 1], color=color, lw=1.0, marker=marker, ms=2.6)
                axd.plot(
                    pts[:, 0],
                    pts[:, 2],
                    color=color,
                    lw=0.8,
                    ls=(0, (2.2, 1.4)),
                    marker=marker,
                    ms=2.0,
                    mfc="white",
                )
    axd.set_xticks(range(5))
    axd.set_xticklabels(["0", "2", "4", "8", "16"])
    axd.set_ylim(-0.03, 1.04)
    axd.set_yticks([0, 0.5, 1])
    axd.set_xlabel("additional heads masked", fontsize=6.5, labelpad=1)
    axd.set_title("(D) masking rebuilt heads", fontsize=7.2, loc="left")
    axd.legend(
        handles=[
            Line2D([], [], color=SUPP, lw=1.0, label="rebuilt heads"),
            Line2D([], [], color=CONTROL, lw=1.0, label="low-attention"),
            Line2D([], [], color=GRAY, lw=1.0, label="target"),
            Line2D([], [], color=GRAY, lw=0.8, ls=(0, (2.2, 1.4)), label="induction"),
        ],
        frameon=False,
        fontsize=5.8,
        loc="center right",
        ncol=1,
        handlelength=1.8,
        bbox_to_anchor=(1.0, 0.5),
    )
    out = Path(__file__).resolve().parents[2] / "outputs/figures/fig_scaling.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
