"""Generate LaTeX tables for the new analyses directly from raw JSON records."""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2] / "artifacts/paper-inputs/scaling"
R = REPO / "results/extensions/pythia_scaling/followup-v2-rebuild-evidence-v1/runs/scaling"
OUT = Path(__file__).resolve().parents[2] / "outputs/tables"


def fmt(x, d=2):
    return f"{x:.{d}f}"


def first_acquisition(records):
    """First observed update satisfying all three target-learning criteria."""
    for record in records:
        keys = record["primary_keys"]
        if (
            record["mc_acc"] >= 0.8
            and keys["pair_both_correct"] >= 0.8
            and record["mc_acc"] - keys["broken"]["accuracy"] >= 0.25
        ):
            return record["step"]
    return None


def regrowth_tables():
    rows_abl, rows_rank = [], []
    for size, label in (("410m", "410M"), ("1b", "1B")):
        f = R / f"regrowth_heads_v1/regrowth_{size}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        orig = {tuple(h) for h in d["original"]}
        intact = np.array(d["states"]["initial"]["intact_attention"])
        forced0 = np.array(d["states"]["initial"]["forced_attention"])
        L, H = intact.shape
        rank = {divmod(int(i), H): r + 1 for r, i in enumerate(np.argsort(-intact.ravel()))}
        for key, rows in d["ablation"].items():
            stream = key.replace("tape", "").replace("_circuit", "")
            by = {(r["n_added"], r["added"]): r for r in rows}
            for n in (0, 2, 4, 8, 16):
                g = by[(n, "regrown")]
                m = by.get((n, "matched_low"), g)
                rows_abl.append(
                    f"{label} & {stream} & {n} & {fmt(g['heldout_induction'])} / {fmt(m['heldout_induction'])} & "
                    f"{fmt(g['target'])} / {fmt(m['target'])} & {fmt(g['pair_both'])} / {fmt(m['pair_both'])} \\\\"
                )
        for key in [k for k in d["states"] if k != "initial"]:
            st = d["states"][key]
            end = np.array(st["forced_attention"])
            um = [(layer, h) for layer in range(L) for h in range(H) if (layer, h) not in orig]
            top = sorted(um, key=lambda x: -end[x])[:4]
            tape, arm = key.replace("tape", "").split("_")
            heads = ", ".join(f"{layer}.{h}" for layer, h in top)
            ranks = ", ".join(str(rank[t]) for t in top)
            init_att = ", ".join(fmt(forced0[t]) for t in top)
            end_att = ", ".join(fmt(end[t]) for t in top)
            rows_rank.append(
                f"{label} & {tape} & {arm} & {heads} & {ranks} & {init_att} & {end_att} & "
                f"{fmt(st['forced_heldout_induction'])} \\\\"
            )
    (OUT / "tab_regrowth_ablation.tex").write_text("\n".join(rows_abl) + "\n")
    (OUT / "tab_regrowth_rank.tex").write_text("\n".join(rows_rank) + "\n")


def capacity_table():
    f = R / "regrowth_heads_v1/capacity.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    rows = []
    for size, label in (
        ("160m", "160M"),
        ("410m", "410M"),
        ("1b", "1B"),
        ("1.4b", "1.4B"),
        ("2.8b", "2.8B"),
    ):
        if size not in d:
            continue
        c = d[size]
        intact = np.array(c["intact_attention"])
        L, H = intact.shape
        orig = {tuple(h) for h in c["original"]}
        um = np.array(
            [intact[layer, h] for layer in range(L) for h in range(H) if (layer, h) not in orig]
        )
        top = np.sort(um)[::-1]
        rows.append(
            f"{label} & {L * H} & {len(orig)} & {fmt(c['intact_heldout_induction'])} & {fmt(c['forced_heldout_induction'])} & "
            f"{int((um >= 0.1).sum())} & {int((um >= 0.2).sum())} & {fmt(top[0])} \\\\"
        )
    rep = R / "regrowth_heads_v1/capacity_410m_replicates.json"
    if rep.exists():
        c = json.loads(rep.read_text())
        for s in (1, 2, 3):
            r = c[f"410m-seed{s}"]
            rows.append(
                f"410M, run {s} & 384 & {r['suppressed']} & {fmt(r['intact_heldout'])} & {fmt(r['forced_heldout'])} & "
                f"{r['n_ge_0.1']} & {r['n_ge_0.2']} & {fmt(r['top5'][0][1])} \\\\"
            )
    (OUT / "tab_capacity.tex").write_text("\n".join(rows) + "\n")


def adaptive_summary():
    out = {}
    for p in sorted((R / "adaptive_suppression_v1").glob("*.json")):
        d = json.loads(p.read_text())
        recs = d["records"]
        name = Path(p).stem
        last = recs[-1]
        out[name] = {
            "first_acquired": first_acquisition(recs),
            "final_heads": len(d["final_heads"]),
            "additions": [(a["step"], a["n_heads"]) for a in d["additions"]],
            "end_target": last["mc_acc"],
            "end_pair": last["primary_keys"]["pair_both_correct"],
            "end_deleted": last["primary_keys"]["broken"]["accuracy"],
            "end_masked_ind_original": last["forced_original"]["ind"]["accuracy"],
            "end_masked_ind_current": last["forced_current"]["ind"]["accuracy"],
            "max_masked_ind_current": max(r["forced_current"]["ind"]["accuracy"] for r in recs),
            "end_nat": last["nat_ce"],
            "updates": d.get("updates"),
            "max_target": max(r["mc_acc"] for r in recs),
        }
    (OUT / "adaptive_summary.json").write_text(json.dumps(out, indent=1))


def run_summary_table():
    """One row per new run, all computed from raw records with the paper's criterion."""

    groups = [
        ("160M, 4 heads, 10,000 updates", "160m_tape{t}_circuit_trace10k"),
        ("160M, top 2 heads", "160m_tape{t}_circuit_top2"),
        ("160M, top 3 heads", "160m_tape{t}_circuit_top3"),
        ("160M, 16 loss-matched heads", "160m_tape{t}_circuit_lossmatched"),
        ("410M, 13 heads (attention $\\ge$0.2)", "410m_tape{t}_circuit_attn02circuit"),
        ("410M, 13 matched control heads", "410m_tape{t}_circuit_attn02control"),
        ("410M, loss-matched heads", "410m_tape{t}_circuit_lossmatched"),
        ("410M, adaptive (induction trigger)", "410m_tape{t}_circuit"),
        ("410M, adaptive (key trigger)", "410m_tape{t}_circuit_keyroute"),
        ("410M, yoked control", "410m_tape{t}_yoked_keyroute"),
        ("410M, static control (22 / 16 heads)", "410m_tape{t}_circuit_keyroutestatic"),
        ("2.8B, 4 heads, 2,000 updates", "2.8b_tape{t}_circuit_trace"),
    ]
    rows = []
    for label, pat in groups:
        cells = []
        for t in (38, 39):
            f = R / "adaptive_suppression_v1" / (pat.format(t=t) + ".json")
            if not f.exists():
                cells.append(None)
                continue
            d = json.loads(f.read_text())
            recs = d["records"]
            last = recs[-1]
            cells.append(
                {
                    "first": first_acquisition(recs),
                    "target": last["mc_acc"],
                    "pair": last["primary_keys"]["pair_both_correct"],
                    "masked": last["forced_current"]["ind"]["accuracy"],
                    "heads": len(d["final_heads"]),
                    "updates": recs[-1]["step"],
                }
            )
        if all(c is None for c in cells):
            continue

        def g(key, f=lambda v: fmt(v)):
            return " / ".join(
                "--"
                if c is None
                else (
                    ("never" if c[key] is None else f"{c[key]:,}") if key == "first" else f(c[key])
                )
                for c in cells
            )

        rows.append(
            f"{label} & {g('heads', lambda v: str(v))} & {g('updates', lambda v: f'{v:,}')} & {g('first')} & {g('target')} & {g('pair')} & {g('masked')} \\\\"
        )
    (OUT / "tab_new_runs.tex").write_text("\n".join(rows) + "\n")


def replicate_table():
    rows = []
    for s in (1, 2, 3):
        cells = []
        for t in (38, 39):
            f = R / "adaptive_suppression_v1" / f"410m-seed{s}_tape{t}_circuit_trace.json"
            if not f.exists():
                cells.append(None)
                continue
            d = json.loads(f.read_text())
            recs = d["records"]
            fm = next(
                (r["step"] for r in recs if r["forced_original"]["ind"]["accuracy"] > 0.2), None
            )
            cells.append(
                (
                    first_acquisition(recs),
                    recs[-1]["mc_acc"],
                    recs[-1]["forced_original"]["ind"]["accuracy"],
                    fm,
                    len(d["final_heads"]),
                )
            )
        if all(c is None for c in cells):
            continue

        def g(i, f):
            return " / ".join(
                "--" if c is None else ("never" if c[i] is None else f(c[i])) for c in cells
            )

        heads = next(c[4] for c in cells if c)
        rows.append(
            f"run {s} & {heads} & {g(0, lambda v: f'{v:,}')} & {g(3, lambda v: f'{v:,}')} & {g(1, fmt)} & {g(2, fmt)} \\\\"
        )
    (OUT / "tab_replicates.tex").write_text("\n".join(rows) + "\n")


def newseed_table():
    f = (
        REPO
        / "results/extensions/controlled_followup/controlled-followup-seeds-v1/runs/controlled_followup_v1/result.json"
    )
    if not f.exists():
        return
    d = json.loads(f.read_text())
    rows = []
    for r in d["new_seed_summary"]:

        def g(k):
            return f"{r[k]['copy_gain']:.2f} / {r[k]['d64']:.2f}"

        rows.append(
            f"{r['seed']} & {r['formation_global_update']:,} & {r['upstream_head'][0]}.{r['upstream_head'][1]} & "
            f"{g('SIR_OPEN')} & {g('SIR_CONTROL')} & {g('SIR_CLOSED')} & {g('RESCUE_SUPPLY')} & {g('RESCUE_NONE')} & "
            f"{g('RESCUE_WRONG')} & {r['SIR_CLOSED_REPLAY']['copy_gain']:.2f} \\\\"
        )
    (OUT / "tab_newseeds.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    regrowth_tables()
    capacity_table()
    adaptive_summary()
    run_summary_table()
    replicate_table()
    newseed_table()
