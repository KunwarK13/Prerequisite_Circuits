"""Portable reports; JSON is authoritative and plots are derived views."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path

from .monitor import Trajectory


def write_html(trajectories: Sequence[Trajectory], destination: str | Path) -> None:
    rows = []
    details = []
    for trajectory in trajectories:
        result = trajectory.summary()
        values = [
            trajectory.run_id,
            result["status"],
            result["observed_through"],
            result["first_observed_acquisition"],
            result["first_observed_loss_of_effectiveness"],
            result["missing_checks"],
        ]
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(v)) if v is not None else 'Not observed'}</td>"
                for v in values
            )
            + "</tr>"
        )
        details.append(
            "<details><summary>"
            + html.escape(trajectory.run_id)
            + ": criterion and context</summary><pre>"
            + html.escape(
                json.dumps(
                    {
                        "criterion": result["criterion"],
                        "training_intervention": result["training_intervention"],
                        "diagnostic_intervention": result["diagnostic_intervention"],
                        "metadata": trajectory.metadata,
                    },
                    indent=2,
                    allow_nan=False,
                )
            )
            + "</pre></details>"
        )
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intervention effectiveness</title>
<style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;color:#192735}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:.8rem;border-bottom:1px solid #ddd}
th{background:#eef3f6}p{line-height:1.6} .scroll,pre{overflow:auto}
details{margin:1rem 0}summary{cursor:pointer}</style>
<h1>Intervention effectiveness during learning</h1>
<p>Each result uses its recorded probe criterion and observation grid. “Not observed”
does not mean impossible. Effectiveness between observations is unmeasured.
Temporal order alone does not establish that prerequisite recovery causes acquisition.</p>
<div class="scroll"><table><thead><tr><th>Run</th><th>Probe criterion</th><th>Last update</th>
<th>First acquisition</th><th>First loss of effectiveness</th><th>Missing checks</th>
</tr></thead><tbody>"""
        + "".join(rows)
        + "</tbody></table></div>"
        + "".join(details)
        + "</html>"
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)


def plot(trajectories: Sequence[Trajectory], destination: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    ):
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        for trajectory in trajectories:
            observations = trajectory.observations
            label = trajectory.metadata.get("label", trajectory.run_id)
            x = [o.step for o in observations]
            (line,) = axes[0].plot(x, [o.target_available for o in observations], label=label)
            measured = [o for o in observations if o.prerequisite_suppressed is not None]
            axes[1].plot(
                [o.step for o in measured],
                [o.prerequisite_suppressed for o in measured],
                color=line.get_color(),
                label=label,
            )
        for ax, title in zip(
            axes,
            [
                "Target: training intervention removed",
                "Prerequisite: diagnostic intervention active",
            ],
        ):
            ax.set(title=title, xlabel="Training updates", ylabel="Accuracy", ylim=(-0.02, 1.02))
            ax.legend(frameon=False, fontsize=8)
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)
