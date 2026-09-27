"""Portable reports; JSON is authoritative and plots are derived views."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

from .monitor import Trajectory


def write_html(trajectories: Sequence[Trajectory], destination: str | Path) -> None:
    """Write an offline report with embedded data; no remote scripts or services."""
    if not trajectories:
        raise ValueError("Supply at least one trajectory")
    data = json.dumps([t.to_dict() for t in trajectories], allow_nan=False)
    # JSON inside a script element must not contain an HTML closing tag.
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = files("prerequisite_circuits").joinpath("report.html").read_text(encoding="utf-8")
    page = template.replace("__TRAJECTORY_DATA__", data)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


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
            axes[1].plot(
                x,
                [o.prerequisite_suppressed for o in observations],
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
