"""Command-line inspection and the small CPU example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .monitor import Trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="Inspect saved trajectories without a GPU")
    report.add_argument("records", nargs="+", type=Path)
    report.add_argument("--html", type=Path)
    report.add_argument("--plot", type=Path)
    report.add_argument("--open", action="store_true", help="Open the HTML report in a browser")
    demo = commands.add_parser("demo", help="Run an illustrative CPU removal/rescue experiment")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.command == "report":
        if args.open and args.html is None:
            parser.error("--open requires --html PATH")
        trajectories = []
        for path in args.records:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                records = value if isinstance(value, list) else [value]
                if not records:
                    raise ValueError("Empty record collection")
                trajectories.extend(Trajectory.from_dict(record) for record in records)
            except (OSError, ValueError, TypeError, KeyError) as error:
                parser.error(f"{path}: {error}")
        if args.html:
            from .reporting import write_html

            write_html(trajectories, args.html)
            if args.open:
                import webbrowser

                webbrowser.open(args.html.resolve().as_uri())
        if args.plot:
            try:
                import matplotlib  # noqa: F401
            except ModuleNotFoundError:
                parser.error("Plotting requires: pip install 'prerequisite-circuits[plot]'")
            from .reporting import plot

            plot(trajectories, args.plot)
        print(json.dumps([t.summary() for t in trajectories], indent=2))
    else:
        if args.threads < 1:
            parser.error("--threads must be positive")
        try:
            import torch
        except ModuleNotFoundError:
            parser.error("The CPU demo requires: pip install 'prerequisite-circuits[train]'")

        torch.set_num_threads(args.threads)
        from .demo import run

        trajectories = run()
        for trajectory in trajectories.values():
            trajectory.save(args.output / f"{trajectory.run_id}.json")
        from .reporting import write_html

        write_html(list(trajectories.values()), args.output / "report.html")
        print(json.dumps({name: t.summary() for name, t in trajectories.items()}, indent=2))
