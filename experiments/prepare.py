"""Prepare isolated paper workspaces; this command never launches training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .artifacts import ROOT, fetch


def reference_module(root: Path):
    spec = importlib.util.spec_from_file_location("historical_reference", root / "reproduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def controlled(destination: Path, checkpoints: bool) -> Path:
    source = fetch("reference-source")
    if checkpoints:
        weights = fetch("controlled-checkpoints")
        shutil.copytree(weights / "artifacts", source / "artifacts", dirs_exist_ok=True)
    reference = reference_module(source)
    return reference.materialize(destination, source, artifacts=checkpoints)


def pythia(destination: Path, hours: float) -> Path:
    if destination.exists():
        raise FileExistsError(destination)
    source = fetch("reference-source")
    scaling = fetch("scaling-source")
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("artifacts", "build", "__pycache__")
    )
    for name in ["extensions", "code", "provenance", "environments"]:
        shutil.copytree(scaling / name, destination / name, dirs_exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(destination / "reproduce.py"),
            "workspace",
            str(destination / "runs/pythia-scaling"),
            "--code-only",
        ],
        check=True,
    )
    config = json.loads((destination / "extensions/pythia_scaling/campaign.json").read_text())
    # A new execution gets a new operational deadline, leaving scientific settings intact.
    config["deadline_utc"] = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
    configuration = destination / "replication.json"
    configuration.write_text(json.dumps(config, indent=2) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["controlled", "pythia"])
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--checkpoints", action="store_true", help="Include controlled starting states"
    )
    parser.add_argument(
        "--hours", type=float, default=72, help="Operational deadline for a new Pythia execution"
    )
    args = parser.parse_args()
    destination = args.destination.resolve()
    if args.hours <= 0:
        parser.error("--hours must be positive")
    if (
        destination == ROOT
        or ROOT in destination.parents
        and destination.parts[len(ROOT.parts)] != "outputs"
    ):
        parser.error("Use outputs/ or a directory outside the release for execution workspaces")
    path = (
        controlled(destination, args.checkpoints)
        if args.study == "controlled"
        else pythia(destination, args.hours)
    )
    print(f"Prepared {path}; no training was started.")
    if args.study == "pythia":
        print(
            f"Set PYTHIA_SCALING_CONFIG to {path / 'replication.json'} before using the archived runners."
        )


if __name__ == "__main__":
    main()
