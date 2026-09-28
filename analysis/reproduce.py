"""Verify release records, regenerate figures/tables, or build the paper on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def verify_inputs() -> int:
    base = ROOT / "artifacts/paper-inputs"
    manifest = json.loads((base / "manifest.json").read_text())
    for relative, expected in manifest.items():
        path = base / relative
        if base.resolve() not in path.resolve().parents:
            raise ValueError("Invalid reference path")
        actual = digest(path)
        if actual != expected["sha256"]:
            raise ValueError(f"Reference input changed: {relative}")
    return len(manifest)


def verify_release() -> int:
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Release evidence changed: {relative}")
    return len(manifest["files"])


def build_paper() -> Path:
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("Install TeX Live with pdfLaTeX; see docs/reproduction.md")
    output = ROOT / "outputs/paper"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "-halt-on-error",
        "-file-line-error",
        "-interaction=nonstopmode",
        "-no-shell-escape",
        f"-output-directory={output}",
        "main.tex",
    ]
    for _ in range(3):
        result = subprocess.run(command, cwd=ROOT / "paper", capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Paper build failed:\n{result.stdout[-6000:]}\n{result.stderr}")
    log = (output / "main.log").read_text()
    if "undefined references" in log or "Rerun to get cross-references right" in log:
        raise RuntimeError(f"Unresolved paper references; inspect {output / 'main.log'}")
    return output / "main.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "figures", "tables", "paper"])
    args = parser.parse_args()
    if args.command == "paper":
        print(build_paper())
        return
    count = verify_release()
    if args.command == "verify":
        print(json.dumps({"verified_release_files": count}))
        return
    count = verify_inputs()
    scripts = sorted((ROOT / "analysis" / args.command).glob("*.py"))
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
    )
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script)], env=environment, capture_output=True, text=True
        )
        if result.returncode:
            raise RuntimeError(f"{script.name}:\n{result.stdout}\n{result.stderr}")
        print(f"Generated {script.stem}", flush=True)
    print(
        json.dumps(
            {
                "verified_raw_files": count,
                "gpu_used": False,
                "outputs": str(ROOT / "outputs" / args.command),
            }
        )
    )


if __name__ == "__main__":
    main()
