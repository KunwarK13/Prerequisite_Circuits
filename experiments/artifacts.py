"""Download pinned paper assets and extract them without modifying reference files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def extract(archive: Path, destination: Path) -> None:
    """Reject traversal, symlinks, duplicate names, and existing destinations."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for entry in source.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                raise ValueError("Unsafe archive path")
            mode = entry.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not supported")
        temporary = Path(tempfile.mkdtemp(prefix=".extract-", dir=destination.parent))
        try:
            source.extractall(temporary)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def fetch(name: str, *, local: Path | None = None) -> Path:
    manifest = json.loads((ROOT / "results/artifacts.json").read_text())
    asset = manifest["assets"][name]
    cache = ROOT / "artifacts"
    cache.mkdir(exist_ok=True)
    archive = cache / asset["filename"]
    if local is not None:
        if digest(local) != asset["sha256"]:
            raise ValueError("Local asset does not match the release checksum")
        if local.resolve() != archive.resolve():
            shutil.copy2(local, archive)
    if not archive.exists():
        if not asset.get("url", "").startswith("https://"):
            raise ValueError("Asset needs a published HTTPS URL or --local archive")
        fd, filename = tempfile.mkstemp(prefix=".download-", dir=cache)
        os.close(fd)
        temporary = Path(filename)
        try:
            with (
                urllib.request.urlopen(asset["url"], timeout=60) as response,
                temporary.open("wb") as out,
            ):
                shutil.copyfileobj(response, out, length=8 << 20)
            if digest(temporary) != asset["sha256"]:
                raise ValueError("Downloaded asset checksum does not match")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if archive.stat().st_size != asset["bytes"] or digest(archive) != asset["sha256"]:
        raise ValueError("Cached archive failed integrity verification")
    destination = cache / name
    if not destination.exists():
        extract(archive, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name",
        choices=["paper-inputs", "reference-source", "controlled-checkpoints", "scaling-source"],
    )
    parser.add_argument("--local", type=Path, help="Import an already downloaded, matching ZIP")
    args = parser.parse_args()
    print(fetch(args.name, local=args.local))


if __name__ == "__main__":
    main()
