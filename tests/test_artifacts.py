import importlib.util
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "release_artifacts", Path(__file__).resolve().parents[1] / "experiments/artifacts.py"
)
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../escape", "a\\escape"])
def test_archive_path_escape_rejected(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr(name, "test")
    with pytest.raises(ValueError):
        artifacts.extract(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_archive_is_transactional_and_never_overwrites(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("source/example.txt", "original")
    target = tmp_path / "output"
    artifacts.extract(archive, target)
    assert (target / "source/example.txt").read_text() == "original"
    with pytest.raises(FileExistsError):
        artifacts.extract(archive, target)
