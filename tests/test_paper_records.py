import hashlib
import json
from pathlib import Path

import pytest

from prerequisite_circuits import Trajectory


@pytest.mark.artifact
def test_compact_trajectories_match_independent_raw_prediction_records():
    root = Path(__file__).resolve().parents[1]
    inputs = root / "artifacts/paper-inputs/scaling"
    if not inputs.exists():
        pytest.skip("paper-inputs asset not downloaded")
    count = 0
    for path in (root / "results/reference").glob("*.json"):
        trajectory = Trajectory.load(path)
        source = inputs / trajectory.metadata["source_record_path"]
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == trajectory.metadata["source_record_sha256"]
        )
        raw = json.loads(source.read_text())
        assert len(raw["records"]) == len(raw["diagnostic_records"]) == len(trajectory.observations)
        for observation, reading, diagnostic in zip(
            trajectory.observations, raw["records"], raw["diagnostic_records"]
        ):
            assert observation.step == reading["step"] == diagnostic["step"]
            assert observation.target_available == reading["mc_acc"]
            assert observation.prerequisite_available == reading["ind_acc"]
            assert observation.prerequisite_suppressed == diagnostic["ind"]["accuracy"]
            assert observation.target_suppressed == diagnostic["mc"]["accuracy"]
            for probe in [
                reading["predictions"]["mc"],
                reading["predictions"]["ind"],
                diagnostic["mc"],
                diagnostic["ind"],
            ]:
                assert sum(probe["correct"]) / len(probe["correct"]) == probe["accuracy"]
            assert diagnostic["heads"] == trajectory.metadata["diagnostic_heads"]
        assert raw["protocol"]["heads"] == trajectory.metadata["training_heads"]
        count += 1
    assert count == 18
