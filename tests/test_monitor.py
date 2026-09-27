import json

import pytest

from prerequisite_circuits import Criterion, Observation, Trajectory
from prerequisite_circuits.reporting import write_html


def test_loss_of_suppression_is_reported_separately_from_acquisition(tmp_path):
    trajectory = Trajectory("compensation")
    for point in [(0, 0.01, 0.7, 0.05), (50, 0.03, 0.7, 0.4), (100, 0.99, 0.8, 0.6)]:
        trajectory.append(Observation(*point))
    summary = trajectory.summary()
    assert summary["first_observed_loss_of_effectiveness"] == 50
    assert summary["first_observed_acquisition"] == 100
    assert summary["status"] == "effectiveness_lost_at_observed_step"
    path = tmp_path / "run.json"
    trajectory.save(path)
    assert Trajectory.load(path).to_dict() == trajectory.to_dict()


def test_no_acquisition_is_censored_at_horizon():
    trajectory = Trajectory(
        "closed", observations=[Observation(0, 0.0, 0.9, 0.1), Observation(100, 0.01, 0.8, 0.1)]
    )
    result = trajectory.summary()
    assert result["first_observed_acquisition"] is None
    assert result["observed_through"] == 100
    assert result["status"] == "effective_at_observed_steps"


def test_probe_floor_is_not_a_successful_intervention():
    trajectory = Trajectory("already-impaired", observations=[Observation(0, 0.01, 0.08, 0.02)])
    assert trajectory.summary()["status"] == "criterion_not_met"


def test_missing_checks_remain_visible():
    trajectory = Trajectory("missing", observations=[Observation(0, 0.01), Observation(100, 0.9)])
    assert trajectory.summary()["status"] == "unmeasured"
    assert trajectory.summary()["missing_checks"] == 2


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_invalid_accuracy_rejected(value):
    with pytest.raises(ValueError):
        Observation(0, value)


def test_ordering_and_nonfinite_metadata_rejected():
    trajectory = Trajectory("ordered", observations=[Observation(2, 0.3)])
    with pytest.raises(ValueError):
        trajectory.append(Observation(2, 0.4))
    with pytest.raises(ValueError):
        Trajectory("bad", metadata={"x": float("nan")})


def test_report_recomputes_summary_and_escapes_user_labels(tmp_path):
    path = tmp_path / "run.json"
    trajectory = Trajectory("<script>alert(1)</script>", observations=[Observation(0, 0.1)])
    trajectory.save(path)
    value = json.loads(path.read_text())
    value["summary"] = {"status": "proven"}
    path.write_text(json.dumps(value))
    assert Trajectory.load(path).summary()["status"] == "unmeasured"
    output = tmp_path / "report.html"
    write_html([trajectory], output)
    assert "<script>alert(1)</script>" not in output.read_text()
    assert "\\u003cscript\\u003e" in output.read_text()


def test_threshold_is_inclusive_and_criterion_is_recorded():
    criterion = Criterion(0.25, 0.25, 0.75)
    assert criterion.satisfied(Observation(0, 0.75, 0.5, 0.25))
    assert not criterion.satisfied(Observation(0, 0.75, 0.49, 0.25))


def test_missing_threshold_rejected():
    with pytest.raises(ValueError, match="required"):
        Criterion(maximum_suppressed=None)


def test_report_embeds_validated_records_and_rejects_empty_input(tmp_path):
    from html.parser import HTMLParser

    class DataParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.reading = False
            self.data = ""

        def handle_starttag(self, tag, attributes):
            self.reading = tag == "script" and dict(attributes).get("id") == "records"

        def handle_data(self, data):
            if self.reading:
                self.data += data

        def handle_endtag(self, tag):
            if tag == "script":
                self.reading = False

    trajectory = Trajectory(
        "</script><script>alert('x')</script>", observations=[Observation(0, 0.1)]
    )
    destination = tmp_path / "report.html"
    write_html([trajectory], destination)
    parser = DataParser()
    parser.feed(destination.read_text())
    assert json.loads(parser.data) == [trajectory.to_dict()]
    with pytest.raises(ValueError, match="at least one"):
        write_html([], destination)


def test_cli_accepts_exported_collection(tmp_path, monkeypatch, capsys):
    from prerequisite_circuits.cli import main

    path = tmp_path / "records.json"
    trajectory = Trajectory("record", observations=[Observation(0, 0.3)])
    path.write_text(json.dumps([trajectory.to_dict(), trajectory.to_dict()]))
    monkeypatch.setattr("sys.argv", ["prereq", "report", str(path)])
    main()
    assert json.loads(capsys.readouterr().out) == [trajectory.summary(), trajectory.summary()]
