"""Exercise the offline report with a real browser; no model dependencies."""

import json
import os

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from prerequisite_circuits import Observation, Trajectory
from prerequisite_circuits.reporting import write_html


def test_offline_report_controls_data_export_and_mobile_layout(tmp_path):
    trajectory = Trajectory(
        "suppress",
        metadata={
            "training_intervention": "training mask",
            "diagnostic_intervention": "diagnostic mask",
        },
        observations=[
            Observation(0, 0.1, 0.9, 0.1, 0.0),
            Observation(50, 0.8),
            Observation(100, 0.9, 0.8, 0.4, 0.8),
        ],
    )
    injected = Trajectory(
        "</script><script>window.injected=true</script>",
        observations=[Observation(0, 0.2)],
    )
    empty = Trajectory("no observations")
    output = tmp_path / "report.html"
    write_html([trajectory, injected, empty], output)
    with playwright.sync_playwright() as session:
        options = {"headless": True}
        if os.environ.get("PREREQ_TEST_BROWSER"):
            options["executable_path"] = os.environ["PREREQ_TEST_BROWSER"]
        browser = session.chromium.launch(**options)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors, network = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: network.append(request.url))
        page.goto(output.as_uri())
        assert page.locator("#run-count").inner_text() == "3 / 3"
        assert page.locator("#outcomes tr").count() == 3
        assert page.evaluate("window.injected") is None
        assert "training mask" in page.locator("#context").inner_text()
        assert "diagnostic mask" in page.locator("#context").inner_text()
        # Missing probe readings remain separate segments, not interpolated values.
        path = page.locator("#probe-chart path").first.get_attribute("d")
        assert path.count("M") == 2
        page.locator("#clear").click()
        assert page.locator("#outcomes tr").count() == 0
        page.locator(".run-chip input").first.check()
        assert page.locator("#run-count").inner_text() == "1 / 3"
        page.locator("#inspect-step").select_option("1")
        assert "Not measured" in page.locator("#measurements").inner_text()
        with page.expect_download() as download_info:
            page.locator("#download").click()
        downloaded = download_info.value.path()
        assert json.loads(downloaded.read_text()) == [
            trajectory.to_dict(),
            injected.to_dict(),
            empty.to_dict(),
        ]
        page.locator("#inspect-run").select_option("2")
        assert page.locator("#inspect-step").is_disabled()
        assert "Not measured" in page.locator("#measurements").inner_text()
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert not errors
        assert all(url.startswith("file:") or url.startswith("blob:") for url in network)
        browser.close()
