"""Logs and metrics (audit M8, M9): records carry their analysis, fallbacks are counted, metrics are scrapable."""
import io
import json
import logging

import pytest

from s2c import obs


@pytest.fixture
def stream():
    out = io.StringIO()
    handler = obs.configure_logging(stream=out, force=True)
    yield out
    logging.getLogger().removeHandler(handler)


def test_a_record_inside_a_job_carries_its_id(stream):
    with obs.job_scope("abc123"):
        logging.getLogger("s2c.test").warning("inside")
    logging.getLogger("s2c.test").warning("outside")
    inside, outside = (json.loads(line) for line in stream.getvalue().strip().splitlines()[-2:])
    assert inside == {**inside, "job_id": "abc123", "message": "inside", "level": "WARNING", "logger": "s2c.test"}
    assert "job_id" not in outside


def test_a_fallback_is_logged_and_counted(stream):
    obs.reset()
    obs.fallback("triposr", RuntimeError("Space down"))
    obs.fallback("triposr", RuntimeError("Space down"))
    assert 's2c_fallbacks_total{name="triposr"} 2' in obs.render()
    record = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert "triposr" in record["message"] and "Space down" in record["message"]


def test_metrics_render_in_prometheus_text():
    obs.reset()
    obs.count("s2c_jobs_total", mode="photos", outcome="done")
    obs.time_spent("s2c_stage_seconds", 1.5, stage="fuse")
    text = obs.render()
    assert 's2c_jobs_total{mode="photos",outcome="done"} 1' in text
    assert 's2c_stage_seconds_sum{stage="fuse"} 1.5' in text and 's2c_stage_seconds_count{stage="fuse"} 1' in text
