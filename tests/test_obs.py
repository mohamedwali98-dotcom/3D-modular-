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


def test_provider_fallbacks_are_counted_where_they_happen(monkeypatch):
    import numpy as np

    from s2c.multiview import hf3d
    from s2c.reading import Crop, ReadingService

    def down(img):
        raise RuntimeError("down")

    class Broken:
        name = "qwen"

        def read(self, crops):
            raise RuntimeError("boom")

    obs.reset()
    monkeypatch.setattr(hf3d, "local_triposr", down)
    monkeypatch.setattr(hf3d, "space_triposr", down)
    with pytest.raises(RuntimeError):
        hf3d.default_provider()(np.zeros((8, 8, 3), np.uint8))
    ReadingService([Broken()]).read([Crop(np.zeros((20, 30, 3), np.uint8), (0, 0, 30, 20))])
    text = obs.render()
    assert 's2c_fallbacks_total{name="triposr_local"} 1' in text and "triposr_space" not in text  # the caller counts it
    assert 's2c_fallbacks_total{name="reader_qwen"} 1' in text


def test_each_metric_family_says_its_type_once_before_its_lines():
    obs.reset()
    obs.count("s2c_jobs_total", mode="photos", outcome="done")
    obs.count("s2c_jobs_total", mode="sheet", outcome="done")
    obs.time_spent("s2c_stage_seconds", 1.0, stage="fuse")
    lines = obs.render().splitlines()
    assert lines.count("# TYPE s2c_jobs_total counter") == 1 and lines.count("# TYPE s2c_stage_seconds summary") == 1
    assert lines.index("# TYPE s2c_jobs_total counter") < lines.index('s2c_jobs_total{mode="photos",outcome="done"} 1')
    assert lines.index("# TYPE s2c_stage_seconds summary") < lines.index('s2c_stage_seconds_sum{stage="fuse"} 1')


def test_a_reader_out_of_time_and_trocr_leaving_the_gpu_are_counted(monkeypatch):
    import time

    import numpy as np

    from s2c.reading import Crop, ReaderResult, ReadingService, trocr

    class Slow:
        name, timeout_s = "qwen", 0.1

        def read(self, crops):
            time.sleep(0.5)
            return [ReaderResult(text="1", confidence=1.0) for _ in crops]

    obs.reset()
    crop = Crop(np.zeros((20, 30, 3), np.uint8), (0, 0, 30, 20))
    ReadingService([Slow()]).read([crop])
    reader = trocr.TrocrReader.__new__(trocr.TrocrReader)
    reader.model_id, reader.device = "m", "cuda"

    def load(model_id, device):
        if device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return None, None

    monkeypatch.setattr(trocr, "_load", load)
    monkeypatch.setattr(trocr.TrocrReader, "_batched", lambda self, p, m, d, crops: [])
    reader.read([crop])
    text = obs.render()
    assert 's2c_fallbacks_total{name="reader_qwen_timeout"} 1' in text and 's2c_fallbacks_total{name="trocr_gpu"} 1' in text


def test_work_handed_to_another_thread_keeps_the_analysis_id(stream):
    import threading

    import numpy as np

    from s2c.reading import Crop, ReaderResult, ReadingService

    class Talking:
        name = "talking"

        def read(self, crops):
            logging.getLogger("s2c.test").warning("reading")
            return [ReaderResult(text="1", confidence=1.0) for _ in crops]

    with obs.job_scope("job42"):
        ReadingService([Talking()]).read([Crop(np.zeros((20, 30, 3), np.uint8), (0, 0, 30, 20))])
        carried = threading.Thread(target=obs.carry(lambda: logging.getLogger("s2c.test").warning("carried")))
        carried.start()
        carried.join()
    records = [json.loads(line) for line in stream.getvalue().strip().splitlines()]
    assert [r.get("job_id") for r in records if r["message"] in ("reading", "carried")] == ["job42", "job42"]


def test_an_operators_own_root_handler_is_kept_and_gets_the_job_id():
    """uvicorn --log-config (or any dictConfig) that gives root a handler wins: no second handler printing every
    line twice, and its records still carry the analysis id."""
    root = logging.getLogger()
    saved = root.handlers[:]
    out = io.StringIO()
    theirs = logging.StreamHandler(out)
    theirs.setFormatter(logging.Formatter("%(job_id)s %(message)s"))
    root.handlers = [theirs]
    try:
        obs.configure_logging()
        assert root.handlers == [theirs]
        with obs.job_scope("j7"):
            logging.getLogger("s2c.test").warning("hello")
    finally:
        root.handlers = saved
    assert out.getvalue().strip() == "j7 hello"


def test_an_operators_own_s2c_level_is_kept(monkeypatch):
    root, ours = logging.getLogger(), logging.getLogger("s2c")
    saved, level = root.handlers[:], ours.level
    monkeypatch.delenv("S2C_LOG_LEVEL", raising=False)
    root.handlers = [logging.StreamHandler(io.StringIO())]
    ours.setLevel(logging.WARNING)
    try:
        obs.configure_logging()
        assert ours.level == logging.WARNING
    finally:
        root.handlers = saved
        ours.setLevel(level)


def test_one_name_is_one_kind_of_metric():
    obs.reset()
    obs.count("s2c_mixed_total")
    with pytest.raises(ValueError):
        obs.time_spent("s2c_mixed_total", 1.0)


def test_a_record_after_the_analysis_has_no_id(stream):
    from s2c.multiview import hf3d

    def say(message):
        logging.getLogger("s2c.test").warning(message)

    with obs.job_scope("job9"):
        hf3d._pool.submit(obs.carry(say), "carried").result()
    hf3d._pool.submit(say, "plain on the same worker").result()
    say("after, same thread")
    records = {r["message"]: r.get("job_id") for r in map(json.loads, stream.getvalue().strip().splitlines())}
    assert records["carried"] == "job9"
    assert records["plain on the same worker"] is None and records["after, same thread"] is None


def test_label_values_are_escaped():
    obs.reset()
    obs.count("s2c_fallbacks_total", name='a"b\\c\nd')
    assert 's2c_fallbacks_total{name="a\\"b\\\\c\\nd"} 1' in obs.render()
