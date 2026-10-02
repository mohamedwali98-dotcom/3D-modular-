"""The in-memory job registry (audit H3, H4): a job being reviewed outlives newer ones, and a job has a time
budget."""
import time

from s2c.multiview.pipeline import MvPipeline
from s2c.web import jobs


def _done() -> jobs.Job:
    job = jobs.new_job(1, MvPipeline())
    job.status = "done"
    return job


def test_the_least_recently_used_job_goes_first(monkeypatch):
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "MAX_JOBS", 3)
    first = _done()
    _done(), _done()
    jobs.touch(first)  # the user is still on Review
    _done()
    assert jobs.get_job(first.job_id) is first


def test_the_time_to_live_counts_from_last_use(monkeypatch):
    monkeypatch.setattr(jobs, "JOBS", {})
    job = _done()
    job.created = time.time() - 2 * jobs.TTL_S
    jobs.touch(job)
    jobs.sweep_jobs()
    assert jobs.get_job(job.job_id) is job


def test_a_job_past_its_budget_stops_with_a_plain_message(monkeypatch):
    from pathlib import Path

    from s2c.multiview.pipeline import ImageInput
    png = (Path(__file__).resolve().parents[1] / "examples" / "mv" / "sketches" / "front.png").read_bytes()
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "JOB_BUDGET_S", 0)
    job = jobs.new_job(1, MvPipeline())
    jobs.run(job, MvPipeline(), [ImageInput(png, "front", "sketch")], None)
    assert job.status == "failed" and job.error == jobs.TOO_LONG


def _front_png() -> bytes:
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / "examples" / "mv" / "sketches" / "front.png").read_bytes()


def test_an_abstained_job_keeps_no_image_it_will_never_use(monkeypatch):
    """Waiting on Review for typed sizes is the usual end of an analysis without a reader: with no hosted helper
    on, fuse never looks at the images again, so they are not kept (200 such jobs once held about 7 GB)."""
    from s2c.multiview.pipeline import ImageInput
    from s2c.multiview.settings import AiSettings
    monkeypatch.setattr(jobs, "JOBS", {})
    pipe = MvPipeline().configured(AiSettings())
    job = jobs.new_job(1, pipe)
    jobs.run(job, pipe, [ImageInput(_front_png(), "front", "sketch")], None)
    assert job.result["abstain"] is not None and job.observed.images == []


def test_a_job_that_may_draw_a_face_keeps_its_images(monkeypatch):
    from s2c.multiview.pipeline import ImageInput
    from s2c.multiview.settings import AiSettings
    monkeypatch.setattr(jobs, "JOBS", {})
    pipe = MvPipeline(image_gen=lambda *a, **k: None).configured(AiSettings(use_qwen_image=True))
    job = jobs.new_job(1, pipe)
    jobs.run(job, pipe, [ImageInput(_front_png(), "front", "sketch")], None)
    assert job.result["abstain"] is not None and len(job.observed.images) == 1


def test_a_request_turned_away_evicts_nothing(monkeypatch):
    """An analysis refused because MAX_RUNNING already run must not push a job someone is reviewing out."""
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "MAX_JOBS", 1)
    monkeypatch.setattr(jobs, "_ACTIVE", {c * 32 for c in "abc"})
    reviewed = _done()
    job = jobs.new_job(1, MvPipeline(), register=False)
    assert not jobs.start(job, MvPipeline(), [], None)
    assert jobs.get_job(reviewed.job_id) is reviewed and jobs.get_job(job.job_id) is None


def test_the_budget_stops_a_job_before_a_stage_never_after_one(monkeypatch):
    """A stage that finished keeps its result: past the budget, the job stops before the next one starts."""
    import pytest
    monkeypatch.setattr(jobs, "JOB_BUDGET_S", 0)
    job = jobs.new_job(1, MvPipeline(), register=False)
    jobs._check(job, {"key": "fuse", "state": "done"})
    with pytest.raises(jobs.JobTimeout):
        jobs._check(job, {"key": "fuse", "state": "running"})


def test_the_budget_ignores_a_jump_of_the_wall_clock():
    job = jobs.new_job(1, MvPipeline(), register=False)
    job.created -= 10 * jobs.JOB_BUDGET_S  # the system clock jumped forward (NTP)
    jobs._check(job, {"key": "draw", "state": "running"})
