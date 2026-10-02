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
