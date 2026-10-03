"""Analysis jobs for the web app: one record per /api/analyze, filled by the pipeline's progress events.
Jobs live in this process's memory for one hour after their last use (so the server runs one worker); past MAX_JOBS
the least recently used finished ones go first. The job id is also the request id /api/merge and /api/model use."""
from __future__ import annotations

import itertools
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from s2c import obs
from s2c.multiview.pipeline import ImageInput, MvPipeline, Observed, forget_images
from s2c.multiview.spec import FACES, MvAbstain

log = logging.getLogger(__name__)
TTL_S = 3600
STAGES = ("label", "outline", "read", "draw", "fuse")
TOOLS = {"label": ("Vision model", True), "outline": ("OpenCV", False), "read": ("Vision model", True),
         "draw": ("Qwen-Image", True), "fuse": ("CadQuery", False)}
# "One sheet (all views)": one photo of a hand-drawn sheet replaces label/outline/read, then joins the
# same draw/fuse stages a per-face photo analysis uses (s2c/web/sketch_adapter.py).
SHEET_STAGES = ("views", "lines", "values", "draw", "fuse")
SHEET_TOOLS = {"views": ("Sketch reader", True), "lines": ("Sketch reader", True),
               "values": ("Sketch reader", True), "draw": ("Qwen-Image", True), "fuse": ("CadQuery", False)}
SHEET_LABELS = {"views": "views found", "lines": "lines classified", "values": "values read"}
ALL_STAGE_KEYS = frozenset({*STAGES, *SHEET_STAGES})
SHEET_UNAVAILABLE = ("Sheet reading is not available yet. Use per-face photos, or try again once "
                     "it is deployed.")
MAX_RUNNING = 3   # analyses running at once; past this /api/analyze answers 429
JOB_BUDGET_S = 900  # an analysis still running after this stops at its next stage
TOO_LONG = "The analysis took too long. Try again with fewer images, or with the AI helpers off."
MAX_JOBS = 200    # jobs kept in memory; the least recently used finished ones go first
DRAW_TOOLS = {"qwen-image": ("Qwen-Image", True), "triposr": ("TripoSR", True), "mirrored": ("mirror", False),
              "assumed": ("assumed", False)}
DRAW_WORDS = {"qwen-image": "drawn — check it", "triposr": "predicted — check it", "mirrored": "mirrored",
              "assumed": "assumed rectangular — check it"}
FAILED = "Analysis failed. Try again or use different photos."


class JobCancelled(Exception):
    pass


class JobTimeout(Exception):
    pass


class UserFacing(Exception):
    """A message written for the person using the app (a remedy). Only these reach the browser: any other error,
    a library's RuntimeError included, gives the fixed FAILED sentence."""

    def __init__(self, remedy: str, detail: str = ""):
        super().__init__(remedy)
        self.remedy, self.detail = remedy, detail  # detail (stage and reason slugs) goes to the log only


@dataclass
class Job:
    job_id: str
    stages: list[dict]
    images: list[dict]
    status: str = "running"
    coverage: dict[str, str] = field(default_factory=lambda: dict.fromkeys(FACES, "empty"))
    result: dict | None = None
    error: str | None = None
    observed: Observed | None = None
    created: float = field(default_factory=time.time)
    clock: float = field(default_factory=time.monotonic)  # when it started, for its time budget: no clock jumps
    used: float = field(default_factory=time.time)  # the last status poll or merge: a job on Review stays
    use: int = field(default_factory=lambda: next(_USES))
    cancel: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    pipe: MvPipeline | None = None  # the pipeline configured with this request's AI settings; merge reuses it
    mode: str = "photos"  # "photos" (per-face) or "sheet" (one sheet, all views)
    # (stage key, image index) -> when that run started: label, outline and read run once per image
    stage_clock: dict = field(default_factory=dict, repr=False)

    def stage(self, key: str) -> dict:
        return next(s for s in self.stages if s["key"] == key)

    def to_json(self) -> dict:
        with self.lock:
            return {"job_id": self.job_id, "status": self.status,
                    "stages": [dict(s) for s in self.stages],
                    "images": [{**i, "circles": list(i["circles"]), "reads": list(i["reads"])} for i in self.images],
                    "coverage": dict(self.coverage), "result": self.result, "error": self.error}


JOBS: dict[str, Job] = {}
_USES = itertools.count()  # the order of use: the clock can give two jobs the same time
_ACTIVE: set[str] = set()  # jobs whose analysis thread is running
_registry_lock = threading.Lock()


def model_name() -> str:
    """The configured vision model, read at request time so the UI names what really runs."""
    return os.environ.get("VLM_MODEL") or "Vision model"


def new_job(n_images: int, pipe: MvPipeline, register: bool = True) -> Job:
    """A photo analysis's record. `register=False` (the API) leaves it out of the registry until it may start:
    a request turned away then evicts nothing."""
    stages = []
    for key in STAGES:
        tool, ai = TOOLS[key]
        stages.append({"key": key, "state": "pending", "tool": tool, "ai": ai, "detail": "",
                       "started": None, "ended": None})
    job = Job(uuid.uuid4().hex, stages, [
        {"index": i, "width": 0, "height": 0, "face": None, "kind": None, "outline": None, "circles": [], "reads": []}
        for i in range(n_images)], pipe=pipe)
    if pipe.chat is None:
        job.stage("label").update(tool="Your face tags", ai=False)
    else:
        job.stage("label")["tool"] = model_name()
    if pipe.batch_reader is not None:
        job.stage("read").update(tool=model_name(), ai=True)  # the batch reader uses the VLM_MODEL chat
    elif pipe.reader is not None:
        job.stage("read").update(tool="TrOCR", ai=True)
    else:
        job.stage("read").update(state="skipped", detail="No reader configured: type the sizes")
    if pipe.image_gen is None or not pipe.draw_faces:
        job.stage("draw").update(tool="TripoSR" if pipe.mesh_provider is not None else "assumed",
                                 ai=pipe.mesh_provider is not None)
    if register:
        _register(job)
    return job


def new_sheet_job(pipe: MvPipeline, register: bool = True) -> Job:
    """One image, one job: the sheet replaces label/outline/read with its own three stages, then joins
    draw/fuse exactly as a per-face photo analysis does."""
    stages = [{"key": key, "state": "pending", "tool": tool, "ai": ai, "detail": "", "started": None, "ended": None}
              for key, (tool, ai) in SHEET_TOOLS.items()]
    job = Job(uuid.uuid4().hex, stages,
              [{"index": 0, "width": 0, "height": 0, "face": None, "kind": None, "outline": None, "circles": [],
                "reads": []}], pipe=pipe, mode="sheet")
    if pipe.image_gen is None or not pipe.draw_faces:
        job.stage("draw").update(tool="TripoSR" if pipe.mesh_provider is not None else "assumed",
                                 ai=pipe.mesh_provider is not None)
    if register:
        _register(job)
    return job


def _register(job: Job) -> None:
    with _registry_lock:
        _add(job)


def _add(job: Job) -> None:
    """Into the registry, evicting the least recently used idle jobs past MAX_JOBS. Hold _registry_lock."""
    idle = sorted((j for jid, j in JOBS.items() if jid not in _ACTIVE), key=lambda j: j.use)
    for old in idle[:max(0, len(JOBS) + 1 - MAX_JOBS)]:
        del JOBS[old.job_id]
    JOBS[job.job_id] = job


def get_job(job_id: str) -> Job | None:
    with _registry_lock:
        return JOBS.get(job_id)


def touch(job: Job) -> None:
    job.used, job.use = time.time(), next(_USES)


def sweep_jobs(ttl: float = TTL_S) -> None:
    now = time.time()
    with _registry_lock:
        for jid in [j for j, job in JOBS.items() if now - job.used > ttl]:
            del JOBS[jid]


def _mm(value: float) -> str:
    return f"{round(value, 2):g}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _details(job: Job, key: str) -> str:
    done = [i for i in job.images if i["face"] is not None]
    if key == "label":
        return ", ".join(f"{i['face']} · {i['kind']}" for i in done)
    if key == "outline":
        return _plural(sum(1 for i in job.images if i["outline"]), "closed outline")
    if key == "read":
        values = [r["value_mm"] for i in job.images for r in i["reads"]]
        return " · ".join(_mm(v) for v in values) + " mm" if values else "No numbers found"
    return ""


def _draw_done(job: Job, filled_by: dict) -> None:
    stage = job.stage("draw")
    for face, by in filled_by.items():
        if face in job.coverage and job.coverage[face] != "observed":
            job.coverage[face] = by
    filled = {f: by for f, by in filled_by.items() if by != "observed"}
    if not filled:
        stage.update(state="skipped", detail="Every face was observed")
        return
    for by in ("qwen-image", "triposr", "mirrored", "assumed"):
        if by in filled.values():
            tool, ai = DRAW_TOOLS[by]
            stage.update(tool=tool, ai=ai)
            break
    groups = {}
    for face, by in filled.items():
        groups.setdefault(by, []).append(face)
    stage["detail"] = "; ".join(f"{' and '.join(faces)} {DRAW_WORDS.get(by, by)}" for by, faces in groups.items())


def reduce(job: Job, name: str, data: dict) -> None:
    """Apply one progress event to the job."""
    if name != "stage" or data.get("key") not in ALL_STAGE_KEYS:
        return
    key, state, now = data["key"], data["state"], time.time()
    with job.lock:
        stage = job.stage(key)
        index = data.get("index")
        image = job.images[index] if index is not None and 0 <= index < len(job.images) else None
        if state == "running":
            job.stage_clock[(key, index)] = now
            stage["state"] = "running"
            stage["started"] = stage["started"] or now
            stage["ended"] = None
            return
        if state == "skipped":
            if stage["state"] != "done":
                stage.update(state="skipped", ended=now)
                if key == "read" and not stage["detail"]:
                    stage["detail"] = "No numbers to read on a photo"
            return
        if state != "done":
            stage.update(state=state, ended=now)
            return
        stage.update(state="done", ended=now)
        stage["started"] = stage["started"] or now
        started = job.stage_clock.pop((key, index), stage["started"])  # this image's run, not the first image's
        obs.time_spent("s2c_stage_seconds", now - started, stage=key)
        if key == "label" and image is not None:
            image.update(face=data.get("face"), kind=data.get("kind"), width=data.get("width", 0),
                         height=data.get("height", 0))
            if data.get("face") in job.coverage:
                job.coverage[data["face"]] = "observed"
        elif key == "outline" and image is not None:
            image["outline"] = data.get("outline")
            image["circles"] = [{"cx": float(c["cx"]), "cy": float(c["cy"]), "d": float(c["d"])}
                                for c in data.get("circles", [])]
        elif key == "read" and image is not None:
            image["reads"] = [{**r, "bbox": list(r["bbox"])} for r in data.get("reads", [])]
        elif key in SHEET_LABELS:
            stage["detail"] = data.get("detail") or f"0 {SHEET_LABELS[key]}"
        if key == "draw":
            _draw_done(job, data.get("filled_by", {}))
        elif key not in ("fuse", *SHEET_LABELS):
            stage["detail"] = _details(job, key)


def _numbers(d) -> dict[str, float]:
    if not isinstance(d, dict):
        return {}
    return {str(k): v for k, v in d.items() if isinstance(v, int | float) and not isinstance(v, bool)}


def abstain_json(res: MvAbstain) -> dict:
    """The contract's Abstain: `partial` is a flat {path: number} map (the known values), with the missing paths
    and the suggested values as their own keys, whatever shape MvAbstain.partial has."""
    out = res.model_dump()
    partial = out.get("partial")
    if isinstance(partial, dict) and {"known", "missing", "suggested"} & partial.keys():
        missing, prov = partial.get("missing"), partial.get("provenance")
        known = _numbers(partial.get("known"))
        out.update(partial=known, suggested=_numbers(partial.get("suggested")),
                   missing=[str(m) for m in missing] if isinstance(missing, list) else [],
                   partial_provenance={str(k): v for k, v in prov.items() if k in known and isinstance(v, str)}
                   if isinstance(prov, dict) else {})
    else:
        out.update(partial=None if partial is None else _numbers(partial), missing=[], suggested={},
                   partial_provenance={})
    return out


def _analysis(job: Job, res, filled_by: dict) -> dict:
    spec = None if isinstance(res, MvAbstain) else res.model_dump(mode="json")
    abstain = abstain_json(res) if isinstance(res, MvAbstain) else None
    return {"request_id": job.job_id, "spec": spec, "abstain": abstain, "filled_by": dict(filled_by)}


def _finish(job: Job, res, filled_by: dict) -> None:
    now = time.time()
    with job.lock:
        if isinstance(res, MvAbstain):
            for stage in job.stages:
                if stage["state"] == "running":
                    stage.update(state="failed", ended=now, detail=res.remedy)
                elif stage["state"] == "pending":
                    failed = stage["key"] == "fuse"
                    stage.update(state="failed" if failed else "skipped", detail=res.remedy if failed else "")
        else:
            env = res.envelope
            job.stage("fuse")["detail"] = f"one part, {_mm(env.x_mm)} × {_mm(env.y_mm)} × {_mm(env.z_mm)} mm"
        job.result = _analysis(job, res, filled_by)
        job.status = "done"


def _check(job: Job, data: dict) -> None:
    """Called at every progress event: a cancelled job stops here, and one past its time budget stops before its
    next stage starts (a stage that just finished keeps its result)."""
    if job.cancel:
        raise JobCancelled()
    if data.get("state") == "running" and time.monotonic() - job.clock > JOB_BUDGET_S:
        raise JobTimeout()


def _timed_out(job: Job) -> None:
    log.warning("analysis %s ran past its %d s budget", job.job_id, JOB_BUDGET_S)
    with job.lock:
        job.status, job.error = "failed", TOO_LONG
        for stage in job.stages:
            if stage["state"] == "running":
                stage.update(state="failed", ended=time.time())


def run(job: Job, pipe: MvPipeline, images: list[ImageInput], reference: str | None) -> None:
    def progress(name: str, data: dict) -> None:
        _check(job, data)
        reduce(job, name, data)

    try:
        observed = pipe.observe(images, reference, progress=progress)
        if isinstance(observed, MvAbstain):
            _finish(job, observed, {})
            return
        job.observed = observed
        res = pipe.fuse(observed, progress=progress)
        forget_images(observed, res, pipe)
        _finish(job, res, observed.filled_by)
    except JobCancelled:
        with job.lock:
            job.status = "cancelled"
    except JobTimeout:
        _timed_out(job)
    except Exception:
        log.exception("analysis %s failed", job.job_id)
        with job.lock:
            job.status, job.error = "failed", FAILED
            for stage in job.stages:
                if stage["state"] == "running":
                    stage.update(state="failed", ended=time.time())
    finally:
        with _registry_lock:
            _ACTIVE.discard(job.job_id)


def _sheet_reading(image_bytes: bytes):
    """`read_sketch` (s2c/sketch/pipeline.py, Task 12) may not exist yet in this tree: import it lazily so
    a sheet-mode job fails with a clear message instead of the whole web app failing to start."""
    try:
        from s2c.sketch import read_sketch
    except ImportError as e:
        raise UserFacing(SHEET_UNAVAILABLE) from e
    return read_sketch(image_bytes)


def _drawn_sheet(image_bytes: bytes, pipe: MvPipeline, projection: str = "auto"):
    """The five steps (sheet_read.read_drawing) on a clean drawing or a photo of a hand sketch: page, faces,
    labels, the rest, numbers. A photo it cannot use gives its abstention (the job fails with the remedy); no
    sheet of views gives None, and the team's sketch reader is tried. A crash is not caught here: it fails the job
    rather than quietly handing the user the other reader's result. Views that cannot be named (an isometric
    picture, a detail) are left out with a note: the web app has no per-view face picker."""
    from s2c.multiview.sheet_read import read_drawing
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return read_drawing(image, projection, reader=pipe.reader, service=pipe.reading())


def _values_detail(read) -> str:
    """What the numbers step found: a drawing's scale, or a sketch's overall sizes (a sketch has no scale)."""
    if read.kind == "sketch":
        n, sizes = len(read.scale.dimensions), sum(d.view is not None for d in read.scale.dimensions)
        return (f"{_plural(n, 'value')} read, {_plural(sizes, 'overall size')}" if n
                else "No numbers read: type the sizes")
    used = len(read.scale.used)
    return (f"{used} dimension{'s' if used != 1 else ''} give the scale" if used
            else "No dimensions read: type the sizes")


def run_sheet(job: Job, pipe: MvPipeline, image: ImageInput, projection: str = "auto") -> None:
    """One sheet with several views, in place of one photo per face: read it, then join the same
    draw/fuse stages `run()` uses so Review and Model & Export are unchanged."""
    from s2c.web.sketch_adapter import observed_from_sketch

    def progress(name: str, data: dict) -> None:
        _check(job, data)
        reduce(job, name, data)

    def fused(name: str, data: dict) -> None:
        """observe() reports per-image label/outline/read stages a sheet job does not have; keep draw and fuse."""
        _check(job, data)
        if data.get("key") in ("draw", "fuse"):
            reduce(job, name, data)

    try:
        progress("stage", {"key": "views", "state": "running"})
        read = _drawn_sheet(image.data, pipe, projection)
        page = read if isinstance(read, MvAbstain) else None
        if page is not None:  # the photo is no usable page: say why and what to do
            raise UserFacing(page.remedy, f"{page.stage}: {page.reason}")
        if read is not None:
            from s2c.multiview.sheet_read import observe_drawing
            naming, crops = read.naming, read.crops
            tool = "Sketch reader" if read.kind == "sketch" else "Drawing reader"
            with job.lock:
                for key in ("views", "lines", "values"):
                    job.stage(key).update(tool=tool, ai=False)
                job.images = [{"index": k, "width": 0, "height": 0, "face": c.face,
                               "kind": "drawing", "outline": None, "circles": [], "reads": []}
                              for k, c in enumerate(crops)]
                for c in crops:
                    if c.face in job.coverage:
                        job.coverage[c.face] = "observed"
            angle = "first-angle" if naming.projection == "first" else "third-angle"
            progress("stage", {"key": "views", "state": "done",
                               "detail": f"{len(crops)} views found ({angle}, from the {naming.projection_source})"})
            progress("stage", {"key": "lines", "state": "running"})
            observed = observe_drawing(read, pipe, progress=fused)
            progress("stage", {"key": "lines", "state": "done", "detail": "centre and hidden lines read"})
            progress("stage", {"key": "values", "state": "done" if read.scale.dimensions else "skipped",
                               "detail": _values_detail(read)})
            if isinstance(observed, MvAbstain):
                _finish(job, observed, {})
                return
            job.observed = observed
            res = pipe.fuse(observed, progress=progress)
            forget_images(observed, res, pipe)
            _finish(job, res, observed.filled_by)
            return
        reading = _sheet_reading(image.data)
        if reading.abstain is not None or not reading.views:
            # say why and what to do; falling through would leave Review asking for a width with no views
            a = reading.abstain
            raise UserFacing(a.remedy if a is not None else
                             "No views found on the sheet. Draw the views with a dark pen and retake.",
                             f"{a.stage}: {a.reason}" if a is not None else "no views")
        progress("stage", {"key": "views", "state": "done",
                           "detail": f"{len(reading.views)} views found by the sketch reader (no drawing sheet found)"})
        progress("stage", {"key": "lines", "state": "running"})
        progress("stage", {"key": "lines", "state": "done",
                           "detail": f"{len(reading.entities)} lines classified"})
        progress("stage", {"key": "values", "state": "running"})
        progress("stage", {"key": "values", "state": "done",
                           "detail": f"{len(reading.dimensions)} values read"})
        observed = observed_from_sketch(reading)
        job.observed = observed
        res = pipe.fuse(observed, progress=progress)
        forget_images(observed, res, pipe)
        _finish(job, res, observed.filled_by)
    except JobCancelled:
        with job.lock:
            job.status = "cancelled"
    except JobTimeout:
        _timed_out(job)
    except UserFacing as e:
        log.warning("sheet analysis %s: %s (%s)", job.job_id, e.remedy, e.detail)
        with job.lock:
            job.status, job.error = "failed", e.remedy
            for stage in job.stages:
                if stage["state"] == "running":
                    stage.update(state="failed", ended=time.time())
    except Exception:
        log.exception("sheet analysis %s failed", job.job_id)
        with job.lock:
            job.status, job.error = "failed", FAILED
            for stage in job.stages:
                if stage["state"] == "running":
                    stage.update(state="failed", ended=time.time())
    finally:
        with _registry_lock:
            _ACTIVE.discard(job.job_id)


def _outcome(job: Job) -> str:
    if job.status == "done":
        return "abstain" if job.result and job.result.get("abstain") else "done"
    return "timeout" if job.error == TOO_LONG else job.status


def _scoped(job: Job, target):
    """The analysis run with its id on every log record, and counted when it ends."""
    def run_scoped(*args):
        with obs.job_scope(job.job_id):
            try:
                target(*args)
            finally:
                obs.count("s2c_jobs_total", mode=job.mode, outcome=_outcome(job))
    return run_scoped


def _launch(job: Job, target, args: tuple) -> bool:
    """Run an analysis in the background. False (and the job is dropped) when MAX_RUNNING already run."""
    with _registry_lock:
        if len(_ACTIVE) >= MAX_RUNNING:
            JOBS.pop(job.job_id, None)
            return False
        if job.job_id not in JOBS:
            _add(job)
        _ACTIVE.add(job.job_id)
    threading.Thread(target=_scoped(job, target), args=args, daemon=True, name=f"analysis-{job.job_id[:8]}").start()
    return True


def start(job: Job, pipe: MvPipeline, images: list[ImageInput], reference: str | None) -> bool:
    """Run the analysis in the background. False (and the job is dropped) when MAX_RUNNING already run."""
    return _launch(job, run, (job, pipe, images, reference))


def start_sheet(job: Job, pipe: MvPipeline, image: ImageInput, projection: str = "auto") -> bool:
    return _launch(job, run_sheet, (job, pipe, image, projection))


def merge(job: Job, pipe: MvPipeline, user_values: dict, accepted: list, rejected: list) -> dict:
    """Fuse again with the user's values, on the pipeline the job was configured with (its AI settings)."""
    pipe = job.pipe or pipe
    observed = job.observed
    res = pipe.fuse(observed, user_values, accepted, rejected)
    forget_images(observed, res, pipe)
    return _analysis(job, res, observed.filled_by)
