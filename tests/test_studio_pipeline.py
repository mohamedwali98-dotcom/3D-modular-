import numpy as np
import pytest

from s2c.multiview import pipeline
from s2c.multiview.fuse import snap_diameter
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.settings import AiSettings, GeometrySettings
from tests.test_mv_pipeline import fake_reads, sketch
from tests.test_mv_qwen_faces import fake_gen


def test_configured_switches_are_per_request():
    def batch(crops):
        return []

    def provider(img):
        raise RuntimeError

    def depth(img):
        raise RuntimeError

    gen = fake_gen(np.zeros((10, 10, 3), np.uint8))
    base = MvPipeline(batch_reader=batch, mesh_provider=provider, image_gen=gen, depth=depth)
    pipe = base.configured(AiSettings(use_reader=False, use_qwen_image=False, use_rescue=False, use_triposr=False,
                                      use_solaria=False, seed=42, attempts=1))
    assert (pipe.batch_reader, pipe.mesh_provider, pipe.depth) == (None, None, None)
    assert (pipe.draw_faces, pipe.rescue_enabled, pipe.seed, pipe.attempts) == (False, False, 42, 1)
    assert base.batch_reader is batch and base.draw_faces and base.rescue_enabled and base.seed == 7


def test_the_seed_and_attempts_reach_qwen_image():
    gen = fake_gen(np.zeros((300, 300, 3), np.uint8))  # an empty drawing: rejected, so every attempt is used
    pipe = MvPipeline(image_gen=gen).configured(AiSettings(use_qwen_image=True, seed=42, attempts=2))
    observed = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch")])
    pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 10})
    assert [c[2] for c in gen.calls] == [42, 43, 42, 43]


def test_switching_off_qwen_image_leaves_faces_to_the_fallback():
    gen = fake_gen(np.zeros((300, 300, 3), np.uint8))
    pipe = MvPipeline(image_gen=gen).configured(AiSettings(use_qwen_image=False))
    observed = pipe.observe([ImageInput(sketch(600, 400), "front", "sketch")])
    pipe.fuse(observed, {"envelope.x_mm": 60, "envelope.y_mm": 40, "envelope.z_mm": 10})
    assert gen.calls == [] and observed.filled_by["top"] == "assumed"


def test_the_clearance_class_picks_the_table():
    assert (snap_diameter(5.37), snap_diameter(5.37, "fine"), snap_diameter(5.7, "coarse")) == (5.5, 5.3, 5.8)
    assert snap_diameter(7.2) == 7.0  # no clearance hole within 0.4 mm: the 0.5 mm grid


def test_snapping_follows_the_geometry_settings(monkeypatch):
    monkeypatch.setattr(pipeline, "read_values", fake_reads([[(60, "below"), (40, "left")], [(60, "below")]] * 3))
    pipe = MvPipeline(reader=lambda crop: ("", 0.0))
    observed = pipe.observe([ImageInput(sketch(600, 400, circles=[(100, 300, 30)]), "front", "sketch"),
                             ImageInput(sketch(600, 100), "top", "sketch")])
    values = {"envelope.z_mm": 10.0}
    raw = pipe.fuse(observed, values, geometry=GeometrySettings(snap=False)).features[0].diameter_mm
    coarse = pipe.fuse(observed, values, geometry=GeometrySettings(clearance="coarse")).features[0].diameter_mm
    assert raw == pytest.approx(5.4, abs=0.25) and raw not in (5.5, 5.8) and coarse == 5.8
