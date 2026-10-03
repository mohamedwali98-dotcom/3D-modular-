"""The web app's copies of the shared models are generated from the Pydantic models (audit M11)."""
from pathlib import Path

from scripts.gen_ts_models import render

GEN = Path(__file__).resolve().parents[1] / "web" / "src" / "api" / "models.gen.ts"


def test_the_generated_models_are_current():
    assert GEN.read_text(encoding="utf-8") == render(), "run: uv run python scripts/gen_ts_models.py"
