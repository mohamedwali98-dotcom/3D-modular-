"""The user's edits to a fused spec, keyed by path as the Review screens send them (audit M6): the envelope's sizes,
a feature's numeric values, and "features[k].keep" = 0 to take a misread feature out. One parser says which paths
exist, so nothing else is written into a feature or silently dropped."""
from __future__ import annotations

import re

from pydantic import ValidationError

from s2c.multiview.spec import FaceBoss, FaceHole, FacePocket, FaceSlot

ENVELOPE = re.compile(r"envelope\.[xyz]_mm")
FEATURE = re.compile(r"features\[(\d+)\]\.(\w+)")
FIELDS = frozenset({name for model in (FaceHole, FaceSlot, FacePocket, FaceBoss) for name in model.model_fields
                    if name not in ("type", "face")} | {"keep"})


FIELD_WORDS = {"a_mm": "position", "b_mm": "position", "diameter_mm": "diameter", "depth_mm": "depth",
               "width_mm": "width", "length_mm": "length", "angle_deg": "angle", "height_mm": "height"}


def left_out(name: str, feature_exists: bool) -> str:
    """What the Review screen says about an edit the part cannot hold, in its own words (no internal path)."""
    if name == "keep":
        return "A feature you took out is no longer on this part."
    what = FIELD_WORDS.get(name, "value")
    why = "that feature has none." if feature_exists else "that feature is no longer on this part."
    return f"{'An' if what[0] in 'aeiou' else 'A'} {what} you typed was left out: {why}"


MODELS = {"hole": FaceHole, "slot": FaceSlot, "pocket": FacePocket, "boss": FaceBoss}


def out_of_range(feature: dict, name: str, value: float) -> str | None:
    """The sentence for a typed value this feature cannot hold (0, or past 10 000 mm), or None: the value is left
    out and said, so one bad number never turns the whole review into an abstain."""
    try:
        MODELS[feature["type"]].model_validate({**feature, name: value})
    except ValidationError:
        what = FIELD_WORDS.get(name, "value")
        unit = " mm" if name.endswith("_mm") else ""
        return f"{'An' if what[0] in 'aeiou' else 'A'} {what} you typed ({value:g}{unit}) was left out: it is out of range."
    return None


class EditError(ValueError):
    def __init__(self, path: str):
        super().__init__(f"{path} cannot be edited")
        self.path = path


def editable(path: str) -> bool:
    if ENVELOPE.fullmatch(path):
        return True
    m = FEATURE.fullmatch(path)
    return m is not None and m.group(2) in FIELDS


def parse(user_values: dict[str, float]) -> dict[str, float]:
    """The edits, every path checked; EditError names the first one that is not an editable value."""
    for path in user_values:
        if not editable(path):
            raise EditError(path)
    return dict(user_values)
