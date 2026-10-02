"""The user's edits to a fused spec, keyed by path as the Review screens send them (audit M6): the envelope's sizes,
a feature's numeric values, and "features[k].keep" = 0 to take a misread feature out. One parser says which paths
exist, so nothing else is written into a feature or silently dropped."""
from __future__ import annotations

import re

from s2c.multiview.spec import FaceBoss, FaceHole, FacePocket, FaceSlot

ENVELOPE = re.compile(r"envelope\.[xyz]_mm")
FEATURE = re.compile(r"features\[(\d+)\]\.(\w+)")
FIELDS = frozenset({name for model in (FaceHole, FaceSlot, FacePocket, FaceBoss) for name in model.model_fields
                    if name not in ("type", "face")} | {"keep"})


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
