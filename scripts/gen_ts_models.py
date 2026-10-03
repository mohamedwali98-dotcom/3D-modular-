"""Writes web/src/api/models.gen.ts from the Pydantic models the web app shares with the server (audit M11): the
spec, the abstention and the settings. Run it after changing s2c/multiview/spec.py or settings.py:
    uv run python scripts/gen_ts_models.py
tests/test_ts_models.py fails while the file is stale. Every property is present (the server always sends it)."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic.json_schema import models_json_schema

from s2c.multiview.settings import AiSettings, ExportSettings, GeometrySettings, MeshSettings, PrintSettings
from s2c.multiview.spec import MultiViewSpec, MvAbstain

OUT = Path(__file__).resolve().parents[1] / "web" / "src" / "api" / "models.gen.ts"
MODELS = (MultiViewSpec, MvAbstain, AiSettings, GeometrySettings, MeshSettings, PrintSettings, ExportSettings)
HEADER = ("// Generated from the Pydantic models by scripts/gen_ts_models.py. Do not edit; after changing\n"
          "// s2c/multiview/spec.py or settings.py run: uv run python scripts/gen_ts_models.py\n")
SCALARS = {"string": "string", "number": "number", "integer": "number", "boolean": "boolean", "null": "null"}


def _literal(value) -> str:
    return "'" + str(value).replace("'", "\\'") + "'" if isinstance(value, str) else json.dumps(value)


def _union(parts: list[str]) -> str:
    return " | ".join(dict.fromkeys(parts))


def ts_type(schema: dict) -> str:
    """The TypeScript type of one JSON schema node."""
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if "const" in schema:
        return _literal(schema["const"])
    if "enum" in schema:
        return _union([_literal(v) for v in schema["enum"]])
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return _union([ts_type(s) for s in schema[key]])
    kind = schema.get("type")
    if kind == "array":
        if "prefixItems" in schema:
            return "[" + ", ".join(ts_type(s) for s in schema["prefixItems"]) + "]"
        item = ts_type(schema.get("items", {}))
        return f"({item})[]" if "|" in item else f"{item}[]"
    if kind == "object":
        extra = schema.get("additionalProperties")
        return f"Record<string, {ts_type(extra)}>" if isinstance(extra, dict) else "Record<string, unknown>"
    return SCALARS.get(kind, "unknown")


def _interface(name: str, schema: dict) -> str:
    doc = " ".join(schema.get("description", "").split()).split(". ")[0].rstrip(".")
    lines = [f"/** {doc}. */"] if doc else []
    lines.append(f"export interface {name} {{")
    lines += [f"  {prop}: {ts_type(node)};" for prop, node in schema.get("properties", {}).items()]
    return "\n".join([*lines, "}"])


def render() -> str:
    _, schema = models_json_schema([(m, "serialization") for m in MODELS])
    defs = schema["$defs"]
    return HEADER + "".join("\n" + _interface(name, defs[name]) + "\n" for name in sorted(defs))


def main() -> None:
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
