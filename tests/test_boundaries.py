"""No module imports another module's private name (audit M7): a helper used elsewhere is public."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _private_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [f"{node.module}.{a.name}" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            and node.module and node.module.split(".")[0] in ("s2c", "tests")
            for a in node.names if a.name.startswith("_") and not a.name.startswith("__")]


@pytest.mark.parametrize("folder", ["s2c", "tests", "scripts"])
def test_no_module_imports_a_private_name(folder):
    found = {str(p.relative_to(ROOT)): names for p in sorted((ROOT / folder).rglob("*.py"))
             if (names := _private_imports(p))}
    assert found == {}
