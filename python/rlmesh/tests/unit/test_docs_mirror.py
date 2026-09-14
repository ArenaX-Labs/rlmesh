"""The docs build's native stub must mirror the exported wire constants.

``docs/conf.py`` fakes ``rlmesh._rlmesh`` so ``rlmesh.adapters`` imports without
the compiled extension. That fake carries a hand-written dict of the wire
vocabulary, so every constant added to the crate has to be added there too or
the documented vocabulary silently drifts from the shipped one. Both files are
read with ``ast`` (no import, no extension needed).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
STUB = REPO_ROOT / "python" / "rlmesh" / "src" / "rlmesh" / "_rlmesh.pyi"
CONF = REPO_ROOT / "docs" / "conf.py"


def _exported_str_constants() -> set[str]:
    """Module-level ``NAME: builtins.str`` annotations in the generated stub.

    Dunders (``__version__``, ``__build__``) are package identity, not wire
    vocabulary, and the docs stub sets them elsewhere.
    """
    module = ast.parse(STUB.read_text(encoding="utf-8"))
    return {
        node.target.id
        for node in module.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and ast.unparse(node.annotation) == "builtins.str"
        and not node.target.id.startswith("_")
    }


def _documented_constants() -> set[str]:
    """Keys of the ``adapter_constants`` dict literal inside ``docs/conf.py``."""
    module = ast.parse(CONF.read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "adapter_constants"
            and isinstance(node.value, ast.Dict)
        ):
            return {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise AssertionError(f"no adapter_constants dict literal in {CONF}")


def test_docs_stub_mirrors_every_exported_wire_constant() -> None:
    assert _documented_constants() == _exported_str_constants()
