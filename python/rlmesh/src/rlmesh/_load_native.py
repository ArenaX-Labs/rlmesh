"""Single point for importing symbols from the compiled ``rlmesh._rlmesh`` core.

Centralizes the ``try/except`` that turns a missing/unbuilt native extension into
one uniform, actionable :class:`ImportError`, so the runtime call sites that pull
native client/server/model classes do not each repeat the guard.
"""

from __future__ import annotations

from typing import Any


def load_native(name: str) -> Any:
    """Return symbol ``name`` from the native ``rlmesh._rlmesh`` module.

    Raises a uniform :class:`ImportError` when the compiled extension is not
    importable (e.g. the wheel was installed without the built core).
    """
    try:
        import rlmesh._rlmesh as native
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError("Failed to import _rlmesh native module.") from e
    return getattr(native, name)
