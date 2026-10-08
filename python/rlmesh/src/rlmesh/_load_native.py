"""Single point for importing symbols from the compiled ``rlmesh._rlmesh`` core.

Centralizes the ``try/except`` that turns a missing/unbuilt native extension into
one uniform, actionable :class:`ImportError`, so the runtime call sites that pull
native client/server/model classes do not each repeat the guard.

It is also where an editable install notices a stale extension: the in-tree
``.so`` is rebuilt only by ``mise run build:python:develop`` (or ``uv run``), so
after a commit or checkout it can keep running a build stamped from an older
HEAD, whose dev workflow edition no freshly built peer shares.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path
from typing import Any

STALE_CHECK_ENV_VAR = "RLMESH_STALE_CHECK"
"""Set to ``0`` (or ``false``/``no``/``off``) to skip the stale-extension check."""

_OPT_OUT = frozenset({"0", "false", "no", "off"})

_stale_checked = False


def load_native(name: str) -> Any:
    """Return symbol ``name`` from the native ``rlmesh._rlmesh`` module.

    Raises a uniform :class:`ImportError` when the compiled extension is not
    importable (e.g. the wheel was installed without the built core).
    """
    try:
        import rlmesh._rlmesh as native
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError("Failed to import _rlmesh native module.") from e
    _warn_if_stale(native)
    return getattr(native, name)


def _warn_if_stale(native: Any) -> None:
    """Warn once per process when an editable build lags the checkout's HEAD.

    Runs only for a source build (``build_info().build_source == "git"``) loaded
    from a repository checkout, never for an installed wheel. Never raises.
    """
    global _stale_checked
    if _stale_checked:
        return
    _stale_checked = True
    try:
        message = _stale_message(native)
    except Exception:
        return
    if message is not None:
        warnings.warn(message, UserWarning, stacklevel=3)


def _stale_message(native: Any) -> str | None:
    """The stale-extension warning for ``native``, or ``None`` when it is current."""
    if os.environ.get(STALE_CHECK_ENV_VAR, "").strip().lower() in _OPT_OUT:
        return None
    info = native.build_info()
    built = info.git
    if info.build_source != "git" or not built:
        return None
    checkout = _checkout_root(Path(native.__file__))
    if checkout is None:
        return None
    head = _git_head(checkout)
    built_head = built.split(".", 1)[0]
    if head is None or head == built_head:
        return None
    return (
        f"the rlmesh native extension is stale: it was built from {built_head} but the "
        f"checkout at {checkout} is at {head}, so its dev workflow edition matches no peer "
        f"built from the current tree. Run `mise run build:python:develop` to rebuild it "
        f"(set {STALE_CHECK_ENV_VAR}=0 to silence this check)."
    )


def _checkout_root(extension: Path) -> Path | None:
    """The rlmesh repository the extension was loaded from, or ``None`` for an install."""
    resolved = extension.resolve()
    if any(part in {"site-packages", "dist-packages"} for part in resolved.parts):
        return None
    for parent in resolved.parents:
        if (parent / "rlmesh.toml").is_file() and (parent / ".git").exists():
            return parent
    return None


def _git_head(checkout: Path) -> str | None:
    """``git rev-parse --short=12 HEAD`` in ``checkout``, or ``None`` when git fails."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=checkout,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = result.stdout.strip()
    return head if result.returncode == 0 and head else None
