"""Projection helpers shared by ``EnvRecipe`` and ``ModelRecipe`` authoring.

Both base classes project a class to an inert ``Recipe`` the same way: the
``name`` must be declared on the concrete class and the class must be importable
by ``module:Class``, and the no-argument lifecycle instantiation must fail with a
recipe-aware error. Keeping that logic here stops the two siblings from drifting.
"""

from __future__ import annotations

import inspect
from typing import TypeVar

from .._schema import RecipeValidationError

_T = TypeVar("_T")


def require_importable_name(cls: type, *, kind: str) -> str:
    """The class's own ``name``, or raise if missing or not container-importable.

    ``name`` must be declared on the concrete class (not inherited, else a no-name
    subclass would silently project under its parent's identity), and the class
    must live in an importable ``module:Class`` path (not ``__main__`` or a local
    scope), because the container imports it by that path. ``kind`` labels the
    error (``"EnvRecipe"`` / ``"ModelRecipe"``).
    """
    name = cls.__dict__.get("name")
    if not isinstance(name, str) or not name:
        inherited = getattr(cls, "name", None)
        hint = (
            f" (it would otherwise inherit {inherited!r})"
            if isinstance(inherited, str) and inherited
            else ""
        )
        raise RecipeValidationError(
            f'{cls.__qualname__} must declare its own `name = "namespace/name"`{hint}'
        )
    if cls.__module__ == "__main__" or "<locals>" in cls.__qualname__:
        raise RecipeValidationError(
            f"{kind} {name!r} is defined in {cls.__module__}:{cls.__qualname__}, "
            "which the container cannot import; define it in an installed module"
        )
    return name


def instantiate(cls: type[_T], *, param_hint: str) -> _T:
    """Construct ``cls()`` with a recipe-aware error if ``__init__`` requires args.

    The lifecycle always instantiates with no arguments, so a required-arg
    ``__init__`` would otherwise fail with a confusing native ``TypeError``;
    ``param_hint`` names where per-construction parameters belong instead.
    """
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls()  # un-introspectable: let cls() raise on its own terms
    required = [
        name
        for name, p in signature.parameters.items()
        if p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if required:
        raise TypeError(
            f"{cls.__qualname__} is instantiated with no arguments, but its "
            f"__init__ requires {required}. {param_hint}"
        )
    return cls()


__all__ = ["instantiate", "require_importable_name"]
