"""Small kept runtime-spec core: the ``DELEGATED`` sentinel and ``ArtifactInput``.

These two are the only pieces of the former recipe schema the kept features still
depend on: ``DELEGATED`` is the self-adapting-model sentinel on the eval/serve
paths, and ``ArtifactInput`` is the runtime weight/asset mount type used by
``Model``/``SandboxModel`` and the artifact resolver. They carry no recipe-document
machinery.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final


class _Delegated:
    """Sentinel: the model self-adapts, so resolve no adapter (vs ``spec=None``)."""

    def __repr__(self) -> str:
        return "DELEGATED"


DELEGATED = _Delegated()
"""Pass as a model ``spec`` so the model self-adapts: resolve no adapter (unlike
``spec=None``). One module-level instance -- ``spec is DELEGATED`` identity is
load-bearing on the eval/serve paths."""


class RecipeValidationError(ValueError):
    """Raised when an ``ArtifactInput`` is constructed with an unsafe/invalid field.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers keep working.
    """


# The name half of "namespace/name".
_RECIPE_NAME: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]*$")
_POSIX_PATH: Final = re.compile(r"^[A-Za-z0-9._/\-]+$")
# A relative path glob: a POSIX path plus the only wildcards supported -- '*'/'**'.
_INCLUDE_GLOB: Final = re.compile(r"^[A-Za-z0-9._/\-*]+$")

_ARTIFACT_URI_SCHEMES: Final = (
    "hf://",
    "gs://",
    "s3://",
    "https://",
    "http://",
    "file://",
)


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise RecipeValidationError(
            f"{field_name} must be a str, got {type(value).__name__}"
        )
    return value


def _check_token(value: str, pattern: re.Pattern[str], field_name: str) -> None:
    # fullmatch, not match: a `$`-anchored pattern's match() accepts a trailing
    # newline; fullmatch rejects any control char / trailing \n.
    if not pattern.fullmatch(value):
        raise RecipeValidationError(
            f"{field_name} {value!r} is not a valid {field_name} token"
        )


def _as_str_tuple(value: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise RecipeValidationError(
            f"{field_name} expects a sequence of strings, not a bare str; "
            f"pass [{value!r}] for a single entry"
        )
    return tuple(_require_str(item, f"{field_name}[]") for item in value)


def _check_include_glob(value: str, field_name: str) -> None:
    if not value:
        raise RecipeValidationError(f"{field_name} entry must be non-empty")
    if value[0] == "/":
        raise RecipeValidationError(
            f"{field_name} {value!r} must be a relative glob, not an absolute path"
        )
    _check_token(value, _INCLUDE_GLOB, field_name)


@dataclass(frozen=True)
class ArtifactInput:
    """A runtime weight/asset mount for a model.

    Weights are ALWAYS a runtime mount -- never baked into the image. A ``uri`` is
    fetched by the rlmesh artifact resolver into a content-addressed cache (root
    ``$RLMESH_CACHE_DIR``, default ``~/.cache/rlmesh/artifacts``); ``local_dir``
    overrides with an explicit host dir for the local (non-sandbox) path. ``None``
    uri means the mount is bound at run time (a ``SandboxModel``/``Model`` launch arg).
    """

    name: str
    target_path: str
    uri: str | None = None
    local_dir: str | None = None
    include: Sequence[str] = ()
    required: bool = True

    def __post_init__(self) -> None:
        _check_token(
            _require_str(self.name, "ArtifactInput.name"),
            _RECIPE_NAME,
            "ArtifactInput.name",
        )
        _check_token(
            _require_str(self.target_path, "ArtifactInput.target_path"),
            _POSIX_PATH,
            "ArtifactInput.target_path",
        )
        if self.uri is not None:
            uri = _require_str(self.uri, "ArtifactInput.uri")
            if not uri.startswith(_ARTIFACT_URI_SCHEMES):
                raise RecipeValidationError(
                    f"ArtifactInput.uri {uri!r} must use one of {_ARTIFACT_URI_SCHEMES}"
                )
        if self.local_dir is not None:
            _require_str(self.local_dir, "ArtifactInput.local_dir")
        includes = _as_str_tuple(self.include, "ArtifactInput.include")
        for glob in includes:
            _check_include_glob(glob, "ArtifactInput.include")
        object.__setattr__(self, "include", includes)


__all__ = ["DELEGATED", "ArtifactInput", "RecipeValidationError"]
