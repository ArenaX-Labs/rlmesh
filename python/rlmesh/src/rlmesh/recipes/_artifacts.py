"""Runtime artifact mounts: the resolver, the rlmesh cache, ``hf_load``, ``input_path``.

Weights are ALWAYS a runtime mount (FINAL_API_SPEC §4.4) -- never baked into the
image. This module resolves a declared :class:`ArtifactInput` to a concrete local
path, backed by a single content-addressed cache rooted at ``$RLMESH_CACHE_DIR``
(default ``~/.cache/rlmesh/artifacts``, XDG-aware). ``hf_load``'s no-``local_dir``
path resolves through this SAME cache -- never the ambient ``HF_HOME`` -- so a repo
downloaded once serves every run, local and sandboxed alike.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._authoring_model import ModelRecipe
    from ._schema import ArtifactInput

__all__ = ["cache_root", "enter_recipe_context", "hf_load", "input_path", "resolve_inputs"]

# The recipe whose load() is currently running, so the module-level input_path()
# convenience can find its mounts (mirrors self.input_path()).
_CURRENT_RECIPE: contextvars.ContextVar[ModelRecipe | None] = contextvars.ContextVar(
    "rlmesh_current_recipe", default=None
)

_HF_SCHEME = "hf://"


def cache_root() -> Path:
    """The rlmesh artifact-cache root (``$RLMESH_CACHE_DIR`` or an XDG default)."""
    explicit = os.environ.get("RLMESH_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "rlmesh" / "artifacts"


@contextlib.contextmanager
def enter_recipe_context(instance: ModelRecipe) -> Iterator[None]:
    """Bind ``instance`` as the current recipe for the module-level ``input_path``."""
    token = _CURRENT_RECIPE.set(instance)
    try:
        yield
    finally:
        _CURRENT_RECIPE.reset(token)


def input_path(name: str) -> str:
    """Resolve a declared ``ArtifactInput`` name to its local path (module-level form).

    Convenience for use inside ``load()`` -- equivalent to ``self.input_path(name)``.
    Only valid while a ``ModelRecipe.load()`` is running (it reads the recipe bound
    by :func:`enter_recipe_context`).
    """
    recipe = _CURRENT_RECIPE.get()
    if recipe is None:
        raise RuntimeError(
            "rlmesh.recipes.input_path() is only valid inside ModelRecipe.load(); "
            "use self.input_path(name) outside that context"
        )
    return recipe.input_path(name)


def resolve_inputs(
    inputs: Sequence[ArtifactInput],
    *,
    in_container: bool,
    overrides: Sequence[ArtifactInput] = (),
) -> dict[str, str]:
    """Resolve every declared mount to a local path, keyed by name.

    In-container: each mount is already materialized at ``target_path`` by the
    bootstrap, so the path is ``target_path``. Locally: ``local_dir`` wins, else
    the ``uri`` is resolved through the rlmesh cache. A required mount that cannot
    be resolved raises; an optional one is skipped.

    ``overrides`` are per-run mounts (``ModelServer``/``SandboxModel`` ``artifacts=``):
    one matching a declared ``name`` replaces it (checkpoint selection -- the launch
    arg wins, FINAL_API_SPEC §4.4); a new name adds a mount.
    """
    override_by_name = {a.name: a for a in overrides}
    resolved: dict[str, str] = {}
    for declared in inputs:
        artifact = override_by_name.get(declared.name, declared)
        path = _resolve_one(artifact, in_container=in_container)
        if path is not None:
            resolved[artifact.name] = path
    declared_names = {a.name for a in inputs}
    for name, artifact in override_by_name.items():
        if name not in declared_names:
            path = _resolve_one(artifact, in_container=in_container)
            if path is not None:
                resolved[name] = path
    return resolved


def _resolve_one(artifact: ArtifactInput, *, in_container: bool) -> str | None:
    if in_container:
        return artifact.target_path
    if artifact.local_dir is not None:
        return artifact.local_dir
    if artifact.uri is not None:
        return _resolve_uri(artifact.uri, include=artifact.include)
    if artifact.required:
        raise FileNotFoundError(
            f"ArtifactInput {artifact.name!r} has no uri/local_dir and is required; "
            "bind it at run time (SandboxModel(artifacts=...)/ModelServer(artifacts=...)) "
            "or set local_dir="
        )
    return None


def _resolve_uri(uri: str, *, include: Sequence[str] = ()) -> str:
    """Resolve a mount uri to a local directory in the rlmesh cache."""
    if uri.startswith("file://"):
        return uri[len("file://") :]
    if uri.startswith(_HF_SCHEME):
        repo, _, revision = uri[len(_HF_SCHEME) :].partition("@")
        return _snapshot_hf(repo, revision or None, allow_patterns=list(include) or None)
    raise NotImplementedError(
        f"artifact uri scheme not resolvable on the local path yet: {uri!r}; "
        "use local_dir= for gs://, s3://, https:// until the remote resolver lands"
    )


def _snapshot_hf(
    repo: str, revision: str | None, *, allow_patterns: list[str] | None = None
) -> str:
    """Download an HF repo snapshot into the rlmesh cache and return the local dir."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "resolving an hf:// artifact requires huggingface_hub "
            "(pip install huggingface_hub)"
        ) from exc

    root = cache_root() / "hf"
    root.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo,
        revision=revision,
        cache_dir=str(root),
        allow_patterns=allow_patterns,
    )


def hf_load(
    repo: str,
    *,
    revision: str | None = None,
    loader: str = "transformers:AutoModel",
    trust_remote_code: bool = False,
    processor: str | None = None,
    local_dir: str | None = None,
    **kwargs: Any,
) -> Any:
    """Batteries-included HuggingFace load -- the one-liner you call inside ``load()``.

    NOT a factory arm: just a helper (FINAL_API_SPEC §3.4). ``loader`` is an explicit
    ``module:Class`` string (declare-do-not-detect), e.g. ``"transformers:AutoModel"``
    or ``"lerobot:SmolVLAPolicy"``. When ``local_dir`` is set (the ArtifactInput-mount
    case) weights load from that dir and ``repo`` is only a loader hint; otherwise
    ``repo@revision`` is resolved through the rlmesh artifact cache (not ``HF_HOME``).

    Returns the model, or ``(model, processor)`` when ``processor`` is requested.
    """
    source = local_dir if local_dir is not None else _snapshot_hf(repo, revision)
    model = _from_pretrained(loader, source, trust_remote_code=trust_remote_code, **kwargs)
    if processor is not None:
        proc = _from_pretrained(processor, source, trust_remote_code=trust_remote_code)
        return model, proc
    return model


def _from_pretrained(loader: str, source: str, **kwargs: Any) -> Any:
    module_name, sep, attr = loader.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(
            f"hf_load loader must be 'module:Class', got {loader!r} "
            "(e.g. 'transformers:AutoModelForVision2Seq' or 'lerobot:SmolVLAPolicy')"
        )
    import importlib

    module = importlib.import_module(module_name)
    obj = module
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise AttributeError(
                f"hf_load loader {loader!r}: {type(obj).__name__} has no attribute {part!r}"
            ) from exc
    from_pretrained = getattr(obj, "from_pretrained", None)
    if from_pretrained is None:
        raise TypeError(f"{loader} has no from_pretrained(); not a HF loader")
    return from_pretrained(source, **kwargs)
