"""Runtime artifact mounts: the resolver, the rlmesh cache, ``hf_load``, ``input_path``.

A model's weights are a runtime mount, never baked into its image. This module
resolves a declared :class:`ArtifactInput` to a local path, backed by one
content-addressed cache under ``$RLMESH_CACHE_DIR`` (default
``~/.cache/rlmesh/artifacts``, XDG-aware). ``hf_load`` without ``local_dir``
resolves through that same cache rather than the ambient ``HF_HOME``, so a repo
downloaded once serves every run, local or sandboxed.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ._schema import RecipeValidationError

if TYPE_CHECKING:
    from ._schema import ArtifactInput

__all__ = [
    "cache_root",
    "enter_recipe_context",
    "hf_load",
    "input_path",
    "local_dir_mounts",
]

# The recipe whose load()/make() is running, so the module-level input_path() can
# find its mounts without an explicit self.
_CURRENT_RECIPE: contextvars.ContextVar[ArtifactConsumer | None] = (
    contextvars.ContextVar("rlmesh_current_recipe", default=None)
)

_HF_SCHEME = "hf://"


class ArtifactConsumer:
    """A recipe that declares runtime ``ArtifactInput`` mounts and resolves them.

    Shared by :class:`ModelRecipe` (weights) and :class:`EnvRecipe` (assets) so the
    materialize() seam is identical for both kinds. Construction binds the two
    private slots; ``input_path`` resolves a declared mount lazily inside
    ``load()``/``make()``.
    """

    #: Runtime weight/asset mounts. Never baked into the image.
    inputs: ClassVar[tuple[ArtifactInput, ...]] = ()

    def __init__(self) -> None:
        # Bound per-instance by the construct functions before load()/make() runs.
        self._rlmesh_inputs: dict[str, ArtifactInput] = {}
        self._rlmesh_in_container: bool = False

    def input_path(self, name: str) -> str:
        """Resolve a declared :class:`ArtifactInput` mount by name to its local path.

        Call it inside ``load()``/``make()``. Resolution order: a path the run
        contract already materialized (env override), else the in-container mount or
        cache fetch, else the resolved host ``local_dir`` or cache path locally.
        """
        declared: dict[str, ArtifactInput] = getattr(self, "_rlmesh_inputs", {})
        try:
            artifact = declared[name]
        except KeyError:
            names = ", ".join(declared) or "<none>"
            raise RecipeValidationError(
                f"{type(self).__name__}.input_path({name!r}): no such ArtifactInput; "
                f"declared inputs: {names}"
            ) from None
        path = resolve_artifact(
            artifact, in_container=getattr(self, "_rlmesh_in_container", False)
        )
        if path is None:
            raise FileNotFoundError(
                f"ArtifactInput {name!r} is optional and unresolved; nothing to load"
            )
        return path


def _env_key(name: str) -> str:
    """An ArtifactInput name as the env-var token form: upper, non-alnum -> ``_``."""
    return "".join(c if c.isalnum() else "_" for c in name).upper()


def _materialized_path_from_env(name: str) -> str | None:
    """An already-materialized path for input ``name`` from the run contract, if set.

    Managed's sidecar materializes inputs out-of-container and exports the path here,
    so an OSS image run under managed reads those bytes instead of re-fetching. The
    ``RLMESH_MODEL_*`` forms are back-compat aliases for what managed emits today.
    """
    key = _env_key(name)
    candidates = [f"RLMESH_INPUT_{key}_PATH", f"RLMESH_MODEL_INPUT_{key}_PATH"]
    if name == "checkpoint":
        candidates.append("RLMESH_MODEL_CHECKPOINT_PATH")
    for env_name in candidates:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def cache_root() -> Path:
    """The rlmesh artifact-cache root: ``$RLMESH_CACHE_DIR`` or an XDG default."""
    explicit = os.environ.get("RLMESH_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "rlmesh" / "artifacts"


@contextlib.contextmanager
def enter_recipe_context(instance: ArtifactConsumer) -> Iterator[None]:
    """Bind ``instance`` as the current recipe for the module-level ``input_path``."""
    token = _CURRENT_RECIPE.set(instance)
    try:
        yield
    finally:
        _CURRENT_RECIPE.reset(token)


def input_path(name: str) -> str:
    """Resolve a declared ``ArtifactInput`` name to its local path.

    The module-level form of ``self.input_path(name)``, valid only while a
    ``ModelRecipe.load()`` is running.
    """
    recipe = _CURRENT_RECIPE.get()
    if recipe is None:
        raise RuntimeError(
            "rlmesh.recipes.input_path() is only valid inside ModelRecipe.load(); "
            "use self.input_path(name) outside that context"
        )
    return recipe.input_path(name)


def merged_inputs(
    inputs: Sequence[ArtifactInput], overrides: Sequence[ArtifactInput]
) -> dict[str, ArtifactInput]:
    """Overlay per-run overrides onto declared mounts, keyed by name.

    An override matching a declared name replaces it (the run-time checkpoint
    selection wins); a new name adds a mount.
    """
    by_name = {a.name: a for a in inputs}
    for override in overrides:
        by_name[override.name] = override
    return by_name


def local_dir_mounts(
    inputs: Sequence[ArtifactInput],
    overrides: Sequence[ArtifactInput] = (),
) -> list[tuple[str, str]]:
    """Resolved ``(host_dir, container_target)`` bind-mounts for declared inputs.

    Only an input with a host ``local_dir`` is mounted; a uri-backed input
    resolves in-container instead (``hf_load`` through the rlmesh cache).
    ``overrides`` supply or replace a declared input's ``local_dir`` by name (the
    run-time checkpoint selection), mounted at the declared target the container
    resolves. A declared ``local_dir`` that is not a directory fails loud here,
    on the host, before the container starts. The mount is read-only; the host
    directory must be readable by the sandbox's non-root container user.
    """
    declared_names = {declared.name for declared in inputs}
    override_dir: dict[str, str] = {}
    for override in overrides:
        # An override for a name the recipe never declared would silently do
        # nothing (the container's recipe has no such input to mount onto), so
        # reject it rather than ignore it.
        if override.name not in declared_names:
            raise ValueError(
                f"artifacts override {override.name!r} matches no declared input "
                f"({sorted(declared_names) or ['<none>']}); it would not be mounted"
            )
        if override.local_dir is not None:
            override_dir[override.name] = override.local_dir
    mounts: list[tuple[str, str]] = []
    for declared in inputs:
        local_dir = override_dir.get(declared.name, declared.local_dir)
        if local_dir is None:
            continue
        target = declared.target_path
        if not target.startswith("/") or ".." in Path(target).parts:
            raise ValueError(
                f"ArtifactInput {declared.name!r} target_path must be an absolute "
                f"container path without '..': {target!r}"
            )
        host = Path(local_dir).expanduser()
        if not host.is_dir():
            raise FileNotFoundError(
                f"ArtifactInput {declared.name!r} local_dir is not a directory: {host} "
                "(it is bind-mounted into the sandbox at run time)"
            )
        mounts.append((str(host.resolve()), target))
    return mounts


def resolve_artifact(artifact: ArtifactInput, *, in_container: bool) -> str | None:
    """Resolve one declared input to a local path (the ``materialize()`` seam).

    An env override (a path the run contract already materialized, e.g. managed's
    sidecar) wins everywhere. In a container a bind-mounted ``local_dir`` input is
    already at ``target_path``; a uri-only input is fetched through the rlmesh cache
    here. On the host, ``local_dir`` wins, else the ``uri`` resolves through the
    cache. Resolution is lazy (called from ``input_path``), so an unused input is
    never fetched; a required, unresolved input raises and an optional one returns
    None.
    """
    override = _materialized_path_from_env(artifact.name)
    if override is not None:
        return override
    if in_container:
        # A local_dir input is bind-mounted at target_path by the sandbox; a uri-only
        # input is not, so fetch it through the cache in-container.
        if artifact.local_dir is not None or artifact.uri is None:
            return artifact.target_path
        return _resolve_uri(artifact.uri, include=artifact.include)
    if artifact.local_dir is not None:
        return artifact.local_dir
    if artifact.uri is not None:
        return _resolve_uri(artifact.uri, include=artifact.include)
    if artifact.required:
        raise FileNotFoundError(
            f"ArtifactInput {artifact.name!r} has no uri/local_dir and is required; "
            "give it a uri (hf://org/repo[@rev]) so it resolves through the rlmesh "
            "cache, or set local_dir= to a path on this host"
        )
    return None


def _resolve_uri(uri: str, *, include: Sequence[str] = ()) -> str:
    if uri.startswith("file://"):
        return uri[len("file://") :]
    if uri.startswith(_HF_SCHEME):
        repo, _, revision = uri[len(_HF_SCHEME) :].partition("@")
        return _snapshot_hf(
            repo, revision or None, allow_patterns=list(include) or None
        )
    raise NotImplementedError(
        f"artifact uri scheme not resolvable on the local path yet: {uri!r}; "
        "use local_dir= for gs://, s3://, https:// until the remote resolver lands"
    )


def _snapshot_hf(
    repo: str, revision: str | None, *, allow_patterns: list[str] | None = None
) -> str:
    if TYPE_CHECKING:
        # huggingface_hub is an optional, unstubbed runtime dep; declare the one call
        # we make so the strict checker sees a real signature without it installed.
        def snapshot_download(
            *,
            repo_id: str,
            revision: str | None,
            cache_dir: str,
            allow_patterns: list[str] | None,
        ) -> str: ...
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "resolving an hf:// artifact requires huggingface_hub; install it "
                "with `pip install rlmesh[hf]` (or `pip install huggingface_hub`)"
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
    """Load a HuggingFace policy from inside ``load()``.

    ``loader`` is an explicit ``module:Class`` string, e.g.
    ``"transformers:AutoModel"`` or ``"lerobot:SmolVLAPolicy"``. With ``local_dir``
    set, weights load from that directory and ``repo`` is only a loader hint;
    otherwise ``repo@revision`` resolves through the rlmesh cache. Returns the
    model, or ``(model, processor)`` when ``processor`` is given.
    """
    source = local_dir if local_dir is not None else _snapshot_hf(repo, revision)
    model = _from_pretrained(
        loader, source, trust_remote_code=trust_remote_code, **kwargs
    )
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
