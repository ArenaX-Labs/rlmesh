"""Class-style authoring: subclass ``ModelRecipe`` to define a policy.

The model-side sibling of :class:`EnvRecipe`. The headline shape is that **one
class IS the policy** (FINAL_API_SPEC §3.2): set the recipe DATA
(``name``/``build``/``setup``/``spec``/``inputs``), then write ``load()`` (build
the model into ``self``) and ``predict()`` (one observation -> one action). The
instance lives as the policy for the whole eval. ``reset()`` (per-episode) and
``close()`` (teardown) are optional hooks.

The projection (:meth:`ModelRecipe.to_recipe`) is execution-free and import-safe,
exactly like ``EnvRecipe.to_recipe``: it reads class attributes and computes the
``module:Class._rlmesh_load`` entrypoint string without importing the model's
heavy dependencies. Put every heavy import inside ``load()``.

Because a method's return annotation is evaluated at class-definition time, every
module defining a ``ModelRecipe`` subclass must start with
``from __future__ import annotations``.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard

from ._artifacts import enter_recipe_context, resolve_inputs
from ._schema import ArtifactInput, Build, PyMake, Recipe, RecipeValidationError, Setup

if TYPE_CHECKING:
    from rlmesh.adapters import ModelSpec

__all__ = [
    "DELEGATED",
    "ModelRecipe",
    "as_authored_model_recipe",
    "construct_authored_model",
    "is_model_recipe",
]

#: The classmethod the projected entrypoint resolves to; runs the load lifecycle.
_LOAD = "_rlmesh_load"


class _Delegated:
    """Sentinel: this model adapts its own observations -- do NOT resolve an adapter.

    Distinct from ``spec=None`` (FINAL_API_SPEC §12, blocker B9): ``None`` means
    "no spec given" and, against a *tagged* env, is treated as a likely bug (the
    server fails loud); ``DELEGATED`` is the explicit, sanctioned "I self-adapt"
    contract (e.g. GR00T's internal modality transform).
    """

    _instance: ClassVar[_Delegated | None] = None

    def __new__(cls) -> _Delegated:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "DELEGATED"


#: Public sentinel for ``ModelRecipe.spec`` / ``ModelServer(spec=)``.
DELEGATED = _Delegated()


class ModelRecipe:
    """Base class for authoring a policy and its recipe together.

    Subclasses set the data attributes and define the policy::

        from __future__ import annotations
        import rlmesh
        from rlmesh.recipes import Build, PipInstall, ArtifactInput, hf_load


        class SmolVLA(rlmesh.ModelRecipe):
            name = "policy/smolvla"
            build = Build(pip=[PipInstall(["lerobot==0.4.0"])], gpu=True)
            inputs = (ArtifactInput("weights", "/weights/smolvla",
                                    uri="hf://lerobot/smolvla_base@<sha>"),)
            spec = ModelSpec(inputs=(...), action=...)

            def load(self):
                self._policy = hf_load("lerobot/smolvla_base", loader="lerobot:SmolVLAPolicy",
                                       local_dir=self.input_path("weights"))

            def predict(self, observation):
                return self._policy.select_action(observation)


        rlmesh.register(SmolVLA)

    The class is referenced by name from the container; only ``load()``/
    ``predict()``/``reset()``/``close()`` run there, and only the inert projected
    recipe travels.
    """

    #: ``namespace/name`` identity. Required to register/serve/sandbox.
    name: ClassVar[str]
    #: The build phase (image). Shared phase-1 vocabulary with ``EnvRecipe``.
    build: ClassVar[Build] = Build()
    #: Construct-time DATA (env vars, files), applied before ``load()``.
    setup: ClassVar[Setup] = Setup()
    #: The full model-side adapter content. ``None`` = no adaptation; ``DELEGATED``
    #: = the model self-adapts (do not resolve); a ``ModelSpec`` = resolve against
    #: the env's tags.
    spec: ClassVar[Any] = None
    #: Runtime weight/asset mounts (FINAL_API_SPEC §4.4). Never baked into the image.
    inputs: ClassVar[tuple[ArtifactInput, ...]] = ()

    def load(self) -> None:
        """Build the model INTO ``self``. Heavy imports here; runs once per process.

        Resolve mounted weights with ``self.input_path(name)``. No return value:
        ``self`` IS the policy.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define load(self) -> None"
        )

    def predict(self, observation: Any) -> Any:
        """Map one (already adapter-transformed) observation to one action.

        With a ``spec``, ``observation`` is in the model's declared input format
        (the resolved adapter has already transformed it). Subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define predict(self, observation)"
        )

    def reset(self) -> None:
        """Optional per-episode hook (e.g. clear an internal action-chunk queue).

        Invoked at every episode boundary by the runtime, before the policy sees
        the first observation of the episode. The default is a no-op.
        """

    def close(self) -> None:
        """Optional teardown: release the GPU model / handles. No-op default."""

    def input_path(self, name: str) -> str:
        """Resolve a declared :class:`ArtifactInput` mount by name to its local path.

        Call inside ``load()``. The path points at the in-container mount
        (``target_path``) under the sandbox, or the resolved host/cache path under
        the local in-process path.
        """
        resolved: dict[str, str] = getattr(self, "_rlmesh_resolved_inputs", {})
        try:
            return resolved[name]
        except KeyError:
            declared = ", ".join(a.name for a in type(self).inputs) or "<none>"
            raise RecipeValidationError(
                f"{type(self).__name__}.input_path({name!r}): no such ArtifactInput; "
                f"declared inputs: {declared}"
            ) from None

    @classmethod
    def _rlmesh_load(cls) -> ModelRecipe:
        """The lifecycle the projected recipe entrypoint runs (in-container).

        Mirror of ``EnvRecipe._rlmesh_construct``. Takes NO kwargs: weights ride
        ``ArtifactInput`` mounts (materialized by the bootstrap before this runs),
        not load kwargs. Instantiates, resolves mounts to their ``target_path``,
        runs ``load()``, and returns the loaded instance (which IS the policy).
        """
        return construct_authored_model(cls, in_container=True)

    @classmethod
    def to_recipe(cls) -> Recipe:
        """Project this class to an inert ``Recipe(kind='model')`` (executes nothing).

        Reads ``name``/``build``/``setup``/``spec``/``inputs`` and computes the
        ``module:Class._rlmesh_load`` entry string. Raises if the class is not
        importable by that path (defined in ``__main__`` or a local scope), or if
        the ``spec`` is local-only (carries an ``InlineCustomInput``/``CustomEncoding``).
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
        module = cls.__module__
        qualname = cls.__qualname__
        if module == "__main__" or "<locals>" in qualname:
            raise RecipeValidationError(
                f"ModelRecipe {name!r} is defined in {module}:{qualname}, which the "
                "container cannot import; define it in an installed module"
            )
        spec = cls.spec
        adapter: object | None = None
        if spec is not None and spec is not DELEGATED:
            _reject_local_only_spec(spec)
            adapter = spec
        entrypoint = f"{module}:{qualname}.{_LOAD}"
        return Recipe(
            name=name,
            kind="model",
            make=PyMake(entrypoint=entrypoint),
            build=cls.build,
            setup=cls.setup,
            adapter=adapter,
            inputs=cls.inputs,
        )

    @classmethod
    def check(cls) -> None:
        """Validate this recipe without importing its dependencies (see :func:`check`)."""
        from ._check import check as _check

        _check(cls.to_recipe())


def _reject_local_only_spec(spec: ModelSpec) -> None:
    """Raise a clear error for a spec that cannot cross the wire (FINAL_API_SPEC §3.3).

    An ``InlineCustomInput`` (in-process callable) or a ``CustomEncoding`` inside a
    ``StateInput`` makes the spec local-only: it cannot be projected to a ``Recipe``
    or sandboxed. Raise *before* the generic ``ModelSpec.to_dict()`` ValueError so
    the message points at the fix.
    """
    from rlmesh.adapters import (
        CustomEncoding,
        InlineCustomInput,
        StateInput,
    )

    for inp in spec.inputs:
        if isinstance(inp, InlineCustomInput):
            raise RecipeValidationError(
                f"ModelSpec input {inp.key!r} is an InlineCustomInput (in-process "
                "callable) and is local-only; the sandbox/register path requires "
                "EntrypointCustomInput"
            )
        if isinstance(inp, StateInput) and any(
            isinstance(c.encoding, CustomEncoding) for c in inp.components
        ):
            raise RecipeValidationError(
                f"ModelSpec input {inp.key!r} uses a CustomEncoding, which is "
                "local-only; use a native-vocabulary RotationEncoding for a "
                "sandboxed model"
            )


def is_model_recipe(source: object) -> TypeGuard[type[ModelRecipe]]:
    """Whether ``source`` is a ``ModelRecipe`` subclass (the class, not an instance)."""
    return isinstance(source, type) and issubclass(source, ModelRecipe)


def as_authored_model_recipe(source: object) -> Recipe | None:
    """Project an authored model source to a ``Recipe``, or None if it is neither."""
    if is_model_recipe(source):
        return source.to_recipe()
    if isinstance(source, Recipe) and source.kind == "model":
        return source
    return None


def _instantiate(cls: type[ModelRecipe]) -> ModelRecipe:
    """Construct ``cls()`` with a recipe-aware error if it requires constructor args."""
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls()
    required = [
        name
        for name, p in signature.parameters.items()
        if p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if required:
        raise TypeError(
            f"{cls.__qualname__} is instantiated with no arguments, but its __init__ "
            f"requires {required}. Put construction parameters in load(self) "
            "(weights ride ArtifactInput), not __init__."
        )
    return cls()


def construct_authored_model(
    cls: type[ModelRecipe],
    *,
    in_container: bool = False,
    load_kwargs: dict[str, object] | None = None,
    artifacts: Sequence[ArtifactInput] = (),
) -> ModelRecipe:
    """Construct a ``ModelRecipe`` IN-PROCESS (or in-container), returning the policy.

    Mirror of ``construct_authored``. Applies ``setup``, resolves the declared
    ``inputs`` mounts (to ``target_path`` in-container, or the host/cache path
    locally), instantiates, runs ``load()``, and returns the loaded instance.

    ``load_kwargs`` (FINAL_API_SPEC §12, edge 14) lets one recipe serve multiple
    construction-time configurations (e.g. GR00T ``embodiment_tag``) without a
    per-variant recipe; they are forwarded to ``load(**load_kwargs)`` when the
    subclass's ``load`` accepts them.
    """
    from ._build import apply_setup

    apply_setup(cls.setup)
    instance = _instantiate(cls)
    resolved = resolve_inputs(cls.inputs, in_container=in_container, overrides=artifacts)
    instance._rlmesh_resolved_inputs = resolved  # type: ignore[attr-defined]
    with enter_recipe_context(instance):
        if load_kwargs:
            instance.load(**load_kwargs)  # type: ignore[arg-type]
        else:
            instance.load()
    return instance
