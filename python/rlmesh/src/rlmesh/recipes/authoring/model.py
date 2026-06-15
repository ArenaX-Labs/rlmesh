"""Class-style authoring: subclass ``ModelRecipe`` to define a policy.

The model-side sibling of :class:`EnvRecipe`. One class is the policy: set the
recipe data (``name``/``build``/``setup``/``spec``/``inputs``), then write
``load()`` to build the model into ``self`` and ``predict()`` to map one
observation to one action. The instance lives as the policy for the whole eval;
``reset()`` and ``close()`` are optional hooks.

``to_recipe()`` projects the class to an inert :class:`Recipe` without importing
the model's dependencies -- it reads class attributes and computes the
``module:Class._rlmesh_load`` entrypoint string. Keep every heavy import inside
``load()``.

A method's return annotation is evaluated at class-definition time, so every
module defining a ``ModelRecipe`` subclass must start with
``from __future__ import annotations``.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard, TypeVar

from .._artifacts import ArtifactConsumer, enter_recipe_context, merged_inputs
from .._schema import ArtifactInput, Build, PyMake, Recipe, RecipeValidationError, Setup

if TYPE_CHECKING:
    from rlmesh.adapters import ModelSpec

__all__ = [
    "DELEGATED",
    "ModelRecipe",
    "as_authored_model_recipe",
    "construct_authored_model",
    "is_model_recipe",
]

_LOAD = "_rlmesh_load"


class _Delegated:
    """Sentinel: the model adapts its own observations, so do not resolve an adapter.

    ``spec=None`` means no spec was given, and against a tagged env that is treated
    as a likely bug. ``DELEGATED`` is the explicit "I self-adapt" contract for a
    model like GR00T that runs its own modality transform.
    """

    _instance: ClassVar[_Delegated | None] = None

    def __new__(cls) -> _Delegated:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "DELEGATED"


DELEGATED = _Delegated()


class ModelRecipe(ArtifactConsumer):
    """Base class for authoring a policy and its recipe together.

    Subclasses set the data attributes and define the policy::

        from __future__ import annotations
        import rlmesh
        from rlmesh.recipes import Build, PipInstall, ArtifactInput, hf_load


        class SmolVLA(rlmesh.ModelRecipe):
            name = "policy/smolvla"
            build = Build(pip=[PipInstall(["lerobot==0.4.0"])], gpu=True)
            inputs = (
                ArtifactInput(
                    "weights", "/weights/smolvla", uri="hf://lerobot/smolvla_base@<sha>"
                ),
            )
            spec = ModelSpec(inputs=(...), action=...)

            def load(self):
                self._policy = hf_load(
                    "lerobot/smolvla_base",
                    loader="lerobot:SmolVLAPolicy",
                    local_dir=self.input_path("weights"),
                )

            def predict(self, observation):
                return self._policy.select_action(observation)


        rlmesh.register(SmolVLA)

    The container references the class by name; only ``load()``/``predict()``/
    ``reset()``/``close()`` run there, and only the inert projected recipe travels.
    """

    #: ``namespace/name`` identity. Required to register, serve, or sandbox.
    name: ClassVar[str]
    #: The build phase. Shares the phase-1 vocabulary with ``EnvRecipe``.
    build: ClassVar[Build] = Build()
    #: Construct-time data (env vars, files), applied before ``load()``.
    setup: ClassVar[Setup] = Setup()
    #: The model's adapter content. ``None`` for no adaptation, ``DELEGATED`` when
    #: the model self-adapts, or a ``ModelSpec`` to resolve against the env's tags.
    spec: ClassVar[Any] = None
    #: ``inputs`` (runtime weight/asset mounts) and ``input_path`` come from
    #: :class:`ArtifactConsumer`, shared with ``EnvRecipe``.

    def load(self) -> None:
        """Build the model into ``self``, with heavy imports here; runs once per process.

        Resolve mounted weights with ``self.input_path(name)``. There is no return
        value -- ``self`` is the policy.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define load(self) -> None"
        )

    def predict(self, observation: Any) -> Any:
        """Map one observation to one action.

        With a ``spec``, the observation arrives in the model's declared input
        format because the resolved adapter has already transformed it. Subclasses
        must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define predict(self, observation)"
        )

    def reset(self) -> None:
        """Per-episode hook, run before the policy sees the first observation.

        Use it to clear per-episode state such as an internal action-chunk queue.
        The default does nothing.
        """

    def close(self) -> None:
        """Release the model and any handles. The default does nothing."""

    @classmethod
    def _rlmesh_load(cls) -> ModelRecipe:
        """The in-container lifecycle the projected entrypoint runs.

        Takes no kwargs: weights resolve through ``input_path`` -- a bind-mounted
        input from its ``target_path``, a uri-only input fetched in-container through
        the cache, or an externally materialized path from the run contract.
        Instantiates, runs ``load()``, and returns the loaded policy.
        """
        return construct_authored_model(cls, in_container=True)

    @classmethod
    def to_recipe(cls) -> Recipe:
        """Project this class to an inert ``Recipe(kind='model')``, executing nothing.

        Raises if the class is not importable by ``module:Class`` (defined in
        ``__main__`` or a local scope), or if the ``spec`` is local-only (it carries
        an ``InlineCustomInput`` or ``CustomEncoding``).
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
        adapter: ModelSpec | None = None
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
        from .._sandbox_validate import check as _check

        _check(cls.to_recipe())


def _reject_local_only_spec(spec: ModelSpec) -> None:
    """Reject a spec that cannot cross the wire, before the generic ValueError.

    An ``InlineCustomInput`` holds an in-process callable, and a ``CustomEncoding``
    inside a ``StateInput`` holds host-side transforms; either makes the spec
    local-only, so it cannot be projected to a ``Recipe`` or sandboxed.
    """
    from rlmesh.adapters import CustomEncoding, InlineCustomInput, StateInput

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


_M = TypeVar("_M", bound="ModelRecipe")


def _instantiate(cls: type[_M]) -> _M:
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
    cls: type[_M],
    *,
    in_container: bool = False,
    load_kwargs: dict[str, object] | None = None,
    artifacts: Sequence[ArtifactInput] = (),
) -> _M:
    """Construct a ``ModelRecipe`` in-process (or in-container) and return the policy.

    Applies ``setup``, resolves the declared ``inputs`` mounts, instantiates, runs
    ``load()``, and returns the loaded instance. ``load_kwargs`` lets one recipe
    serve several construction-time configurations (such as a GR00T
    ``embodiment_tag``) without a per-variant recipe.
    """
    from .._construct import apply_setup

    apply_setup(cls.setup)
    instance = _instantiate(cls)
    instance._rlmesh_inputs = merged_inputs(cls.inputs, artifacts)  # pyright: ignore[reportPrivateUsage]
    instance._rlmesh_in_container = in_container  # pyright: ignore[reportPrivateUsage]
    with enter_recipe_context(instance):
        if load_kwargs:
            instance.load(**load_kwargs)  # type: ignore[arg-type]
        else:
            instance.load()
    return instance
