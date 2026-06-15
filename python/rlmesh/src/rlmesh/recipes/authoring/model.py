"""Class-style authoring: subclass ``ModelRecipe`` to define a policy.

The model-side sibling of :class:`EnvRecipe`. One class is the policy: set the
recipe data (``name``/``build``/``setup``/``spec``/``inputs``), then write
``load()`` to build the model into ``self`` and ``predict()`` to map one
observation to one action. ``reset()``/``close()`` are optional hooks.

``to_recipe()`` projects the class to an inert :class:`Recipe` without importing
the model's dependencies, so keep every heavy import inside ``load()``. Because a
method return annotation is evaluated at class-definition time, every module
defining a subclass must start with ``from __future__ import annotations``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard, TypeVar

from .._artifacts import ArtifactConsumer, enter_recipe_context, merged_inputs
from .._schema import ArtifactInput, Build, PyMake, Recipe, RecipeValidationError, Setup
from ._common import instantiate, require_importable_name

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

#: Where per-construction parameters belong (named in the no-arg __init__ error).
_PARAM_HINT = (
    "Put construction parameters in load(self) (weights ride ArtifactInput), "
    "not __init__."
)


class _Delegated:
    """Sentinel: the model self-adapts, so resolve no adapter (vs ``spec=None``)."""

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
        """Build the model into ``self`` (heavy imports here); runs once per process.

        Resolve mounted weights with ``self.input_path(name)``. No return value --
        ``self`` is the policy.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define load(self) -> None"
        )

    def predict(self, observation: Any) -> Any:
        """Map one observation to one action. Subclasses must override.

        With a ``spec``, the observation arrives in the model's declared input
        format -- the resolved adapter has already transformed it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define predict(self, observation)"
        )

    def reset(self) -> None:
        """Per-episode hook before the first observation (e.g. clear an action queue)."""

    def close(self) -> None:
        """Release the model and any handles. The default does nothing."""

    @classmethod
    def _rlmesh_load(cls) -> ModelRecipe:
        """The in-container lifecycle the projected entrypoint runs (no kwargs)."""
        return construct_authored_model(cls, in_container=True)

    @classmethod
    def to_recipe(cls) -> Recipe:
        """Project this class to an inert ``Recipe(kind='model')``, executing nothing.

        Raises if the class is not importable by ``module:Class`` (defined in
        ``__main__`` or a local scope), or if the ``spec`` is local-only (it carries
        an ``InlineCustomInput`` or ``CustomEncoding``).
        """
        name = require_importable_name(cls, kind="ModelRecipe")
        spec = cls.spec
        adapter: ModelSpec | None = None
        if spec is not None and spec is not DELEGATED:
            _reject_local_only_spec(spec)
            adapter = spec
        entrypoint = f"{cls.__module__}:{cls.__qualname__}.{_LOAD}"
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
    """Reject a spec that cannot cross the wire (in-process callables / host transforms)."""
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
    for component in spec.action.components:
        if isinstance(component.encoding, CustomEncoding):
            raise RecipeValidationError(
                f"ModelSpec action role {component.role!r} uses a CustomEncoding, "
                "which is local-only; use a native-vocabulary RotationEncoding for "
                "a sandboxed model"
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


def construct_authored_model(
    cls: type[_M],
    *,
    in_container: bool = False,
    load_kwargs: dict[str, object] | None = None,
    artifacts: Sequence[ArtifactInput] = (),
) -> _M:
    """Construct a ``ModelRecipe`` in-process (or in-container) and return the policy.

    Applies ``setup``, resolves the declared ``inputs`` mounts, instantiates, runs
    ``load()``. ``load_kwargs`` forwards construction-time keywords to ``load()`` on
    the in-process ``Model`` path only -- ``to_recipe()`` does not bake them and the
    container entrypoint (:meth:`ModelRecipe._rlmesh_load`) runs with none. For a
    sandboxed or registered model, select a variant via ``setup.env`` (member params,
    which cross the wire) or a per-variant recipe, not ``load_kwargs``.
    """
    from .._construct import apply_setup

    # In a container the bootstrap has already applied the recipe's setup with the
    # RLMESH_PARAMS_JSON member overrides merged in; re-applying the class's
    # original setup here would clobber that selection back to the default.
    if not in_container:
        apply_setup(cls.setup)
    instance = instantiate(cls, param_hint=_PARAM_HINT)
    instance._rlmesh_inputs = merged_inputs(cls.inputs, artifacts)  # pyright: ignore[reportPrivateUsage]
    instance._rlmesh_in_container = in_container  # pyright: ignore[reportPrivateUsage]
    with enter_recipe_context(instance):
        if load_kwargs:
            instance.load(**load_kwargs)  # type: ignore[arg-type]
        else:
            instance.load()
    return instance
