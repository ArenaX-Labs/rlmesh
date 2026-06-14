"""Model registration: the class form + the flat ``hf=``/``load=`` sugar.

Registers a model by name into the shared recipe registry (so ``resolve(name)``
sees both kinds, FINAL_API_SPEC §6.5) AND keeps the live ``ModelRecipe`` subclass
so the local in-process path can construct it without an entrypoint round-trip.
Env keywords (``gym=``/``factory=``) and model keywords (``hf=``/``load=``/``spec=``)
are disjoint, so a registration is unambiguously one kind.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

from ..recipes._artifacts import hf_load
from ..recipes._authoring_model import ModelRecipe, is_model_recipe
from ..recipes._registry import register as _register_recipe
from ..recipes._schema import ArtifactInput

__all__ = ["lookup_model_class", "register"]

# name -> live ModelRecipe subclass, so ModelServer(name) constructs in-process
# (the local path) without importing by entrypoint string.
_MODEL_CLASSES: dict[str, type[ModelRecipe]] = {}


def lookup_model_class(name: str) -> type[ModelRecipe] | None:
    """Return the live ``ModelRecipe`` subclass registered under ``name``, or None."""
    return _MODEL_CLASSES.get(name)


@overload
def register(source: type[ModelRecipe], *, overwrite: bool = ...) -> type[ModelRecipe]: ...
@overload
def register(
    source: str,
    *,
    hf: str | None = ...,
    load: str | None = ...,
    spec: Any = ...,
    revision: str | None = ...,
    loader: str = ...,
    trust_remote_code: bool = ...,
    packages: Sequence[str] = ...,
    artifacts: Sequence[ArtifactInput] = ...,
    overwrite: bool = ...,
) -> type[ModelRecipe]: ...
def register(
    source: type[ModelRecipe] | str,
    *,
    hf: str | None = None,
    load: str | None = None,
    spec: Any = None,
    revision: str | None = None,
    loader: str = "transformers:AutoModel",
    trust_remote_code: bool = False,
    packages: Sequence[str] = (),
    artifacts: Sequence[ArtifactInput] = (),
    overwrite: bool = False,
) -> type[ModelRecipe]:
    """Register a model.

    * **Class** -- ``register(MyModelRecipe)`` / ``@register``: stores the projected
      ``kind='model'`` recipe and keeps the live class.
    * **Flat ``hf=``** -- ``register("policy/x", hf="org/repo", spec=SPEC,
      loader="lerobot:SmolVLAPolicy")``: synthesizes a ``ModelRecipe`` whose
      ``load()`` calls :func:`hf_load`. The rung-1 one-liner.
    * **Flat ``load=``** -- ``register("policy/x", load="mod:make_policy", spec=SPEC)``:
      synthesizes a ``ModelRecipe`` whose ``load()`` calls the named factory.
    """
    if is_model_recipe(source):
        if hf or load:
            raise TypeError("register(ModelRecipe) takes no hf=/load=; those are the flat form")
        cls = source
    elif isinstance(source, str):
        cls = _flat_model_class(
            source,
            hf=hf,
            load=load,
            spec=spec,
            revision=revision,
            loader=loader,
            trust_remote_code=trust_remote_code,
            packages=packages,
            artifacts=tuple(artifacts),
        )
    else:
        raise TypeError(
            "rlmesh.models.register() takes a ModelRecipe subclass or a name string "
            f"with hf=/load=; got {type(source).__name__}"
        )
    _register_recipe(cls.to_recipe(), overwrite=overwrite)
    _MODEL_CLASSES[cls.__dict__["name"]] = cls
    return cls


def _flat_model_class(
    name: str,
    *,
    hf: str | None,
    load: str | None,
    spec: Any,
    revision: str | None,
    loader: str,
    trust_remote_code: bool,
    packages: Sequence[str],
    artifacts: tuple[ArtifactInput, ...],
) -> type[ModelRecipe]:
    if (hf is None) == (load is None):
        raise TypeError(f"register({name!r}, ...) needs exactly one of hf= or load=")

    from ..recipes._schema import Build, PipInstall

    build = Build(pip=[PipInstall(list(packages))]) if packages else Build()
    namespace: dict[str, Any] = {
        "name": name,
        "build": build,
        "spec": spec,
        "inputs": tuple(artifacts),
        "__module__": __name__,
    }

    if hf is not None:
        def load_fn(self: ModelRecipe) -> None:
            self._policy = hf_load(  # type: ignore[attr-defined]
                hf, revision=revision, loader=loader, trust_remote_code=trust_remote_code
            )
    else:
        assert load is not None

        def load_fn(self: ModelRecipe) -> None:
            from .._bootstrap.entrypoint import resolve_entrypoint

            factory = resolve_entrypoint(load, label="model loader")
            self._policy = factory()  # type: ignore[attr-defined]

    def predict_fn(self: ModelRecipe, observation: Any) -> Any:
        return _turnkey_predict(self._policy, observation)  # type: ignore[attr-defined]

    namespace["load"] = load_fn
    namespace["predict"] = predict_fn
    cls = type(_class_name(name), (ModelRecipe,), namespace)
    return cls


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("/", "_").split("_")) or "FlatModel"


def _turnkey_predict(policy: Any, observation: Any) -> Any:
    """Best-effort predict for a flat-registered turnkey policy.

    Tries the common policy call conventions, in order, and raises a clear error
    pointing the user to subclass ``ModelRecipe`` for a custom ``predict``.
    """
    for attr in ("select_action", "predict", "act", "get_action"):
        method = getattr(policy, attr, None)
        if callable(method):
            return method(observation)
    if callable(policy):
        return policy(observation)
    raise TypeError(
        "flat-registered policy exposes no select_action/predict/act/get_action and "
        "is not callable; subclass ModelRecipe and define predict(self, observation)"
    )
