"""Shared Python model wrapper: construct from any source, eval against an env, serve."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast

from .._framework_bridge import ValueBridge
from ..types import Value

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModel, ServeOptions

    from ..recipes._schema import ArtifactInput
    from ._eval import RunResult

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
LifecycleCallback = Callable[[], None]
PredictFn = Callable[[ObsT], ActT]


class ModelBase(Generic[ObsT, ActT]):
    """A model: a predict callable, or the ``ModelRecipe`` it is built from.

    ``run(env, seeds=...)`` drives the model against an env and returns a typed
    :class:`RunResult` -- it resolves the adapter from the env's published tags and
    this model's spec, so ``predict`` works in the model's own input/output format
    with no per-env glue. ``serve()`` hosts the model as an endpoint for the runtime
    to dial.

    Args:
        source: A predict callable, a ``ModelRecipe`` (class or instance), a
            ``kind='model'`` Recipe, or a registered model name. A recipe carries
            its own spec, lifecycle, and weight mounts.
        spec: Optional :class:`rlmesh.adapters.ModelSpec` for a callable source
            (a recipe declares its own); makes this an *adapted* model.
        on_reset / on_episode_end / on_close: Optional lifecycle callbacks.
        artifacts / load_kwargs: Per-run overrides applied when ``source`` is a
            recipe constructed in-process.
        trust_entrypoints: Allow ``module:callable`` custom-input entrypoints in a
            spec to be imported during adapter resolution.

    Examples:
        >>> from rlmesh.numpy import Model
        >>> result = Model(lambda observation: 0).run("127.0.0.1:5555", seeds=[0])
        >>> result.mean_reward
        0.0
    """

    _bridge: ClassVar[ValueBridge]
    #: Framework remote-env client used when ``run`` is given a bare address.
    _remote_env_cls: ClassVar[type | None] = None

    def __init__(
        self,
        source: Callable[..., object] | object,
        *,
        spec: object | None = None,
        on_reset: LifecycleCallback | None = None,
        on_episode_end: LifecycleCallback | None = None,
        on_close: LifecycleCallback | None = None,
        artifacts: Sequence[ArtifactInput] = (),
        load_kwargs: Mapping[str, object] | None = None,
        trust_entrypoints: bool = False,
    ) -> None:
        self._bridge.ensure_available()
        from ._eval import coerce_model

        coerced = coerce_model(
            source,
            spec=spec,
            artifacts=tuple(artifacts),
            load_kwargs=dict(load_kwargs) if load_kwargs else None,
        )
        self._raw_predict = cast("PredictFn[ObsT, ActT]", coerced.predict)
        self._spec = coerced.spec
        self._policy = coerced.policy
        self._on_reset = on_reset if on_reset is not None else coerced.on_reset
        self._on_close = on_close if on_close is not None else coerced.on_close
        self._on_episode_end = on_episode_end
        self._trust_entrypoints = trust_entrypoints
        # An adapted (spec'd) model resolves its adapter only at run(env), so its
        # native worker is a fail-loud placeholder -- run() does the adapting.
        worker_predict = (
            self._raw_predict
            if self._spec is None
            else cast("PredictFn[ObsT, ActT]", _unwired)
        )
        self._install_worker(worker_predict, self._on_reset)

    @property
    def spec(self) -> object | None:
        """The model's content: a ``ModelSpec``, ``DELEGATED``, or ``None``."""
        return self._spec

    def _install_worker(
        self,
        predict_fn: PredictFn[ObsT, ActT],
        on_reset: LifecycleCallback | None,
    ) -> None:
        """Build the native model worker around ``predict_fn`` (the serve path)."""
        try:
            from rlmesh._rlmesh import PyModel
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError("Failed to import _rlmesh native module.") from e

        def wrapped_predict(observation: Value) -> Value:
            decoded = cast(ObsT, self._bridge.decode(observation))
            action = predict_fn(decoded)
            return self._bridge.encode(action)

        self._worker: PyModel = PyModel(
            predict_fn=wrapped_predict,
            on_reset=on_reset,
            on_episode_end=self._on_episode_end,
            on_close=self._on_close,
        )

    def run(
        self,
        env_or_address: object,
        *,
        seeds: Sequence[int] | None = None,
        max_episodes: int | None = None,
        instruction: str | None = None,
        close_env: bool = False,
        token: str = "",
    ) -> RunResult:
        """Drive this model against an env and return a :class:`RunResult`.

        Resolves the adapter from the env's tags and this model's spec, then runs a
        per-episode loop. ``seeds`` gives a per-episode seed and sets the episode
        count unless ``max_episodes`` is given; ``instruction`` is written into the
        model's text inputs each episode. ``env_or_address`` is an env object
        exposing ``reset``/``step`` (e.g. a ``RemoteEnv``), an object with an
        ``address``, or a bare address string the loop dials.
        """
        from ._eval import evaluate

        return evaluate(
            self._raw_predict,
            self._spec,
            env_or_address,
            seeds=seeds,
            max_episodes=max_episodes,
            instruction=instruction,
            close_env=close_env,
            token=token,
            on_reset=self._on_reset,
            on_episode_end=self._on_episode_end,
            on_close=self._on_close,
            trust_entrypoints=self._trust_entrypoints,
            remote_env_cls=type(self)._remote_env_cls,
        )

    def serve(
        self, address: str, *, token: str = "", options: ServeOptions | None = None
    ) -> None:
        """Host this model as an endpoint (blocking).

        A spec-less model serves directly. A spec'd model must be driven with
        ``run(env)``: its adapter resolves from an env contract, which a passive
        endpoint only sees on dial-in (not implemented), so serving the raw predict
        would silently skip the spec's transforms.
        """
        from ..adapters import ModelSpec

        if isinstance(self._spec, ModelSpec):
            raise NotImplementedError(
                "serve() cannot host a spec'd model yet: its adapter resolves from "
                "an env contract, which a served endpoint only sees on dial-in (not "
                "implemented). Drive it with Model(model).run(env), or use "
                "spec=DELEGATED if the model adapts its own observations."
            )
        self._worker.serve(address, token, options)

    def run_local(self, env_address: str, *, token: str = "") -> None:
        """Native worker loop against a remote env, until interrupted (no metrics)."""
        self._worker.run_local(env_address, token)

    def run_local_for_episodes(
        self, env_address: str, *, token: str = "", max_episodes: int
    ) -> None:
        """Native worker loop against a remote env for a fixed episode count (no metrics)."""
        self._worker.run_local_for_episodes(env_address, token, max_episodes)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _unwired(_observation: object) -> object:
    raise RuntimeError(
        "this model has a spec; its adapter resolves from the env contract. Drive "
        "it with .run(env) passing an env object, not serve()/run_local()."
    )


__all__ = ["LifecycleCallback", "ModelBase", "PredictFn"]
