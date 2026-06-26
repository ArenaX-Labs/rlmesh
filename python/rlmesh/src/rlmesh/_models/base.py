"""Shared Python model wrapper: construct from any source, eval against an env, serve."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from .._framework_bridge import ValueBridge
from ..types import Value

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModel, ServeOptions

    from ._eval import RunResult, Session

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
LifecycleCallback = Callable[[], None]
PredictFn = Callable[[ObsT], ActT]


class ModelBase(Generic[ObsT, ActT]):
    """A model: a policy you serve or drive against an env.

    Construct one two ways:

    * **Wrap** a predict callable -- ``Model(lambda obs: ...)``.
    * **Subclass** ``Model`` and override ``predict`` (and optionally ``load`` for
      weight loading, ``reset``/``close`` lifecycle hooks, and the ``spec`` class
      attribute), then instantiate it -- ``class P(Model): ...`` then ``P()``.

    ``run(env, seeds=...)`` drives the model against an env and returns a typed
    :class:`RunResult` -- it resolves the adapter from the env's published tags and
    this model's spec, so ``predict`` works in the model's own input/output format
    with no per-env glue. ``serve()`` hosts the model as an endpoint for the runtime
    to dial.

    Args:
        source: A predict callable. Omit it when subclassing (``predict`` is the
            source then).
        spec: Optional :class:`rlmesh.adapters.ModelSpec`; makes this an *adapted*
            model. Pass :data:`rlmesh.NO_ADAPTER` to explicitly skip adapter
            resolution. Overrides the ``spec`` class attribute when both are set.
        on_reset / on_episode_end / on_close: Optional lifecycle callbacks (they
            override a subclass's ``reset``/``close`` methods).
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
    #: The model's content: a ``ModelSpec``, ``NO_ADAPTER``, or ``None``. Set it as a
    #: class attribute when subclassing; the ``spec=`` kwarg overrides it per instance.
    spec: object | None = None

    def __init__(
        self,
        source: Callable[..., object] | object | None = None,
        *,
        spec: object | None = None,
        on_reset: LifecycleCallback | None = None,
        on_episode_end: LifecycleCallback | None = None,
        on_close: LifecycleCallback | None = None,
        trust_entrypoints: bool = False,
    ) -> None:
        self._bridge.ensure_available()

        if source is None and type(self).predict is not ModelBase.predict:  # pyright: ignore[reportUnknownMemberType]
            # Subclass-authoring mode: a Model subclass that overrides predict (and
            # optionally load/reset/close/spec). Load weights once, before the worker
            # is built, then drive self.predict.
            self.load()
            raw_predict: Callable[..., object] = self.predict
            resolved_spec = spec if spec is not None else type(self).spec
            coerced_on_reset: LifecycleCallback | None = self.reset
            coerced_on_close: LifecycleCallback | None = self.close
            policy: object = self
        elif source is None:
            raise TypeError(
                "Model() needs a predict callable, e.g. Model(lambda obs: ...), "
                "or subclass Model and override predict()."
            )
        else:
            from ._eval import coerce_model

            coerced = coerce_model(source, spec=spec)
            raw_predict = coerced.predict
            resolved_spec = coerced.spec
            coerced_on_reset = coerced.on_reset
            coerced_on_close = coerced.on_close
            policy = coerced.policy

        self._raw_predict = cast("PredictFn[ObsT, ActT]", raw_predict)
        self.spec = resolved_spec
        self._policy = policy
        self._on_reset = on_reset if on_reset is not None else coerced_on_reset
        self._on_close = on_close if on_close is not None else coerced_on_close
        self._on_episode_end = on_episode_end
        self._trust_entrypoints = trust_entrypoints
        self._install_worker(self._on_reset)

    def load(self) -> None:
        """Load weights into ``self`` (``from_pretrained`` etc.); heavy imports here.

        Optional subclass hook; a no-op by default. Called once during ``__init__``
        (subclass mode only) before the native worker is built.
        """

    def predict(self, observation: ObsT) -> ActT:
        """Map one observation to an action (or an action chunk per ``spec``).

        Override when subclassing ``Model``; the default raises. A model built by
        wrapping a predict callable uses that callable instead of this method.
        """
        raise NotImplementedError(
            "Model subclasses must override predict(), or construct a Model by "
            "wrapping a predict callable, e.g. Model(lambda obs: ...)."
        )

    def reset(self) -> None:
        """Optional: called at each episode boundary (no-op by default)."""

    def close(self) -> None:
        """Optional: release resources at the end of a run (no-op by default)."""

    def _install_worker(self, on_reset: LifecycleCallback | None) -> None:
        """Build the native model worker (the serve path).

        A spec'd model's adapter resolves per route at ``configure_route`` (the
        served endpoint receives the env contract there): ``configure`` resolves
        it from the route's contract and the native worker applies it around the
        raw predict. A spec-less / ``NO_ADAPTER`` model serves its own predict.
        """
        try:
            from rlmesh._rlmesh import PyModel
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError("Failed to import _rlmesh native module.") from e

        from ._eval import resolve_route_adapter

        spec = self.spec
        bridge = self._bridge
        raw_predict = self._raw_predict
        trust = self._trust_entrypoints

        def configure(env_contract: object) -> object:
            # The served route resolves to a native plan plus neutral host holes
            # the Rust engine drives (it owns the per-episode frame buffers and
            # adapter application); a spec-less / NO_ADAPTER model returns None.
            adapter = resolve_route_adapter(
                spec, cast("Any", env_contract), trust_entrypoints=trust
            )
            return adapter.serve_route(bridge) if adapter is not None else None

        def predict_neutral(observation: Value) -> Value:
            # The engine has already applied the adapter (declarative transform,
            # frame-stacking, customs, enc-shims) in Rust, so the obs handed here
            # is the final model input; bridge it into the framework, run the user
            # predict, and bridge the action back. A spec-less route hands the raw
            # observation through the identical path (no adapter).
            return bridge.encode(raw_predict(cast(ObsT, bridge.decode(observation))))

        self._worker: PyModel = PyModel(
            predict_fn=predict_neutral,
            configure_fn=configure,
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
            self.spec,
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
            bridge=type(self)._bridge,
        )

    def session(
        self,
        env_or_address: object,
        *,
        instruction: str | None = None,
        close_env: bool = False,
        token: str = "",
        trust_entrypoints: bool | None = None,
    ) -> Session[ObsT, ActT]:
        """Bind this model to an env and return a :class:`Session` to drive by hand.

        The manual counterpart of :meth:`run`: drive ``reset`` / ``predict`` / ``step``
        yourself, or call :meth:`Session.run` to pump whole episodes. ``env_or_address``
        is an env object, a remote-env handle, or an address string (see :meth:`run`).
        """
        from ._eval import Session

        return Session(
            predict=self._raw_predict,
            spec=self.spec,
            env=env_or_address,
            on_reset=self._on_reset,
            on_episode_end=self._on_episode_end,
            on_close=self._on_close,
            trust_entrypoints=(
                self._trust_entrypoints
                if trust_entrypoints is None
                else trust_entrypoints
            ),
            bridge=self._bridge,
            remote_env_cls=type(self)._remote_env_cls,
            instruction=instruction,
            close_env=close_env,
            token=token,
        )

    def serve(
        self, address: str, *, token: str = "", options: ServeOptions | None = None
    ) -> None:
        """Host this model as an endpoint (blocking).

        A spec'd model resolves its adapter per route from the env contract the
        ``configure_route`` handshake delivers, then applies it around predict; a
        spec-less / ``NO_ADAPTER`` model serves its own predict directly.
        """
        self._worker.serve(address, token, options)

    def run_local(self, env_address: str, *, token: str = "") -> None:
        """Native worker loop against a remote env, until the env ends.

        Runs the session to completion for its side effects. Telemetry is
        surfaced on the serving runtime via its ``on_telemetry`` hook, not
        returned here.
        """
        return self._worker.run_local(env_address, token)

    def run_local_for_episodes(
        self, env_address: str, *, token: str = "", max_episodes: int
    ) -> None:
        """Native worker loop against a remote env for a fixed episode count.

        Runs for the requested episode count for its side effects; see
        :meth:`run_local` for where telemetry is surfaced.
        """
        return self._worker.run_local_for_episodes(env_address, token, max_episodes)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _as_model(model: object) -> ModelBase[Any, Any]:
    """Normalize a model source to a built :class:`ModelBase` instance.

    A ``Model`` instance is used as-is; a ``Model`` subclass *class* is instantiated
    once; a bare predict callable (or duck-typed policy object) is wrapped in the NumPy
    framework ``Model``. (Served ``RemoteModel`` / ``SandboxModel`` handles bind via
    their own ``.session`` and never reach here.)
    """
    if isinstance(model, ModelBase):
        return cast("ModelBase[Any, Any]", model)
    if isinstance(model, type) and issubclass(model, ModelBase):
        return cast("ModelBase[Any, Any]", model())
    from ..numpy import Model

    return Model(cast(object, model))


def session(
    model: object,
    env: object,
    *,
    instruction: str | None = None,
    close_env: bool = False,
    token: str = "",
    trust_entrypoints: bool | None = None,
) -> Session[Any, Any]:
    """Bind a model to an env and return a :class:`Session` to drive by hand or via run().

    ``model`` is a local :class:`Model` (instance, subclass class, or a bare predict
    callable) or a served handle (:class:`RemoteModel` / :class:`SandboxModel`); ``env``
    is a local env, a remote-env handle, or an address string.
    """
    # A handle that knows how to bind itself -- Model, RemoteModel, SandboxModel -- has
    # its own ``.session``; anything else (a callable / subclass class) is normalized.
    binder = getattr(model, "session", None)
    if callable(binder) and not isinstance(model, type):
        return cast(
            "Session[Any, Any]",
            binder(
                env,
                instruction=instruction,
                close_env=close_env,
                token=token,
                trust_entrypoints=trust_entrypoints,
            ),
        )
    return _as_model(model).session(
        env,
        instruction=instruction,
        close_env=close_env,
        token=token,
        trust_entrypoints=trust_entrypoints,
    )


def run(
    model: object,
    env: object,
    *,
    seeds: Sequence[int] | None = None,
    max_episodes: int | None = None,
    instruction: str | None = None,
    close_env: bool = False,
    token: str = "",
    trust_entrypoints: bool | None = None,
) -> RunResult:
    """Drive ``model`` against ``env`` to completion and return a :class:`RunResult`.

    The auto-pump convenience over :func:`rlmesh.session` -- equivalent to
    ``rlmesh.session(model, env).run(seeds=...)``. Works for a local :class:`Model` or a
    served :class:`RemoteModel` / :class:`SandboxModel`.
    """
    return session(
        model,
        env,
        instruction=instruction,
        close_env=close_env,
        token=token,
        trust_entrypoints=trust_entrypoints,
    ).run(seeds=seeds, max_episodes=max_episodes)


__all__ = ["LifecycleCallback", "ModelBase", "PredictFn", "run", "session"]
