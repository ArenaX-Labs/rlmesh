"""Shared Python model wrapper: construct from any source, eval against an env, serve."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from .._value_conversion import ValueBridge
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

        # Subclass-overridden predict corners declare their capability; capture each
        # so the worker advertises it (wrapper-mode models, which supply a single
        # predict callable, never define them).
        def _overridden(method_name: str) -> Callable[..., object] | None:
            if getattr(type(self), method_name) is getattr(ModelBase, method_name):
                return None
            return cast("Callable[..., object]", getattr(self, method_name))

        raw_predict_chunk = _overridden("predict_chunk")
        raw_predict_batch = _overridden("predict_batch")
        raw_predict_chunk_batch = _overridden("predict_chunk_batch")
        self._raw_predict = cast("PredictFn[ObsT, ActT]", raw_predict)
        self._raw_predict_chunk = raw_predict_chunk
        self._raw_predict_batch = raw_predict_batch
        self._raw_predict_chunk_batch = raw_predict_chunk_batch
        self.spec = resolved_spec
        self._policy = policy
        self._on_reset = on_reset if on_reset is not None else coerced_on_reset
        self._on_close = on_close if on_close is not None else coerced_on_close
        self._on_episode_end = on_episode_end
        self._trust_entrypoints = trust_entrypoints
        self._install_worker(self._on_reset)

    def load(self, **kwargs: Any) -> None:
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

    def predict_chunk(self, observation: ObsT, horizon: int) -> ActT:
        """Optional: map one observation to a CHUNK of actions (leading axis = chunk).

        Override this alongside :meth:`predict` when the policy emits an action
        chunk in one forward pass (ACT, diffusion, flow, VLA action heads). The
        runtime owns the replay: it replays the chunk one action per step without
        re-calling the model. Defining this method *is* the chunk capability — a
        single-action model leaves it unimplemented and the runtime re-plans every
        step.

        ``horizon`` is the runtime-chosen replay horizon: return **up to**
        ``horizon`` actions (the runtime replays them before re-planning). An
        autoregressive head should decode exactly ``horizon`` actions so it does not
        waste decode on a longer natural chunk; a fixed-size head may return more and
        the runtime caps to ``horizon``. The default raises.
        """
        raise NotImplementedError(
            "this model does not emit action chunks; override predict_chunk() to "
            "return a chunk (leading axis = chunk), or use predict() per step."
        )

    def predict_batch(self, observations: list[ObsT]) -> list[ActT]:
        """Optional: map N lane observations to N actions in one batched call.

        Override (alongside :meth:`predict`) to run a single forward pass for a
        vectorized route instead of one call per lane. Receives a list of N
        observations (one per sub-environment) and returns N actions in order. The
        engine prefers this corner for a vectorized route; the default raises and the
        route is driven per-lane through :meth:`predict`.
        """
        raise NotImplementedError(
            "this model does not define a batched predict_batch(); it is driven "
            "one lane at a time through predict()."
        )

    def predict_chunk_batch(self, observations: list[ObsT], horizon: int) -> list[ActT]:
        """Optional: map N lane observations to N action CHUNKS in one batched call.

        The batched counterpart of :meth:`predict_chunk` — one forward for the whole
        vector, returning N chunks (each leading axis = chunk) in order, each up to
        ``horizon`` actions. The engine prefers it for a vectorized chunked route.
        The default raises.
        """
        raise NotImplementedError(
            "this model does not define a batched predict_chunk_batch(); use "
            "predict_chunk() (per lane) or predict_batch()."
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

        # The chunk corner, when the model defines one: identical bridging, but the
        # user returns a chunk (leading axis = chunk) the native engine splits.
        raw_predict_chunk = self._raw_predict_chunk
        predict_chunk_neutral: Callable[[Value, int], Value] | None = None
        if raw_predict_chunk is not None:
            chunk_fn = raw_predict_chunk

            def _predict_chunk_neutral(observation: Value, horizon: int) -> Value:
                return bridge.encode(
                    chunk_fn(cast(ObsT, bridge.decode(observation)), horizon)
                )

            predict_chunk_neutral = _predict_chunk_neutral

        # The batched corners (one forward for the whole vector), when defined: the
        # engine hands a list of N neutral lane inputs; bridge each, run the user's
        # batched predict, then bridge each returned action/chunk back. The chunk
        # corner additionally receives the replay horizon.
        raw_predict_batch = self._raw_predict_batch
        predict_batch_neutral: Callable[[list[Value]], list[Value]] | None = None
        if raw_predict_batch is not None:
            batch_fn = raw_predict_batch

            def _predict_batch_neutral(observations: list[Value]) -> list[Value]:
                framework = [bridge.decode(o) for o in observations]
                actions = cast("Sequence[object]", batch_fn(framework))
                return [bridge.encode(a) for a in actions]

            predict_batch_neutral = _predict_batch_neutral

        raw_predict_chunk_batch = self._raw_predict_chunk_batch
        predict_chunk_batch_neutral: (
            Callable[[list[Value], int], list[Value]] | None
        ) = None
        if raw_predict_chunk_batch is not None:
            chunk_batch_fn = raw_predict_chunk_batch

            def _predict_chunk_batch_neutral(
                observations: list[Value], horizon: int
            ) -> list[Value]:
                framework = [bridge.decode(o) for o in observations]
                chunks = cast("Sequence[object]", chunk_batch_fn(framework, horizon))
                return [bridge.encode(c) for c in chunks]

            predict_chunk_batch_neutral = _predict_chunk_batch_neutral

        self._worker: PyModel = PyModel(
            predict_fn=predict_neutral,
            configure_fn=configure,
            on_reset=on_reset,
            on_episode_end=self._on_episode_end,
            on_close=self._on_close,
            predict_chunk_fn=predict_chunk_neutral,
            predict_batch_fn=predict_batch_neutral,
            predict_chunk_batch_fn=predict_chunk_batch_neutral,
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
        action_horizon: int = 1,
    ) -> RunResult:
        """Drive this model against an env and return a :class:`RunResult`.

        Resolves the adapter from the env's tags and this model's spec, then runs a
        per-episode loop. ``seeds`` gives a per-episode seed and sets the episode
        count unless ``max_episodes`` is given; ``instruction``, when given,
        overrides *every* text input the spec declares on each step -- at its
        placement in the input tree (bare-root, top-level, or nested) and in that
        input's declared shape (a bare ``str``, or ``[instruction]`` for a
        ``container='list'`` text input). ``env_or_address`` is an env object
        exposing ``reset``/``step`` (e.g. a ``RemoteEnv``), an
        :class:`~rlmesh.EnvFactory` (built and tag-stamped, then driven locally), an
        object with an ``address``, or a bare address string the loop dials.

        ``action_horizon`` (> 1) replays an action chunk one step per env step,
        re-planning every ``action_horizon`` steps — only when this model defines
        :meth:`predict_chunk`; otherwise it runs un-chunked.
        """
        return self.session(
            env_or_address,
            instruction=instruction,
            close_env=close_env,
            token=token,
            action_horizon=action_horizon,
        ).run(seeds=seeds, max_episodes=max_episodes)

    def session(
        self,
        env_or_address: object,
        *,
        instruction: str | None = None,
        close_env: bool = False,
        token: str = "",
        trust_entrypoints: bool | None = None,
        action_horizon: int = 1,
    ) -> Session[ObsT, ActT]:
        """Bind this model to an env and return a :class:`Session` to drive by hand.

        The manual counterpart of :meth:`run`: drive ``reset`` / ``predict`` / ``step``
        yourself, or call :meth:`Session.run` to pump whole episodes. ``env_or_address``
        is an env object, an :class:`~rlmesh.EnvFactory`, a remote-env handle, or an
        address string (see :meth:`run`).
        ``action_horizon`` (> 1) replays an action chunk one step per env step when
        this model defines :meth:`predict_chunk` (see :meth:`run`).
        """
        from ._eval import Session

        return Session(
            predict=self._raw_predict,
            predict_chunk=self._raw_predict_chunk,
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
            action_horizon=action_horizon,
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


def as_model(model: object) -> ModelBase[Any, Any]:
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
    is a local env, an :class:`~rlmesh.EnvFactory`, a remote-env handle, or an address
    string. A spec'd model resolves its adapter from the env's tags -- a local env must
    carry them (via :func:`rlmesh.adapters.tag` or an :class:`~rlmesh.EnvFactory`).

    Pass :data:`rlmesh.RANDOM_SAMPLE` as ``model`` for a random baseline: each step
    samples the env's action space, no spec or adapter involved.
    """
    from ._adapter_mode import NO_ADAPTER
    from ._eval import RANDOM_SAMPLE, Session

    if model is RANDOM_SAMPLE:
        # A random baseline samples the env's action space and ignores observations,
        # so it adapts nothing -- skip adapter resolution even on a tagged env.
        return cast(
            "Session[Any, Any]",
            Session(
                env=env,
                # RANDOM_SAMPLE is a private sentinel Session special-cases by
                # identity; cast keeps it out of the public `predict` signature.
                predict=cast("Any", RANDOM_SAMPLE),
                spec=NO_ADAPTER,
                instruction=instruction,
                close_env=close_env,
                token=token,
                trust_entrypoints=bool(trust_entrypoints),
            ),
        )
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
    return as_model(model).session(
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
