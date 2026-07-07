"""Shared Python model wrapper: construct from any source, eval against an env, serve."""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)

from .._value_conversion import ValueBridge, identity_bridge, tree_map
from ..types import EnvTarget, LocalEnvTarget, Value, ViewArg

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModel, ServeOptions
    from rlmesh.params import ParamSpec

    from ._eval import RunHooks, RunResult, Session

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
LifecycleCallback = Callable[[], None]
PredictFn = Callable[[ObsT], ActT]
Corner = Callable[..., object]

#: The four predict corners, ordered general -> specific. A model overrides any
#: subset; :func:`_synthesize_corners` derives the rest.
_CORNERS = ("predict", "predict_chunk", "predict_batch", "predict_chunk_batch")


def _served_handle(model: object) -> bool:
    """Whether ``model`` is a served handle (``RemoteModel`` / ``SandboxModel``).

    A served handle binds itself via its own ``.session``; a local
    :class:`ModelBase` (or a class / bare callable) is normalized through
    :func:`as_model` instead.
    """
    return (
        callable(getattr(model, "session", None))
        and not isinstance(model, type)
        and not isinstance(model, ModelBase)
    )


def _debatch(bridge: ValueBridge, batched_fn: Corner) -> Corner:
    """Single-lane corner from a batched one: run a batch of one and unwrap.

    Wraps the lone observation into a 1-lane batch (``tree_stack``), calls the
    batched corner, and peels lane 0 back off (``tree_unstack``). Trailing args --
    positional or keyword -- pass straight through, so this turns ``predict_batch``
    into ``predict`` and ``predict_chunk_batch`` into ``predict_chunk`` (whose
    ``horizon`` the local Session passes by keyword).
    """

    def derived(observation: object, *rest: object, **rest_kw: object) -> object:
        fused = bridge.tree_stack([observation])
        return bridge.tree_unstack(batched_fn(fused, *rest, **rest_kw), 1)[0]

    return derived


def _first_frame(chunk: object) -> object:
    """The first action of a single chunk, matching the engine's ``split_chunk``.

    A ``Mapping`` (Dict-space action) carries the chunk axis INSIDE each leaf, so
    recurse per value; a ``list``/``tuple`` is itself the frame axis; an array's
    leading axis is the frame axis; a scalar/text is a single-frame chunk and stands
    as its own action. Keeps local de-chunking byte-consistent with the served path
    (``stateful.rs::split_chunk``).
    """
    if isinstance(chunk, Mapping):
        items = cast("Mapping[Any, Any]", chunk)
        return {k: _first_frame(v) for k, v in items.items()}
    if isinstance(chunk, (list, tuple)):
        seq = cast("Any", chunk)
        if len(seq) == 0:
            raise ValueError(
                "predict_chunk returned an empty chunk; return at least one action"
            )
        return seq[0]
    if getattr(chunk, "ndim", 0) >= 1:
        arr = cast("Any", chunk)
        if arr.shape[0] == 0:
            raise ValueError(
                "predict_chunk returned an empty chunk; return at least one action"
            )
        return arr[0]
    return chunk


def _dechunk(chunk_fn: Corner, *, batched: bool) -> Corner:
    """Single-action corner from a chunk one: run horizon 1 and take the first action.

    Single: take the first frame (``_first_frame``, ``split_chunk`` semantics).
    Batched: the chunk leaves are ``[B, horizon, ...]``, so take frame 0 past the
    batch axis (``leaf[:, 0]``), leaving non-array leaves (text) untouched.
    """

    def derived(observation: object) -> object:
        chunk = chunk_fn(observation, 1)
        if not batched:
            return _first_frame(chunk)
        return tree_map(
            chunk,
            lambda leaf: cast("Any", leaf)[:, 0]
            if getattr(leaf, "ndim", 0) >= 2
            else leaf,
        )

    return derived


def _horizon_mode(fn: Corner) -> Literal["positional", "keyword"] | None:
    """How a chunk corner declared it receives the execution horizon, if at all.

    A policy returns its *native* chunk and usually ignores the horizon -- its length
    is fixed by the trained weights. An autoregressive decoder that can stop early
    declares an optional parameter (``execution_horizon``), either as a second
    positional parameter (``"positional"``) or keyword-only (``"keyword"``);
    detecting that once here keeps the common corner a clean ``predict_chunk(obs)``
    while the rare one is handed how many actions the runtime will execute.
    ``None`` means the corner takes no horizon.
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    positional = sum(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params
    )
    if positional >= 2 or any(p.kind == p.VAR_POSITIONAL for p in params):
        return "positional"
    if any(p.kind == p.KEYWORD_ONLY and p.name == "execution_horizon" for p in params):
        return "keyword"
    return None


def _normalize_chunk(fn: Corner | None) -> Corner | None:
    """Adapt a chunk corner to the internal ``(obs, horizon)`` contract.

    Every internal caller -- :func:`_synthesize_corners`, :func:`_dechunk`, the native
    neutral wrappers, the local replay -- passes ``(obs, horizon)`` positionally. A
    corner that didn't ask for the horizon is wrapped to swallow it, and one that
    declared a keyword-only ``execution_horizon`` is called by keyword, so the rest
    of the pipeline is uniform and the author writes whichever signature their
    model needs.
    """
    if fn is None:
        return None
    mode = _horizon_mode(fn)
    if mode == "positional":
        return fn
    if mode == "keyword":
        chunk_fn = fn

        def by_keyword(observation: object, execution_horizon: object) -> object:
            return chunk_fn(observation, execution_horizon=execution_horizon)

        return by_keyword

    def absorbing(observation: object, _execution_horizon: object) -> object:
        return fn(observation)

    return absorbing


def _synthesize_corners(
    bridge: ValueBridge,
    predict: Corner | None,
    predict_chunk: Corner | None,
    predict_batch: Corner | None,
    predict_chunk_batch: Corner | None,
) -> tuple[Corner | None, Corner | None, Corner | None, Corner | None]:
    """Fill missing predict corners by deriving downward from the most general one.

    The corners form a 2x2 lattice over a batch axis and a chunk axis. A less
    general corner always derives from a more general one: *debatch* drops the
    batch axis, *de-chunk* (horizon 1, take the first action) drops the chunk axis.
    Going *up* the chunk axis is impossible -- chunking is a model capability, not
    glue -- and going up the batch axis is left to the engine's per-lane loop. So a
    model that defines ``predict_chunk_batch`` gets all four for free, and
    ``predict()`` is available unless only a chunk corner exists on the raw Value
    bridge (de-chunk needs array leaves; surfaced as a clear error by the caller).

    The ambiguous ``predict`` (derivable from either ``predict_batch`` or
    ``predict_chunk``) prefers the un-chunked ``predict_batch`` so a single-step
    call stays un-chunked instead of paying for a chunk decode it would discard.
    """
    p, pc, pb, pcb = predict, predict_chunk, predict_batch, predict_chunk_batch
    can_dechunk = bridge.name != identity_bridge.name

    if pcb is not None:
        if pc is None:
            pc = _debatch(bridge, pcb)
        if pb is None and can_dechunk:
            pb = _dechunk(pcb, batched=True)
    if p is None:
        if pb is not None:
            p = _debatch(bridge, pb)
        elif pc is not None and can_dechunk:
            p = _dechunk(pc, batched=False)
    return p, pc, pb, pcb


# Suppresses the ``__init__`` auto-load on the served-construction path so the
# bootstrap is authoritative: ``construct_authored_model`` builds the worker
# without loading, then runs ``load(**binding)`` once with the resolved params.
# Local-dev construction (the contextvar default ``False``) keeps the eager
# auto-load. See the model load() seam in the declared-params design.
_suppress_autoload: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "rlmesh_suppress_model_autoload", default=False
)


@contextlib.contextmanager
def suppress_autoload() -> Iterator[None]:
    """Suppress ``ModelBase``'s ``__init__`` auto-load within the block.

    The served-construction seam: build the worker without loading weights, then
    apply the resolved binding via ``load(**binding)`` once (see
    :func:`rlmesh._bootstrap.loaders.construct_authored_model`).
    """
    token = _suppress_autoload.set(True)
    try:
        yield
    finally:
        _suppress_autoload.reset(token)


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
        on_episode_end / on_close: Optional lifecycle callbacks (they override a
            subclass's ``reset``/``close`` methods, and a wrapped policy's). There is
            no episode-*begin* hook: per-episode state is lazy-seeded on first
            predict, so a stateful model clears its state at episode *end* via
            ``on_episode_end`` (a subclass's ``reset()`` is wired here). This fires
            identically on the local ``run(env)``/``session`` loop and the served
            wire path (driven there by the explicit ``ResetAdapter``), so local and
            served behave the same.
        trust_entrypoints: Allow ``module:callable`` custom-input entrypoints in a
            spec to be imported during adapter resolution.

    Example::

        from rlmesh.numpy import Model

        result = Model(lambda observation: 0).run("127.0.0.1:5555", seeds=[0])
        print(result.mean_reward)
    """

    _bridge: ClassVar[ValueBridge]
    #: Framework remote-env client used when ``run`` is given a bare address.
    _remote_env_cls: ClassVar[type | None] = None
    #: The model's content: a ``ModelSpec``, ``NO_ADAPTER``, or ``None``. Set it as a
    #: class attribute when subclassing; the ``spec=`` kwarg overrides it per instance.
    spec: object | None = None
    #: Optional declared construction-parameter surface for ``load`` -- the model
    #: counterpart of :attr:`rlmesh.EnvFactory.params`, presented/swept the same way
    #: (see :mod:`rlmesh.params`). Advisory today: a dashboard reads it via
    #: ``rlmesh.describe``; binding it into ``load`` is gated on the served-load seam.
    params: ClassVar[ParamSpec | None] = None
    #: Compute device for model inputs (torch/jax only). Set it in :meth:`load`
    #: alongside moving your weights -- one source of truth -- and rlmesh moves every
    #: obs tensor leaf onto it before :meth:`predict`, so you never call ``.to(device)``
    #: yourself. ``None`` (the default) leaves obs as decoded; ignored by the numpy /
    #: raw-Value model (no device concept).
    device: object | None = None
    #: Whether the runtime may FUSE this model's batched corner across routes: a
    #: grouped predict then concatenates lanes from *different* envs into one
    #: :meth:`predict_batch` / :meth:`predict_chunk_batch` call (one forward for
    #: the whole group), so the batch size varies with what the control plane
    #: grouped. On (the default) suits any row-independent forward. Set ``False``
    #: when your batched forward assumes same-route, fixed-size batches -- a
    #: shape-pinned jit/compile trace, per-batch statistics, batch-position
    #: logic -- and grouped predicts will run per route instead.
    allow_fusion: bool = True

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Return this model's full metadata envelope (see :func:`rlmesh.describe`)."""
        from .._describe import describe  # lazy: avoid an import cycle at module load

        return describe(cls, kind="model")

    @overload
    def __init__(
        self,
        source: type[Any],
        *,
        spec: object | None = None,
        on_episode_end: LifecycleCallback | None = None,
        on_close: LifecycleCallback | None = None,
        trust_entrypoints: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self,
        source: PredictFn[ObsT, ActT],
        *,
        spec: object | None = None,
        on_episode_end: LifecycleCallback | None = None,
        on_close: LifecycleCallback | None = None,
        trust_entrypoints: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self,
        source: object | None = None,
        *,
        spec: object | None = None,
        on_episode_end: LifecycleCallback | None = None,
        on_close: LifecycleCallback | None = None,
        trust_entrypoints: bool = False,
    ) -> None: ...
    def __init__(
        self,
        source: Callable[..., object] | object | None = None,
        *,
        spec: object | None = None,
        on_episode_end: LifecycleCallback | None = None,
        on_close: LifecycleCallback | None = None,
        trust_entrypoints: bool = False,
    ) -> None:
        """Build the model from ``source`` (see the class docstring for the modes).

        The overloads are the typed face of the permissive runtime contract: a
        *predict callable* flows its annotated observation/action types onto the
        model's generic parameters (``Model(predict)`` with
        ``def predict(obs: X) -> Y`` is a ``Model[X, Y]``), while a class source
        (instantiated by :func:`coerce_model`), a duck-typed policy object, or an
        unannotated callable falls back to the framework's default value types.
        """
        self._bridge.ensure_available()

        # A Model subclass overrides at least one predict corner (plus optionally
        # load/reset/close/spec); the corners it leaves out are derived below, so
        # overriding predict_chunk_batch alone is enough. A wrapped callable
        # supplies a single predict.
        def _overridden(method_name: str) -> Corner | None:
            if getattr(type(self), method_name) is getattr(ModelBase, method_name):
                return None
            return cast("Corner", getattr(self, method_name))

        corners = {name: _overridden(name) for name in _CORNERS}

        if source is None and any(fn is not None for fn in corners.values()):
            # Subclass-authoring mode. Load weights once, before the worker is built.
            # The served-construction path suppresses this eager load and calls
            # load(**binding) itself once the params are resolved.
            if not _suppress_autoload.get():
                self.load()
            resolved_spec = spec if spec is not None else type(self).spec
            # A subclass's reset() is wired to the episode-END edge (on_episode_end),
            # the only per-episode boundary both local and served paths signal.
            coerced_on_episode_end: LifecycleCallback | None = self.reset
            coerced_on_close: LifecycleCallback | None = self.close
            policy: object = self
            raw_predict: Corner | None = corners["predict"]
            raw_predict_chunk = corners["predict_chunk"]
            raw_predict_batch = corners["predict_batch"]
            raw_predict_chunk_batch = corners["predict_chunk_batch"]
        elif source is None:
            raise TypeError(
                "Model() needs a predict callable, e.g. Model(lambda obs: ...), or "
                "subclass Model and override predict() (or predict_chunk / "
                "predict_batch / predict_chunk_batch)."
            )
        else:
            # A Model builds its own worker; wrapping one as another model's source
            # would double-construct. Reject it here -- the single construction
            # gateway -- so the coercion helper stays free of a back-import to base.
            if isinstance(source, ModelBase) or (
                isinstance(source, type) and issubclass(source, ModelBase)
            ):
                raise TypeError(
                    "Model received a Model as its source. Instantiate your Model "
                    "subclass directly (it builds its own worker), or pass a predict "
                    "callable."
                )
            from ._coerce import coerce_model

            coerced = coerce_model(source, spec=spec)
            raw_predict = coerced.predict
            raw_predict_chunk = coerced.predict_chunk
            raw_predict_batch = coerced.predict_batch
            raw_predict_chunk_batch = coerced.predict_chunk_batch
            resolved_spec = coerced.spec
            coerced_on_episode_end = coerced.on_episode_end
            coerced_on_close = coerced.on_close
            policy = coerced.policy

        # Normalize the chunk corners to the internal (obs, horizon) contract before
        # synthesis: a corner that declared no execution_horizon param is wrapped to
        # swallow it, so the author writes predict_chunk(obs) while an autoregressive
        # decoder may add `execution_horizon: int = 1` to decode exactly that many.
        raw_predict_chunk = _normalize_chunk(raw_predict_chunk)
        raw_predict_chunk_batch = _normalize_chunk(raw_predict_chunk_batch)

        # Climb the corner lattice: derive the corners the model didn't define from
        # the most general one it did, so predict_chunk_batch alone yields all four
        # on a framework bridge. (A chunk-only model on the raw Value bridge can't
        # de-chunk, so predict() stays unavailable and the guard below raises.)
        raw_predict, raw_predict_chunk, raw_predict_batch, raw_predict_chunk_batch = (
            _synthesize_corners(
                self._bridge,
                raw_predict,
                raw_predict_chunk,
                raw_predict_batch,
                raw_predict_chunk_batch,
            )
        )
        if raw_predict is None:
            raise TypeError(
                "could not derive predict(): define predict(), or a corner it can be "
                "derived from (a chunk-only model on the raw Value bridge needs an "
                "explicit predict())."
            )

        self._raw_predict = cast("PredictFn[ObsT, ActT]", raw_predict)
        self._raw_predict_chunk = raw_predict_chunk
        self._raw_predict_batch = raw_predict_batch
        self._raw_predict_chunk_batch = raw_predict_chunk_batch
        self.spec = resolved_spec
        self._policy = policy
        self._on_close = on_close if on_close is not None else coerced_on_close
        self._on_episode_end = (
            on_episode_end if on_episode_end is not None else coerced_on_episode_end
        )
        self._trust_entrypoints = trust_entrypoints
        self._worker: PyModel | None = None

    def _to_device(self, value: object) -> object:
        """Move every framework tensor leaf of an input onto :attr:`device`.

        A no-op when ``device`` is None or this model's framework has no device
        (numpy / raw Value); non-tensor leaves pass through. Read at predict time so
        a ``device`` set in :meth:`load` (which runs after the served worker is built)
        is honored.
        """
        return self._bridge.to_device(value, self.device)

    def _require_device_support(self) -> None:
        """Reject a ``device`` set on a framework with no device (mirrors EnvServer)."""
        if self.device is not None and not self._bridge.supports_device():
            raise ValueError(
                "device=... requires a torch/jax model; this model's framework "
                "(numpy / raw Value) has no device."
            )

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

    def predict_chunk(self, observation: ObsT) -> ActT:
        """Optional: map one observation to a CHUNK of actions (leading axis = chunk).

        Override this alongside :meth:`predict` when the policy emits an action
        chunk in one forward pass (ACT, diffusion, flow, VLA action heads). Return
        your model's **native** chunk; the runtime owns the replay -- it executes the
        first ``execution_horizon`` actions one per step without re-calling the model,
        then re-plans. Defining this method *is* the chunk capability: a single-action
        model leaves it unimplemented and the runtime re-plans every step.

        Most policies ignore the execution horizon -- their chunk length is fixed by
        the trained weights, and the runtime simply uses a prefix of the native chunk.
        An autoregressive decoder that can stop early may add an optional
        ``execution_horizon: int = 1`` parameter -- as a second positional parameter
        or keyword-only (keep the default, so it stays a compatible override of this
        one-arg base); the runtime fills it with how many actions it will execute,
        so the head can decode exactly that many instead of its full natural length.
        The default raises.
        """
        raise NotImplementedError(
            "this model does not emit action chunks; override predict_chunk() to "
            "return a chunk (leading axis = chunk), or use predict() per step."
        )

    def predict_batch(self, observations: ObsT) -> ActT:
        """Optional: one batched forward over all N vectorized lanes.

        Override (alongside :meth:`predict`) to run a single forward pass for a
        vectorized route instead of one call per lane. The runtime fuses the N
        per-lane observations into one *batched* observation -- every leaf gains a
        leading batch axis, so a Dict observation arrives as ``{key: array[N, ...]}``
        (a NumPy/Torch/JAX array per leaf, stacked for this model's framework), not a
        list of N dicts. Return the batched action the same way: one value whose
        leaves carry the leading batch axis (e.g. ``array[N, action_dim]``); the
        runtime splits it back per lane. The engine prefers this corner for a
        vectorized route; the default raises and the route is driven per-lane through
        :meth:`predict`.

        (The dependency-free ``rlmesh.Model`` over raw ``Value`` trees can't fuse
        opaque tensors, so it instead receives the per-lane ``list`` and returns one.)
        """
        raise NotImplementedError(
            "this model does not define a batched predict_batch(); it is driven "
            "one lane at a time through predict()."
        )

    def predict_chunk_batch(self, observations: ObsT) -> ActT:
        """Optional: one batched forward returning an action CHUNK per lane.

        The batched counterpart of :meth:`predict_chunk`: receives the fused batched
        observation (leaves ``[N, ...]``; see :meth:`predict_batch`) and returns the
        batched native chunk -- leaves ``[N, chunk, ...]`` (batch axis first, then the
        per-lane chunk axis). The runtime splits the batch axis back per lane, executes
        a prefix (``execution_horizon``) of each, then re-plans. The engine prefers it
        for a vectorized chunked route. Like :meth:`predict_chunk`, an autoregressive
        decoder may add an optional ``execution_horizon: int = 1`` second parameter to
        decode exactly that many. The default raises.
        """
        raise NotImplementedError(
            "this model does not define a batched predict_chunk_batch(); use "
            "predict_chunk() (per lane) or predict_batch()."
        )

    def reset(self) -> None:
        """Optional: called at each episode boundary (no-op by default)."""

    def close(self) -> None:
        """Optional: release resources at the end of a run (no-op by default)."""

    def _install_worker(self) -> PyModel:
        """Build (once) and return the native model worker (the serve path).

        A spec'd model's adapter resolves per env at ``resolve_adapter`` (the
        served endpoint receives the env contract there): ``configure`` resolves
        it from the env's contract and the native worker applies it around the
        raw predict. A spec-less / ``NO_ADAPTER`` model serves its own predict.

        Built lazily on the first serve-path call, not in ``__init__``: a model
        driven locally (``run()`` / ``session()`` against an in-process loop)
        never needs the native worker, and the worker's callbacks close over
        ``self`` -- a reference cycle through a native class the Python cycle
        collector cannot free -- so an unserved model should not pay for one.
        """
        if self._worker is not None:
            return self._worker
        from .._load_native import load_native
        from ._eval import resolve_adapter

        spec = self.spec
        bridge = self._bridge
        raw_predict = self._raw_predict

        def configure(env_contract: object) -> object:
            """Resolve the served route: a native plan plus neutral host holes.

            The Rust engine drives the result (it owns the per-episode frame
            buffers and adapter application); a spec-less / ``NO_ADAPTER``
            model returns ``None``. Trust is read at resolve time (not
            captured at worker build), so a ``run()``'s per-call
            ``trust_entrypoints`` override stays scoped to that run even
            though the worker itself is cached.
            """
            adapter = resolve_adapter(
                spec,
                cast("Any", env_contract),
                trust_entrypoints=self._trust_entrypoints,
            )
            return adapter.serve_route(bridge) if adapter is not None else None

        def predict_neutral(observation: Value) -> Value:
            # The engine has already applied the adapter (declarative transform,
            # frame-stacking, customs, enc-shims) in Rust, so the obs handed here
            # is the final model input; bridge it into the framework, run the user
            # predict, and bridge the action back. A spec-less route hands the raw
            # observation through the identical path (no adapter).
            return bridge.encode(
                raw_predict(cast(ObsT, self._to_device(bridge.decode(observation))))
            )

        # The chunk corner, when the model defines one: identical bridging, but the
        # user returns a chunk (leading axis = chunk) the native engine splits.
        raw_predict_chunk = self._raw_predict_chunk
        predict_chunk_neutral: Callable[[Value, int], Value] | None = None
        if raw_predict_chunk is not None:
            chunk_fn = raw_predict_chunk

            def _predict_chunk_neutral(observation: Value, horizon: int) -> Value:
                return bridge.encode(
                    chunk_fn(
                        cast(ObsT, self._to_device(bridge.decode(observation))), horizon
                    )
                )

            predict_chunk_neutral = _predict_chunk_neutral

        # The batched corners (one forward for the whole vector), when defined: the
        # engine hands a list of N neutral lane inputs. Fuse them into ONE batched
        # value (every leaf gains a leading batch axis -- a Dict obs becomes
        # {key: array[N, ...]}, the shape every RL/VLA runtime hands a policy), run
        # the user's single batched forward, then split the batched action/chunk
        # back into N per-lane values the engine replays. The native (Value) bridge
        # can't fuse opaque tensors, so it passes the per-lane list through.
        raw_predict_batch = self._raw_predict_batch
        predict_batch_neutral: Callable[[list[Value]], list[Value]] | None = None
        if raw_predict_batch is not None:
            batch_fn = raw_predict_batch

            def _predict_batch_neutral(observations: list[Value]) -> list[Value]:
                if not observations:
                    return []
                fused = bridge.tree_stack(
                    [self._to_device(bridge.decode(o)) for o in observations]
                )
                actions = batch_fn(fused)
                parts = bridge.tree_unstack(actions, len(observations))
                return [bridge.encode(a) for a in parts]

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
                if not observations:
                    return []
                fused = bridge.tree_stack(
                    [self._to_device(bridge.decode(o)) for o in observations]
                )
                chunks = chunk_batch_fn(fused, horizon)
                # Split the batch axis only; each lane's chunk (horizon) axis stays.
                parts = bridge.tree_unstack(chunks, len(observations))
                return [bridge.encode(c) for c in parts]

            predict_chunk_batch_neutral = _predict_chunk_batch_neutral

        worker: PyModel = load_native("PyModel")(
            predict_fn=predict_neutral,
            configure_fn=configure,
            on_episode_end=self._on_episode_end,
            on_close=self._on_close,
            predict_chunk_fn=predict_chunk_neutral,
            predict_batch_fn=predict_batch_neutral,
            predict_chunk_batch_fn=predict_chunk_batch_neutral,
            allow_fusion=self.allow_fusion,
        )
        self._worker = worker
        return worker

    def run(
        self,
        env_or_address: LocalEnvTarget,
        *,
        seeds: Sequence[int] | None = None,
        max_episodes: int | None = None,
        max_episode_steps: int | None = None,
        max_episode_seconds: float | None = None,
        hooks: RunHooks | None = None,
        instruction: str | None = None,
        close_env: bool = False,
        trust_entrypoints: bool | None = None,
        execution_horizon: int = 1,
        view: ViewArg = None,
    ) -> RunResult:
        """Drive this model against an env on the native runtime loop.

        One loop for every env shape: a single env and a vectorized
        (``num_envs > 1``) one run through the same native runtime the served
        path uses -- the route resolves at connect (adapter, per-episode frame
        buffers), the engine dispatches the most specific predict corner
        (:meth:`predict_batch` / :meth:`predict_chunk_batch` for a vectorized
        route), and ``execution_horizon`` (> 1) executes that many actions of
        each predicted chunk before re-planning (needs :meth:`predict_chunk`).

        ``env_or_address`` is a bare address string the loop dials, an object
        with an ``address`` (:class:`~rlmesh.EnvServer`, ``RemoteEnv`` /
        ``RemoteVectorEnv``), or a local env object -- served on a loopback
        port for the duration of the run (tag it via
        :func:`rlmesh.adapters.tag` for a spec'd model).

        ``seeds`` gives a per-episode reset seed and sets the episode count
        unless ``max_episodes`` is given (both default to one episode; an empty
        ``seeds`` returns an empty result). ``max_episode_steps`` /
        ``max_episode_seconds`` truncate an episode at a step / wall-clock cap,
        and a non-terminating env is bounded regardless: without an explicit
        cap the runtime truncates any episode at 100,000 steps (the same
        built-in bound as :meth:`Session.run <rlmesh.Session.run>`). Explicit
        seeds and caps need the runtime to own resets, so they reject an
        autoresetting vector env (drive it with ``max_episodes``; seeds on a
        driver-reset vector env must be a multiple of ``num_envs``). For a
        vectorized env ``max_episodes`` is a lower bound reached in lane
        batches (episodes complete interleaved). A per-call
        ``trust_entrypoints`` override applies to this run only. On the
        result, each episode's ``predict_ms`` / ``step_ms`` carry the run's
        session-mean op latencies (the runtime aggregates timing per op, not
        per episode).

        The Session-only knobs -- ``hooks``, ``instruction``, ``view`` -- are
        not part of this loop; use :meth:`session` and
        :meth:`Session.run <rlmesh.Session.run>` for step-level observation,
        instruction override, or a live viewer.
        """
        rejected = [
            name
            for name, given in (
                ("hooks", hooks is not None),
                ("instruction", instruction is not None),
                ("view", view is not None),
            )
            if given
        ]
        if rejected:
            raise ValueError(
                f"run() drives the native runtime loop, which does not support "
                f"{', '.join(rejected)}; use session().run(...) for these."
            )
        if execution_horizon < 1:
            raise ValueError(f"execution_horizon must be >= 1, got {execution_horizon}")
        self._require_device_support()
        if execution_horizon > 1 and self._raw_predict_chunk is None:
            warnings.warn(
                f"execution_horizon={execution_horizon} was requested but "
                "the model defines no predict_chunk(); running un-chunked.",
                stacklevel=2,
            )
        if max_episodes is None:
            max_episodes = len(seeds) if seeds is not None else 1
        if max_episodes == 0:
            from ._eval import RunResult

            return RunResult()

        previous_trust = self._trust_entrypoints
        if trust_entrypoints is not None:
            self._trust_entrypoints = trust_entrypoints
        try:
            episodes = self._run_native(
                env_or_address,
                max_episodes=max_episodes,
                seeds=seeds,
                max_episode_steps=max_episode_steps,
                max_episode_seconds=max_episode_seconds,
                close_env=close_env,
                execution_horizon=execution_horizon,
            )
        finally:
            self._trust_entrypoints = previous_trust
        from ._eval import EpisodeResult, RunResult

        return RunResult(
            episodes=tuple(
                EpisodeResult(
                    index=episode["index"],
                    seed=episode["seed"],
                    steps=episode["steps"],
                    reward=episode["reward"],
                    terminated=episode["terminated"],
                    truncated=episode["truncated"],
                    success=episode["success"],
                    duration_s=episode["duration_s"],
                    predict_ms=episode["predict_ms"],
                    step_ms=episode["step_ms"],
                )
                for episode in episodes
            )
        )

    def _run_native(
        self,
        env_or_address: object,
        *,
        max_episodes: int,
        seeds: Sequence[int] | None,
        max_episode_steps: int | None,
        max_episode_seconds: float | None,
        close_env: bool,
        execution_horizon: int,
    ) -> list[dict[str, Any]]:
        """Normalize the env target, drive the native loop, return the episodes.

        Target handling follows :func:`~rlmesh._models._connect.classify_env_target`
        (the same precedence ``session()`` uses): a bare address is dialed; a
        remote handle is dialed by its own address (it cannot be re-served); a
        local env or a built :class:`~rlmesh.EnvFactory` env is served on a
        loopback port for the duration of the run. ``close_env`` maps per kind:
        the wire close op for a bare address, the handle's ``shutdown``/``close``
        for a handle, and the env object's ``close()`` for a locally served env.

        A local single env's spec/tags pairing is pre-flighted in Python so a
        mismatch surfaces as the typed ``AdapterResolutionError`` rather than
        stringified out of the native route resolve (a vector env's local spaces
        are batched while tags describe one lane, so its check stays at the
        served resolve).
        """
        from ._connect import classify_env_target

        kind, payload = classify_env_target(env_or_address)
        env_obj: object = payload
        address: str | None = None
        server = None
        if kind == "address":
            address = cast(str, payload)
        elif kind == "handle":
            handle_address = getattr(payload, "address", None)
            if not isinstance(handle_address, str):
                raise TypeError(
                    "this env handle exposes env_contract but no dialable "
                    "address; pass the env's address string instead"
                )
            address = handle_address
        elif kind == "factory":
            from ._connect import factory_env

            env_obj = factory_env(payload)
        if address is None:
            from .._server import EnvServer, VectorServerEnvLike

            if not hasattr(env_obj, "single_observation_space"):
                from ._connect import local_contract
                from ._resolve import resolve_adapter

                resolve_adapter(
                    self.spec,
                    local_contract(env_obj),
                    trust_entrypoints=self._trust_entrypoints,
                )
            server = EnvServer(cast("VectorServerEnvLike", env_obj), "127.0.0.1:0")
            server.start()
            address = server.address
        try:
            return self._run_local_for_episodes(
                address,
                max_episodes=max_episodes,
                execution_horizon=execution_horizon,
                seeds=list(seeds) if seeds is not None else None,
                max_episode_steps=max_episode_steps,
                max_episode_seconds=max_episode_seconds,
                close_env=close_env and kind == "address",
            )
        except RuntimeError as error:
            if "active Join session" in str(error):
                raise RuntimeError(
                    "the target env already has an active session: a RemoteEnv / "
                    "RemoteVectorEnv handle that was driven with reset()/step() "
                    "holds the env's single session slot. close() the handle (or "
                    "its open session) before run(), or drive it interactively "
                    "via session()."
                ) from error
            raise
        finally:
            if server is not None:
                server.shutdown()
                if close_env:
                    close = getattr(env_obj, "close", None)
                    if callable(close):
                        close()
            elif close_env and kind == "handle":
                from ._connect import shutdown_env

                shutdown_env(env_obj)

    def session(
        self,
        env_or_address: LocalEnvTarget,
        *,
        instruction: str | None = None,
        close_env: bool = False,
        trust_entrypoints: bool | None = None,
        execution_horizon: int = 1,
        view: ViewArg = None,
    ) -> Session[ObsT, ActT]:
        """Bind this model to an env and return a :class:`Session` to drive by hand.

        The manual counterpart of :meth:`run`: drive ``reset`` / ``predict`` / ``step``
        yourself, or call :meth:`Session.run` to pump whole episodes -- as many times
        as you like; the caller-held session (its connection, viewer, and adapter
        state) stays open until you close it (``close()`` or the ``with`` block).
        ``env_or_address`` is an env object, an :class:`~rlmesh.EnvFactory`, a
        remote-env handle, or an address string (see :meth:`run`).
        ``execution_horizon`` (> 1) executes that many actions per predicted chunk, one
        per env step, when this model defines :meth:`predict_chunk` (see :meth:`run`).
        """
        from ._eval import Session

        self._require_device_support()
        if execution_horizon > 1 and self._raw_predict_chunk is None:
            warnings.warn(
                f"execution_horizon={execution_horizon} was requested but "
                "the model defines no predict_chunk(); running un-chunked.",
                stacklevel=2,
            )
        return Session._create(  # pyright: ignore[reportPrivateUsage]
            predict=self._raw_predict,
            predict_chunk=self._raw_predict_chunk,
            spec=self.spec,
            env=env_or_address,
            device=self.device,
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
            execution_horizon=execution_horizon,
            view=view,
        )

    def serve(
        self,
        address: str,
        *,
        options: ServeOptions | None = None,
    ) -> None:
        """Host this model as an endpoint (blocking).

        A spec'd model resolves its adapter per env from the env contract the
        ``resolve_adapter`` handshake delivers, then applies it around predict; a
        spec-less / ``NO_ADAPTER`` model serves its own predict directly.
        """
        self._require_device_support()
        self._install_worker().serve(address, options)

    def _run_local(
        self, env_address: str, *, execution_horizon: int = 1
    ) -> list[dict[str, Any]]:
        """Native worker loop against a remote env, until the env ends.

        Returns the runtime report's per-episode summaries (one dict per
        completed episode, ``EpisodeResult``-shaped keys). Telemetry is
        surfaced on the serving runtime via its ``on_telemetry`` hook, not
        returned here. Drives vectorized (``num_envs > 1``) envs through the
        native engine: the route resolves at connect (adapter + batched
        predict corners), and ``execution_horizon`` (> 1) enables action
        chunking exactly as the served path's ``ResolveAdapter`` pin does.
        """
        return self._install_worker().run_local(env_address, execution_horizon)

    def _run_local_for_episodes(
        self,
        env_address: str,
        *,
        max_episodes: int,
        execution_horizon: int = 1,
        seeds: list[int] | None = None,
        max_episode_steps: int | None = None,
        max_episode_seconds: float | None = None,
        close_env: bool = False,
    ) -> list[dict[str, Any]]:
        """Native worker loop against a remote env for a fixed episode count.

        Returns the per-episode summaries; see :meth:`_run_local` for the
        shape, where telemetry is surfaced, and what ``execution_horizon``
        does. ``seeds`` / the episode caps mirror :meth:`run` (explicit seeds
        and caps need runtime-owned resets, i.e. autoreset disabled).
        """
        return self._install_worker().run_local_for_episodes(
            env_address,
            max_episodes,
            execution_horizon,
            seeds,
            max_episode_steps,
            max_episode_seconds,
            close_env,
        )

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


@overload
def session(
    model: ModelBase[ObsT, ActT],
    env: LocalEnvTarget,
    *,
    instruction: str | None = None,
    close_env: bool = False,
    trust_entrypoints: bool | None = None,
    execution_horizon: int = 1,
    view: ViewArg = None,
) -> Session[ObsT, ActT]: ...
@overload
def session(
    model: type[Any],
    env: EnvTarget,
    *,
    instruction: str | None = None,
    close_env: bool = False,
    trust_entrypoints: bool | None = None,
    execution_horizon: int = 1,
    view: ViewArg = None,
) -> Session[Any, Any]: ...
@overload
def session(
    model: PredictFn[ObsT, ActT],
    env: LocalEnvTarget,
    *,
    instruction: str | None = None,
    close_env: bool = False,
    trust_entrypoints: bool | None = None,
    execution_horizon: int = 1,
    view: ViewArg = None,
) -> Session[ObsT, ActT]: ...
@overload
def session(
    model: object,
    env: EnvTarget,
    *,
    instruction: str | None = None,
    close_env: bool = False,
    trust_entrypoints: bool | None = None,
    execution_horizon: int = 1,
    view: ViewArg = None,
) -> Session[Any, Any]: ...
def session(
    model: object,
    env: EnvTarget,
    *,
    instruction: str | None = None,
    close_env: bool = False,
    trust_entrypoints: bool | None = None,
    execution_horizon: int = 1,
    view: ViewArg = None,
) -> Session[Any, Any]:
    """Bind a model to an env and return a :class:`Session` to drive by hand or via run().

    ``model`` is a local :class:`Model` (instance, subclass class, or a bare predict
    callable) or a served handle (:class:`RemoteModel` / :class:`SandboxModel`); ``env``
    is a local env, an :class:`~rlmesh.EnvFactory`, a remote-env handle, or an address
    string. A spec'd model resolves its adapter from the env's tags -- a local env must
    carry them (via :func:`rlmesh.adapters.tag` or an :class:`~rlmesh.EnvFactory`).
    A *served* model rejects ``instruction=`` (the env's own instruction is used).

    The returned session is yours: :meth:`Session.run` leaves it open, so drive it
    (or re-run it) as often as you like, then close it via ``close()`` or the
    ``with`` block.

    Pass :data:`rlmesh.RANDOM_SAMPLE` as ``model`` for a random baseline: each step
    samples the env's action space, no spec or adapter involved.

    Typing: a :class:`Model` instance or an annotated predict callable flows its
    observation/action types onto the returned ``Session`` (``predict``/``step``
    are typed accordingly); a class source, duck-typed policy, or served handle
    yields ``Session[Any, Any]``.
    """
    from ._adapter_mode import NO_ADAPTER
    from ._eval import RANDOM_SAMPLE, Session

    if model is RANDOM_SAMPLE:
        if execution_horizon > 1:
            warnings.warn(
                f"execution_horizon={execution_horizon} was requested but "
                "RANDOM_SAMPLE defines no predict_chunk(); running un-chunked.",
                stacklevel=2,
            )
        # A random baseline samples the env's action space and ignores observations,
        # so it adapts nothing -- skip adapter resolution even on a tagged env.
        return Session._create(  # pyright: ignore[reportPrivateUsage]
            env=env,
            # RANDOM_SAMPLE is a private sentinel Session special-cases by
            # identity; cast keeps it out of the public `predict` signature.
            predict=cast("Any", RANDOM_SAMPLE),
            spec=NO_ADAPTER,
            instruction=instruction,
            close_env=close_env,
            trust_entrypoints=bool(trust_entrypoints),
            execution_horizon=execution_horizon,
            view=view,
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
                trust_entrypoints=trust_entrypoints,
                execution_horizon=execution_horizon,
                view=view,
            ),
        )
    return as_model(model).session(
        cast("LocalEnvTarget", env),
        instruction=instruction,
        close_env=close_env,
        trust_entrypoints=trust_entrypoints,
        execution_horizon=execution_horizon,
        view=view,
    )


def run(
    model: object,
    env: EnvTarget,
    *,
    seeds: Sequence[int] | None = None,
    max_episodes: int | None = None,
    max_episode_steps: int | None = None,
    max_episode_seconds: float | None = None,
    hooks: RunHooks | None = None,
    instruction: str | None = None,
    close_env: bool = False,
    trust_entrypoints: bool | None = None,
    execution_horizon: int = 1,
    view: ViewArg = None,
) -> RunResult:
    """Drive ``model`` against ``env`` to completion and return a :class:`RunResult`.

    A *local* model (a :class:`Model` instance, subclass class, or bare predict
    callable) runs on :meth:`Model.run`'s native runtime loop -- single or
    vectorized env, batched predict corners, runtime-enforced seeds/caps; see
    its docstring for the parameter surface (``hooks`` / ``instruction`` /
    ``view`` are :func:`rlmesh.session`-only). A served :class:`RemoteModel` /
    :class:`SandboxModel` (and the :data:`rlmesh.RANDOM_SAMPLE` baseline) runs
    through its own session loop, which supports all parameters.
    """
    from ._eval import RANDOM_SAMPLE

    if model is not RANDOM_SAMPLE and not _served_handle(model):
        return as_model(model).run(
            cast("LocalEnvTarget", env),
            seeds=seeds,
            max_episodes=max_episodes,
            max_episode_steps=max_episode_steps,
            max_episode_seconds=max_episode_seconds,
            hooks=hooks,
            instruction=instruction,
            close_env=close_env,
            trust_entrypoints=trust_entrypoints,
            execution_horizon=execution_horizon,
            view=view,
        )
    sess = session(
        model,
        env,
        instruction=instruction,
        close_env=close_env,
        trust_entrypoints=trust_entrypoints,
        execution_horizon=execution_horizon,
        view=view,
    )
    try:
        return sess.run(
            seeds=seeds,
            max_episodes=max_episodes,
            max_episode_steps=max_episode_steps,
            max_episode_seconds=max_episode_seconds,
            hooks=hooks,
        )
    finally:
        sess.close()


__all__ = ["LifecycleCallback", "ModelBase", "PredictFn", "run", "session"]
