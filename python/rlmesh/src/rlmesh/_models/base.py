"""Shared Python model wrapper: construct from any source, eval against an env, serve."""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import dataclasses
import functools
import inspect
import os
import uuid
import weakref
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

from .._load_native import load_native
from .._upgrade import model_session, removed
from .._value_conversion import ValueBridge, identity_bridge, tree_map
from ..types import EnvTarget, LocalEnvTarget, Value, ViewArg
from ._calls import ModelCalls
from ._episodes import EpisodeStore
from ._instruction import text_placements, tree_set

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModel, ServeOptions
    from rlmesh.params import ParamSpec

    from .._server import EnvServer
    from ._binding import Binding
    from ._eval import HooksArg, NativeRunContext, RunResult

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
#: An episode-end / close hook. ``on_close`` takes nothing; ``on_episode_end``
#: may take the ended episode's id (see :func:`accepts_episode_id`), so the
#: alias stays arity-agnostic rather than splitting the public ``Model(...)``
#: signature in two.
LifecycleCallback = Callable[..., None]
PredictFn = Callable[[ObsT], ActT]
Corner = Callable[..., object]

#: The four predict corners, ordered general -> specific. A model overrides any
#: subset; :func:`_synthesize_corners` derives the rest. PREDICT_CORNERS is the
#: introspection-facing name (describe reports which corners a model defines).
_CORNERS = ("predict", "predict_chunk", "predict_batch", "predict_chunk_batch")
PREDICT_CORNERS = _CORNERS
_ModelT = TypeVar("_ModelT", bound="ModelBase[Any, Any]")

_NATIVE_CHUNK_WITHOUT_CORNER = (
    "native_chunk declares how long ONE action chunk is, but this model "
    "defines no predict_chunk() / predict_chunk_batch(). Add a chunk "
    "corner, or drop native_chunk."
)


def _random_only(observation: object) -> object:
    """``Model(RANDOM_SAMPLE)``'s predict: the baseline acts only in a run or binding."""
    _ = observation
    raise TypeError(
        "RANDOM_SAMPLE samples the env's action space: run or bind it to an env"
    )


def _served_handle(model: object) -> bool:
    """Whether ``model`` is a served handle (a ``SandboxModel``).

    A served handle binds itself via its own ``.session``; a local
    :class:`ModelBase` (or a subclass class) is normalized through
    :func:`as_model` instead.
    """
    return (
        callable(getattr(model, "session", None))
        and not isinstance(model, type)
        and not isinstance(model, ModelBase)
    )


def accepts_context(fn: Corner, base_arity: int) -> bool:
    """Whether ``fn``'s ``base_arity``-th positional parameter is a trailing ``context``.

    E.g. ``predict(obs)`` has base_arity 1, ``predict_chunk(obs, horizon)`` has
    base_arity 2 -- both may additionally accept a trailing context, but only
    when that extra parameter is actually named ``context``: an unrelated extra
    optional positional must NOT be mistaken for it. A genuinely variadic
    ``*args`` is assumed to accept anything, including a context.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) > base_arity and positional[base_arity].name == "context"


def accepts_episode_id(callback: LifecycleCallback) -> bool:
    """Whether an episode-end hook wants the ended episode's id.

    ``Model.on_episode_end(self, episode_id="")`` is the full form;
    ``on_episode_end(self)`` stays a valid override (and a bare
    ``on_episode_end=lambda: ...`` stays valid too), so the id is passed only when
    the callback has somewhere to put it.
    """
    try:
        params = list(inspect.signature(callback).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    return any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)


def _episode_end_hook(
    callback: LifecycleCallback | None,
) -> Callable[[str], None] | None:
    """Normalize an episode-end hook to the ``(episode_id)`` form the store calls."""
    if callback is None:
        return None
    if accepts_episode_id(callback):
        return callback

    def without_id(_episode_id: str) -> None:
        callback()

    return without_id


def _with_episode_state(
    store: EpisodeStore,
    fn: Corner | None,
    *,
    arity: Literal[1, 2],
    batched: bool,
) -> Corner | None:
    """Enrich a corner's raw episode context from the model's episode store.

    The engine hands each corner the raw ``{"episode_id", "episode_seed"}`` it
    knows -- one dict for a per-lane corner, one per row for a batched one. This
    turns each into the full :class:`~rlmesh.types.PredictContext`
    (``predict_index`` / ``predict_seed`` / ``state``), which is the only place
    per-episode bookkeeping lives. A corner that never declared a ``context``
    parameter is returned untouched, so a model that never asked pays nothing --
    not even a store entry.
    """
    if fn is None or not accepts_context(fn, arity):
        return fn
    corner = fn

    def enrich(context: Any) -> Any:
        if batched:
            return [store.context(row) for row in context]
        return store.context(context)

    if arity == 1:

        def with_state(observation: object, context: object = None) -> object:
            if context is None:
                return corner(observation)
            return corner(observation, enrich(context))

        return with_state

    def with_state_chunk(
        observation: object, horizon: object, context: object = None
    ) -> object:
        if context is None:
            return corner(observation, horizon)
        return corner(observation, horizon, enrich(context))

    return with_state_chunk


def _debatch(
    bridge: ValueBridge, batched_fn: Corner, *, arity: Literal[1, 2]
) -> Corner:
    """Single-lane corner from a batched one: run a batch of one and unwrap.

    Wraps the lone observation into a 1-lane batch (``tree_stack``), calls the
    batched corner, and peels lane 0 back off (``tree_unstack``). ``arity`` is
    the derived corner's own fixed positional arity -- 1 turns ``predict_batch``
    into ``predict(observation)``, 2 turns ``predict_chunk_batch`` into
    ``predict_chunk(observation, horizon)``. A batched corner's context is a
    LIST (one per row), so the single lane's own context is handed down as a
    one-element list -- the batch of one it is being run as.
    """
    takes_context = accepts_context(batched_fn, arity)

    if arity == 1:

        def derived_predict(observation: object, context: object = None) -> object:
            fused = bridge.tree_stack([observation])
            args = (
                (fused, [context])
                if takes_context and context is not None
                else (fused,)
            )
            return bridge.tree_unstack(batched_fn(*args), 1)[0]

        return derived_predict

    def derived_predict_chunk(
        observation: object, horizon: object, context: object = None
    ) -> object:
        fused = bridge.tree_stack([observation])
        args = (
            (fused, horizon, [context])
            if takes_context and context is not None
            else (fused, horizon)
        )
        return bridge.tree_unstack(batched_fn(*args), 1)[0]

    return derived_predict_chunk


def _first_frame(chunk: object) -> object:
    """The first action of a single chunk, matching the engine's ``split_chunk``.

    A ``Mapping`` (Dict-space action) carries the chunk axis INSIDE each leaf, so
    recurse per value; a ``list`` is itself the frame axis; an array's
    leading axis is the frame axis; a tuple, a scalar or text is a single-frame
    chunk and stands as its own action. Keeps local de-chunking byte-consistent
    with the served path (``stateful.rs::split_chunk``).
    """
    if isinstance(chunk, Mapping):
        items = cast("Mapping[Any, Any]", chunk)
        return {k: _first_frame(v) for k, v in items.items()}
    if isinstance(chunk, list):
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


def _chunk_len(chunk: object, *, batched: bool) -> int | None:
    """Frames in one chunk (``batched``: per lane, axis 1), ``None`` if unmeasurable."""
    if isinstance(chunk, Mapping):
        for value in cast("Mapping[Any, Any]", chunk).values():
            if (n := _chunk_len(value, batched=batched)) is not None:
                return n
        return None
    if batched:
        shape = getattr(chunk, "shape", None)
        return int(shape[1]) if shape is not None and len(shape) >= 2 else None
    if isinstance(chunk, list):
        return len(cast("Any", chunk))
    if isinstance(chunk, tuple):
        # A tuple is one frame, as the native split_chunk keeps it whole.
        return 1
    if getattr(chunk, "ndim", 0) >= 1:
        return int(cast("Any", chunk).shape[0])
    return None


def _dechunk(
    chunk_fn: Corner, *, batched: bool, native_chunk: Callable[[], int | None]
) -> Corner:
    """Single-action corner from a chunk one: run horizon 1 and take the first action.

    Single: take the first frame (``_first_frame``, ``split_chunk`` semantics).
    Batched: the chunk leaves are ``[B, horizon, ...]``, so take frame 0 past the
    batch axis (``leaf[:, 0]``), leaving non-array leaves (text) untouched. The
    derived corner's own arity is always 1 (``observation`` [, ``context``]);
    ``chunk_fn`` is the normalized ``(observation, horizon)`` contract, and a
    trailing context (see :func:`accepts_context`) only reaches it when its
    own signature asks for one -- most catalog models author only
    ``predict_chunk``, so this is the path their un-chunked ``predict()``
    actually runs through.
    """
    takes_context = accepts_context(chunk_fn, 2)

    def derived(observation: object, context: object = None) -> object:
        args = (
            (observation, 1, context)
            if context is not None and takes_context
            else (observation, 1)
        )
        chunk = chunk_fn(*args)
        # The runtime never sees this chunk (it is sliced to one action right
        # here), so a declared K is held to it at the only place it is whole.
        declared = native_chunk()
        if declared is not None:
            got = _chunk_len(chunk, batched=batched)
            if got is not None and got != declared:
                raise ValueError(
                    f"model declares native_chunk={declared} but its chunk corner "
                    f"returned {got} frames at execution_horizon=1: return the WHOLE "
                    "native chunk and let the runtime execute its prefix (do not "
                    "slice to the horizon), or drop the declaration"
                )
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
    ``None`` means the corner takes no horizon. A trailing ``context`` parameter
    is never the horizon (see :func:`accepts_context`), so it does not count.
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    positional = sum(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "context"
        for p in params
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
    chunk_fn = fn
    by_keyword = mode == "keyword"

    if accepts_context(chunk_fn, 1):
        # No horizon but a context: its base arity is 1, so the context sits
        # where the horizon would have been. Keep it rather than swallow it.
        def normalized_with_context(
            observation: object, execution_horizon: object, context: object = None
        ) -> object:
            if by_keyword:
                return chunk_fn(
                    observation, context, execution_horizon=execution_horizon
                )
            return chunk_fn(observation, context)

        return normalized_with_context

    def normalized(observation: object, execution_horizon: object) -> object:
        if by_keyword:
            return chunk_fn(observation, execution_horizon=execution_horizon)
        return chunk_fn(observation)

    return normalized


def _synthesize_corners(
    bridge: ValueBridge,
    predict: Corner | None,
    predict_chunk: Corner | None,
    predict_batch: Corner | None,
    predict_chunk_batch: Corner | None,
    native_chunk: Callable[[], int | None] = lambda: None,
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
            pc = _debatch(bridge, pcb, arity=2)
        if pb is None and can_dechunk:
            pb = _dechunk(pcb, batched=True, native_chunk=native_chunk)
    if p is None:
        if pb is not None:
            p = _debatch(bridge, pb, arity=1)
        elif pc is not None and can_dechunk:
            p = _dechunk(pc, batched=False, native_chunk=native_chunk)
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


#: Models that started a ``serve`` in this process, by ``id`` (a model need not
#: be hashable), held weakly.
_serving_models: weakref.WeakValueDictionary[int, ModelBase[Any, Any]] = (
    weakref.WeakValueDictionary()
)


def _close_serving_models() -> None:
    """Close every model still serving when the interpreter exits.

    A serve left running (typically on a daemon thread) keeps native threads
    calling into Python, and one that does so once finalization has begun
    kills the process with a signal. ``atexit`` runs while the interpreter is
    still whole, and ``close()`` is how a serve is stopped: it drains the serve,
    waits for in-flight calls and fires the close hook, as the serve's own end
    would. A model already closed is a no-op.
    """
    for model in list(_serving_models.values()):
        with contextlib.suppress(Exception):
            model.close()


_ = atexit.register(_close_serving_models)


@dataclasses.dataclass(frozen=True)
class BatchOptions:
    """How a served model coalesces predicts from every session it answers in.

    Requests that arrive close together go to the model as one grouped predict,
    one fused forward for a model that fuses. The model still runs one forward
    at a time: batching raises fill, not concurrency. A batch closes at
    ``max_batch_size`` groups (one route's rows each) or ``batch_byte_budget``
    observation bytes; while the model is idle it also closes
    ``batch_window_ms`` after its first request, and while a forward runs it
    stays open until that forward ends. A model that cannot fuse
    (``allow_fusion = False``, or no batched corner) is never batched.

    ``max_batch_size=1`` turns batching off. ``max_in_flight_batches=2`` lets
    the next batch assemble while one runs; 1 assembles only while the model is
    idle. ``batch_records`` appends one JSON line per batch, in the managed
    platform's ``telemetry/batches.jsonl`` shape.
    """

    max_batch_size: int = 32
    batch_window_ms: int = 5
    batch_byte_budget: int = 256 * 1024 * 1024 - 4 * 1024
    max_in_flight_batches: int = 2
    batch_records: str | os.PathLike[str] | None = None


class ModelBase(Generic[ObsT, ActT]):
    """A model: a policy you serve or drive against an env.

    Construct one two ways:

    * **Wrap** a predict callable -- ``Model(lambda obs: ...)``.
    * **Subclass** ``Model`` and override ``predict`` (and optionally ``load`` for
      weight loading, ``on_episode_end``/``close`` lifecycle hooks, and the ``spec`` class
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
            subclass's ``on_episode_end``/``close`` methods, and a wrapped policy's
            ``reset``/``close``). There is no episode-*begin* hook: per-episode
            state is lazy-seeded on first predict, so a stateful model clears its
            state at episode *end* via ``on_episode_end``. This fires
            identically on the local ``run(env)``/``session`` loop and the served
            wire path (driven there by the explicit ``ResetAdapter``), so local and
            served behave the same.
        trust_entrypoints: Allow ``module:callable`` custom-input entrypoints in a
            spec to be imported during adapter resolution.

    Example::

        from rlmesh.numpy import Model

        result = Model(lambda observation: 0).run("127.0.0.1:5555/env", seeds=[0])
        print(result.mean_reward)
    """

    _bridge: ClassVar[ValueBridge]
    #: The model's content: a ``ModelSpec``, ``NO_ADAPTER``, or ``None``. Set it as a
    #: class attribute when subclassing; the ``spec=`` kwarg overrides it per instance.
    spec: object | None = None
    #: Optional declared construction-parameter surface for ``load`` -- the model
    #: counterpart of :attr:`rlmesh.EnvFactory.params`, presented/swept the same way
    #: (see :mod:`rlmesh.params`). :meth:`from_config` and the served binding
    #: (``RLMESH_MAKE_KWARGS``) validate their keywords against it before
    #: ``load(**config)`` runs; a dashboard reads it via ``rlmesh.describe``.
    params: ClassVar[ParamSpec | None] = None
    #: The workflow edition this model was authored against -- a sticky
    #: declaration, the source-resident analogue of NixOS's ``stateVersion``:
    #: write it once and upgrading rlmesh cannot change how this model behaves.
    #: Paste the value :func:`rlmesh.current_workflow_edition` reports. ``None``
    #: (the default) declares nothing and floats to this build's newest edition,
    #: which is reported once per process. Overridden by
    #: ``RLMESH_WORKFLOW_EDITION`` and by an explicit ``run``/``session``
    #: keyword; see :doc:`the precedence table </editions/index>`.
    workflow_edition: ClassVar[str | None] = None
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
    #: This model's NATIVE chunk length K -- how many per-step actions ONE
    #: :meth:`predict_chunk` / :meth:`predict_chunk_batch` call returns. Set it as a
    #: class attribute, or in :meth:`load` when the checkpoint decides it
    #: (``self.native_chunk = int(self.policy.horizon)``); it is read once, after
    #: ``load()``, when the worker is built. ``None`` (the default) leaves the chunk
    #: length undeclared: the runtime executes ``min(len(chunk), execution_horizon)``
    #: and cannot check anything. Declaring it is a promise -- the runtime then
    #: refuses an ``execution_horizon`` above K and fails a predict whose chunk is
    #: not exactly K long, which is what catches a model still slicing its own chunk
    #: down to the horizon.
    native_chunk: int | None = None

    #: A checkpoint may set this in load(); Rust selects and validates it.
    execution_horizon: int | None = None

    def _served(self, address: str) -> Callable[[object], object]:
        """Take ``address`` as a served model's; the predict stand-in it returns refuses."""
        from .._rlmesh import is_v2_address, parse_address

        if not is_v2_address(address):
            raise ValueError(
                f"Model({address!r}): a served model is reached by its participant "
                "address (e.g. 'lab:7447/openpi'); 0.2 no longer dials a v1 model "
                "endpoint"
            )
        parsed = parse_address(address)
        if not parsed.get("participant"):
            raise ValueError(
                f"Model({address!r}) names no model: end the address with the model's "
                "name, e.g. 'lab:7447/openpi'"
            )
        self._served_address = address

        def served_only(observation: object) -> object:
            raise TypeError(
                f"Model({address!r}) is served elsewhere and predicts per env: run it, "
                "bind it to a target, or serve it"
            )

        return served_only

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Return this model's full metadata envelope (see :func:`rlmesh.describe`)."""
        from .._describe import describe  # lazy: avoid an import cycle at module load

        return describe(cls, kind="model")

    @classmethod
    def from_config(cls: type[_ModelT], **config: Any) -> _ModelT:
        """Build this model with ``load(**config)``: the configured constructor.

        ``MyModel()`` loads with ``load``'s own defaults; ``from_config`` is the
        same construction with keywords -- every keyword configures :meth:`load`
        and is validated first against the signature and the declared
        :attr:`params` (unknown names, missing required ones, and out-of-domain
        values raise :class:`~rlmesh.params.ParamError`). The instance is built
        without the automatic load, then ``load(**resolved)`` runs exactly once;
        a served model's binding takes the identical path, so a configuration
        that works locally serves the same. Only an authored subclass qualifies:
        a framework base class with no predict corner of its own is refused
        (wrap an existing policy with ``Model(predict, spec=...)`` instead).
        """
        if all(getattr(cls, name) is getattr(ModelBase, name) for name in _CORNERS):
            raise TypeError(
                f"{cls.__name__}.from_config() needs an authored Model subclass "
                "(one that implements predict() or another predict corner); wrap "
                "an existing policy with Model(predict, spec=...) instead."
            )
        from .._bootstrap.loaders import construct_authored_model

        return cast("_ModelT", construct_authored_model(cls, **config))

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
        # The model's one call gate: every corner and hook call, from every
        # session, run and serve on this model, runs through it one at a time.
        self._calls = ModelCalls()
        #: Held while a run drives this model (RUN-9).
        self._bridge.ensure_available()

        # A Model subclass overrides at least one predict corner (plus optionally
        # load/on_episode_end/close/spec); the corners it leaves out are derived below, so
        # overriding predict_chunk_batch alone is enough. A wrapped callable
        # supplies a single predict.
        def _overridden(method_name: str) -> Corner | None:
            if getattr(type(self), method_name) is getattr(ModelBase, method_name):
                return None
            return cast("Corner", getattr(self, method_name))

        corners = {name: _overridden(name) for name in _CORNERS}
        #: A served model's participant address (``Model("lab:7447/openpi")``):
        #: runs, bindings and serves then drive that model.
        self._served_address: str | None = None
        #: The host a run or binding serves a local env on, each in a session
        #: of its own, when not this process's: a sandboxed model's, which its
        #: container can reach.
        self._env_session: str | None = None
        #: The random baseline (``Model(RANDOM_SAMPLE)``): runs and bindings
        #: sample the env's action space.
        self._random = False
        from ._coerce import RANDOM_SAMPLE

        if source is RANDOM_SAMPLE:
            self._random = True
            source = _random_only
        if isinstance(source, str):
            source = self._served(source)
        #: Whether this subclass loads in ``__init__`` (after its declarations
        #: are checked); the served-construction path suppresses it and calls
        #: ``load(**binding)`` itself once the params are resolved.
        autoload = False

        if source is None and any(fn is not None for fn in corners.values()):
            # Subclass-authoring mode. Weights load once, after the declarations
            # below are checked, so a bad declaration costs no weight load.
            autoload = not _suppress_autoload.get()
            resolved_spec = spec if spec is not None else type(self).spec
            coerced_on_episode_end: LifecycleCallback | None = self.on_episode_end
            # An authored close() is the public close itself (wrapped by
            # __init_subclass__ to drop the worker), never a hook: wiring it
            # here would fire it twice when it calls super().close().
            coerced_on_close: LifecycleCallback | None = None
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

        # The model's one bounded per-episode store, wrapped around every corner
        # that asked for a context (before synthesis, so a derived corner inherits
        # the enrichment instead of needing its own). Its end edge -- the episode's
        # entry dropped, then the author's own hook -- is what both the served
        # ResetAdapter and the local session boundary drive.
        self._episodes = EpisodeStore(
            _episode_end_hook(
                on_episode_end if on_episode_end is not None else coerced_on_episode_end
            )
        )
        raw_predict = _with_episode_state(
            self._episodes, raw_predict, arity=1, batched=False
        )
        raw_predict_chunk = _with_episode_state(
            self._episodes, raw_predict_chunk, arity=2, batched=False
        )
        raw_predict_batch = _with_episode_state(
            self._episodes, raw_predict_batch, arity=1, batched=True
        )
        raw_predict_chunk_batch = _with_episode_state(
            self._episodes, raw_predict_chunk_batch, arity=2, batched=True
        )

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
                self._native_chunk,
            )
        )
        if raw_predict is None:
            raise TypeError(
                "could not derive predict(): define predict(), or a corner it can be "
                "derived from (a chunk-only model on the raw Value bridge needs an "
                "explicit predict())."
            )
        self._has_chunk_corner = (
            raw_predict_chunk is not None or raw_predict_chunk_batch is not None
        )
        # A class-level K with no chunk corner can never be honored (and nothing
        # would ever check it), so reject it where the corners are known, before
        # any weights load. A K set in load() meets the same door after load.
        if (
            getattr(type(policy), "native_chunk", None) is not None
            and not self._has_chunk_corner
        ):
            raise TypeError(_NATIVE_CHUNK_WITHOUT_CORNER)

        # Every corner call goes through the model's call gate, one at a time.
        self._raw_predict = cast(
            "PredictFn[ObsT, ActT]", self._calls.gated(raw_predict)
        )
        self._raw_predict_chunk = self._calls.gated(raw_predict_chunk)
        self._raw_predict_batch = self._calls.gated(raw_predict_batch)
        self._raw_predict_chunk_batch = self._calls.gated(raw_predict_chunk_batch)
        self.spec = resolved_spec
        self._policy = policy
        #: A bare predict callable, whose own attribute is its edition
        #: declaration (``None`` for a subclass or a wrapped policy).
        self._edition_source: object = source if policy is None else None
        #: The close hook: the ``on_close=`` option, which replaces an authored
        #: ``close()`` body too, else a wrapped policy's own ``close``.
        self._on_close = on_close if on_close is not None else coerced_on_close
        #: The episode-end edge both paths drive: drop the store entry, then fire
        #: the model's own hook with the id that ended. After ``close()`` it
        #: submits nothing.
        episodes = self._episodes
        calls = self._calls

        def end_episode(episode_id: str = "") -> None:
            calls.hook(episodes.end, episode_id)

        def drop_env(env_id: str) -> None:
            calls.hook(episodes.drop_env, env_id)

        self._on_episode_end: LifecycleCallback = end_episode
        self._drop_env_entries: Callable[[str], None] = drop_env
        self._trust_entrypoints = trust_entrypoints
        self._worker: PyModel | None = None
        #: Per-run state the cached native worker reads at call time: the
        #: ``instruction=`` override and the hooks' run context.
        self._instruction: str | None = None
        self._native_run: NativeRunContext | None = None
        self._loaded_native_chunk: int | None = None
        self._loaded_execution_horizon: int | None = None
        self._loaded_allow_fusion = True
        if autoload:
            self.load()
        if autoload or policy is not self:
            self._finish_load()

    def _finish_load(self) -> None:
        """Read the policy's declarations once, after ``load()``, and cache them.

        ``native_chunk`` (a checkpoint may set it in ``load()``) is refused here
        when the model has no chunk corner (``TypeError``, as a class-level K is)
        or when it is below 1 (``ValueError``); ``allow_fusion`` is read with it.
        A wrapping ``Model(policy)`` reads the policy's and reports them as its
        own attributes; a wrapped bare callable keeps the defaults.
        """
        policy = self._policy
        native_chunk = (
            getattr(policy, "native_chunk", None) if policy is not None else None
        )
        allow_fusion = (
            bool(getattr(policy, "allow_fusion", True)) if policy is not None else True
        )
        if native_chunk is not None:
            if not self._has_chunk_corner:
                raise TypeError(_NATIVE_CHUNK_WITHOUT_CORNER)
            if int(native_chunk) < 1:
                raise ValueError(
                    "native_chunk must be >= 1 (or None when undeclared), got "
                    f"{native_chunk!r}"
                )
            native_chunk = int(native_chunk)
        self._loaded_native_chunk = native_chunk
        loaded = (
            getattr(policy, "execution_horizon", None) if policy is not None else None
        )
        # Validate a checkpoint declaration even when a later run overrides it.
        if loaded is not None:
            load_native("resolve_execution_horizon")(loaded, None, None, native_chunk)
        self._loaded_execution_horizon = loaded
        self._loaded_allow_fusion = allow_fusion
        if policy is not self and policy is not None:
            # A wrapping Model reports its policy's declarations.
            self.native_chunk = native_chunk
            self.execution_horizon = loaded
            self.allow_fusion = allow_fusion
            setattr(self, "workflow_edition", self._class_workflow_edition())  # noqa: B010

    def _execution_horizon(self, explicit: int | None) -> int:
        """Marshal declarations to Rust, which selects and validates the horizon."""
        from ..adapters import ModelSpec

        loaded = self._loaded_execution_horizon
        if explicit is None and self._served_address is not None:
            loaded = self._install_worker().default_execution_horizon()
        timing = (
            self.spec.timing.execution if isinstance(self.spec, ModelSpec) else None
        )
        return load_native("resolve_execution_horizon")(
            explicit, loaded, timing, self._native_chunk()
        )

    def _native_chunk(self) -> int | None:
        """The native chunk K, read once after ``load()`` (see :meth:`_finish_load`)."""
        return self._loaded_native_chunk

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

        Optional subclass hook; a no-op by default. Runs exactly once per
        instance, before any predict: with its own defaults from ``MyModel()``,
        or with the keywords :meth:`from_config` / a served binding resolved
        against the signature and :attr:`params`.
        """

    def predict(self, observation: ObsT) -> ActT:
        """Map one observation to an action (or an action chunk per ``spec``).

        Override when subclassing ``Model``. A model built by wrapping a predict
        callable, or a served one (``Model("host:port/name")``), predicts here
        only as another model's part: held as an attribute of a model a run,
        bind or serve drives (``self.inner = inner``), ``self.inner.predict(obs)``
        in that model's ``predict`` answers the episode it is answering. Its
        observation is in the env's format, so the calling model is spec-less.
        """
        from .._load_native import load_native

        try:
            action = load_native("nested_predict")(
                id(self), self._bridge.encode(observation)
            )
        except LookupError:
            if self._served_address is not None:
                raise TypeError(
                    f"Model({self._served_address!r}) is served elsewhere and "
                    "predicts per env: run it, bind it to a target, serve it, or "
                    "call it from the predict of a model that holds it"
                ) from None
            raise NotImplementedError(
                "Model subclasses must override predict(), or construct a Model by "
                "wrapping a predict callable, e.g. Model(lambda obs: ...)."
            ) from None
        return cast(ActT, self._bridge.decode(action))

    def reset_options(self, episode: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Optional: options to add to the env reset that starts ``episode``.

        ``episode`` holds the ``episode_id``, its ``seed`` and its
        ``trial_index``. A run asks before each reset it sends, and the env's
        ``reset(options=...)`` receives them (the runtime's own reserved keys
        win a clash). A reset of several lanes gets each key as a per-lane
        list. An env that autoresets (``NEXT_STEP``) restarts its lanes itself,
        with no reset to carry them. The default adds none.
        """
        return None

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

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        authored_init = cls.__dict__.get("__init__")
        if callable(authored_init):
            # A model that only sets its parts (``self.inner = inner``) need
            # not call super().__init__(): the base builds the model after it.
            @functools.wraps(authored_init)
            def init(self: ModelBase[Any, Any], *args: Any, **kwargs: Any) -> None:
                authored_init(self, *args, **kwargs)
                if "_calls" not in self.__dict__:
                    super(cls, self).__init__()

            cls.__init__ = init  # type: ignore[method-assign]
        authored_close = cls.__dict__.get("close")
        if callable(authored_close):
            # An authored close() is the model's close hook: it runs at most once,
            # inside the terminal close (serves stopped, in-flight calls done),
            # and an on_close= option replaces it. An override need not call
            # super().close(); inside the close that call is a no-op.
            @functools.wraps(authored_close)
            def close(self: ModelBase[Any, Any], *args: Any, **kwargs: Any) -> None:
                if self._calls.closing_here():
                    # A subclass's authored close calling super().close().
                    authored_close(self, *args, **kwargs)
                    return
                self._close_with(
                    functools.partial(authored_close, self, *args, **kwargs)
                )

            cls.close = close  # type: ignore[method-assign]
        if "reset" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__}.reset is never called: the Model episode hook is "
                "on_episode_end(self, episode_id='') (a wrapped policy object's "
                "reset() is still honored)"
            )

    def on_episode_end(self, episode_id: str = "") -> None:
        """Optional: called when an episode ends (no-op by default).

        The only per-episode boundary both the local loop and the served wire path
        signal, so a stateful model clears its state here. ``episode_id`` is the
        episode that just ended -- declare it when the model keeps state keyed by
        id (several episodes interleave through one served model);
        ``def on_episode_end(self)`` remains a valid override when it does not.
        """

    def close(self) -> None:
        """Release this model's resources: the explicit end of its life.

        Never called by ``run()`` or a session close -- a model instance is
        borrowed there and stays usable. Override it to unload weights; on a
        wrapped policy the base fires the ``on_close`` callback (or the
        policy's own ``close``). Serving calls it when the server stops.

        Terminal and idempotent: it stops a ``serve`` still running on the model
        (requests it already received finish, and that ``serve`` returns),
        waits for in-flight corner calls, then fires the close hook once. Every
        later corner call, ``run``, ``session`` or ``serve`` -- and ``predict``,
        ``step`` and ``run`` on a session opened before -- raises
        ``RuntimeError``; a session closed afterwards fires no
        ``on_episode_end``.
        """
        if self._calls.closing_here():
            return
        self._close_with(None)

    def _close_with(self, authored: Callable[[], object] | None) -> None:
        """The one terminal close; ``authored`` is a subclass's own close body."""
        calls = self._calls
        if not calls.begin_close():
            return
        try:
            hook = self._on_close if self._on_close is not None else authored
            if hook is not None:
                calls.run_close_hook(hook)
        finally:
            calls.finish_close()
            # The native worker's callbacks close over self through a class the
            # cycle collector cannot traverse; dropping it lets the model be freed.
            self._worker = None

    def _install_worker(self) -> PyModel:
        """Build (once) and return the native model worker (the serve path).

        A spec'd model's adapter resolves per env at ``resolve_adapter`` (the
        served endpoint receives the env contract there): ``configure`` resolves
        it from the env's contract and the native worker applies it around the
        raw predict. A spec-less / ``NO_ADAPTER`` model serves its own predict.

        Built lazily on the first serve, run or bind, not in ``__init__``: a
        model only called directly (``predict``) never needs the native worker, and the worker's callbacks close over
        ``self`` -- a reference cycle through a native class the Python cycle
        collector cannot free -- so an unserved model should not pay for one.
        """
        if self._worker is not None:
            return self._worker
        from .._load_native import load_native
        from ._eval import resolve_adapter

        if self._served_address is not None:
            served: PyModel = load_native("PyModel").served(self._served_address)
            self._worker = served
            return served
        if self._random:
            random: PyModel = load_native("PyModel").random_sample()
            self._worker = random
            return random

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
            if self._native_run is not None:
                self._native_run.contract = env_contract
            return adapter.serve_route(bridge) if adapter is not None else None

        placements = text_placements(spec)

        def observe(observation: Value) -> Any:
            """The engine's assembled model input, in the framework, on device.

            A run's ``instruction=`` lands in every text leaf here, per lane,
            in the leaf's declared shape (the Session loop does the same).
            """
            payload = self._to_device(bridge.decode(observation))
            instruction = self._instruction
            if instruction is not None:
                for placement in placements:
                    value: Any = [instruction] if placement.as_list else instruction
                    payload = tree_set(payload, placement.segments, value)
            return payload

        raw_predict_takes_context = accepts_context(raw_predict, 1)

        def predict_neutral(
            observation: Value, context: Mapping[str, Any] | None = None
        ) -> Value:
            # The engine has already applied the adapter (declarative transform,
            # frame-stacking, customs, enc-shims) in Rust, so the obs handed here
            # is the final model input; bridge it into the framework, run the user
            # predict, and bridge the action back. A spec-less route hands the raw
            # observation through the identical path (no adapter). ``context``
            # (this lane's episode identity) reaches the author's own predict()
            # only when its signature declares a trailing ``context`` param.
            decoded = cast(ObsT, observe(observation))
            if context is not None and raw_predict_takes_context:
                action = cast("Corner", raw_predict)(decoded, context)
            else:
                action = raw_predict(decoded)
            return bridge.encode(action)

        # The chunk corner, when the model defines one: identical bridging, but the
        # user returns a chunk (leading axis = chunk) the native engine splits.
        raw_predict_chunk = self._raw_predict_chunk
        predict_chunk_neutral: Callable[..., Value] | None = None
        if raw_predict_chunk is not None:
            chunk_fn = raw_predict_chunk
            chunk_fn_takes_context = accepts_context(chunk_fn, 2)

            def _predict_chunk_neutral(
                observation: Value,
                horizon: int,
                context: Mapping[str, Any] | None = None,
            ) -> Value:
                decoded = cast(ObsT, observe(observation))
                if context is not None and chunk_fn_takes_context:
                    chunk = chunk_fn(decoded, horizon, context)
                else:
                    chunk = chunk_fn(decoded, horizon)
                return bridge.encode(chunk)

            predict_chunk_neutral = _predict_chunk_neutral

        # The batched corners (one forward for the whole vector), when defined: the
        # engine hands a list of N neutral lane inputs. Fuse them into ONE batched
        # value (every leaf gains a leading batch axis -- a Dict obs becomes
        # {key: array[N, ...]}, the shape every RL/VLA runtime hands a policy), run
        # the user's single batched forward, then split the batched action/chunk
        # back into N per-lane values the engine replays. The native (Value) bridge
        # can't fuse opaque tensors, so it passes the per-lane list through.
        raw_predict_batch = self._raw_predict_batch
        predict_batch_neutral: (
            Callable[[list[Value], list[Mapping[str, Any]]], list[Value]] | None
        ) = None
        if raw_predict_batch is not None:
            batch_fn = raw_predict_batch
            batch_fn_takes_context = accepts_context(batch_fn, 1)

            def _predict_batch_neutral(
                observations: list[Value], context: list[Mapping[str, Any]]
            ) -> list[Value]:
                if not observations:
                    return []
                fused = bridge.tree_stack([observe(o) for o in observations])
                # `context` is row-aligned with `observations`: one per lane of the
                # fused batch, which may span independent episodes.
                actions = (
                    batch_fn(fused, context)
                    if batch_fn_takes_context
                    else batch_fn(fused)
                )
                parts = bridge.tree_unstack(actions, len(observations))
                return [bridge.encode(a) for a in parts]

            predict_batch_neutral = _predict_batch_neutral

        raw_predict_chunk_batch = self._raw_predict_chunk_batch
        predict_chunk_batch_neutral: (
            Callable[[list[Value], int, list[Mapping[str, Any]]], list[Value]] | None
        ) = None
        if raw_predict_chunk_batch is not None:
            chunk_batch_fn = raw_predict_chunk_batch
            chunk_batch_fn_takes_context = accepts_context(chunk_batch_fn, 2)

            def _predict_chunk_batch_neutral(
                observations: list[Value],
                horizon: int,
                context: list[Mapping[str, Any]],
            ) -> list[Value]:
                if not observations:
                    return []
                fused = bridge.tree_stack([observe(o) for o in observations])
                chunks = (
                    chunk_batch_fn(fused, horizon, context)
                    if chunk_batch_fn_takes_context
                    else chunk_batch_fn(fused, horizon)
                )
                # Split the batch axis only; each lane's chunk (horizon) axis stays.
                parts = bridge.tree_unstack(chunks, len(observations))
                return [bridge.encode(c) for c in parts]

            predict_chunk_batch_neutral = _predict_chunk_batch_neutral

        # The serve's close is this model's own (`serve` calls close() when the
        # server stops), so the worker carries no close hook.
        # Asked per reset only when the model defines its own.
        defines = next(c for c in type(self).__mro__ if "reset_options" in c.__dict__)
        reset_options = cast(
            "Callable[[Mapping[str, Any]], object] | None",
            self._calls.gated(self.reset_options) if defines is not ModelBase else None,
        )
        worker: PyModel = load_native("PyModel")(
            predict_fn=predict_neutral,
            reset_options_fn=reset_options,
            configure_fn=configure,
            on_episode_end=self._on_episode_end,
            on_close=None,
            predict_chunk_fn=predict_chunk_neutral,
            predict_batch_fn=predict_batch_neutral,
            predict_chunk_batch_fn=predict_chunk_batch_neutral,
            allow_fusion=self._loaded_allow_fusion,
            native_chunk=self._native_chunk(),
            on_route_drop=self._drop_env_entries,
        )
        # The models this one holds as parts predict through their own
        # workers on the episodes this one answers.
        parts: list[ModelBase[Any, Any]] = [
            part
            for part in vars(self).values()
            if isinstance(part, ModelBase) and part is not self
        ]
        inners = [(id(part), part._install_worker()) for part in parts]
        if inners:
            worker.set_inner_models(inners)
        self._worker = worker
        return worker

    def run(
        self,
        env_or_address: LocalEnvTarget,
        *,
        seeds: Sequence[int] | None = None,
        episodes: int | None = None,
        max_episode_steps: int | None = None,
        max_episode_seconds: float | None = None,
        hooks: HooksArg = None,
        instruction: str | None = None,
        close_env: bool = False,
        trust_entrypoints: bool | None = None,
        execution_horizon: int | None = None,
        prefetch_lead: int = 0,
        trial_index_base: int = 0,
        workflow_edition: str | None = None,
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
        ``prefetch_lead`` (> 0) turns on async inference over that replay:
        with that many (or fewer) frames of the current chunk left, the
        runtime predicts the next chunk while they execute, so the chunk is
        conditioned on an observation up to ``prefetch_lead`` steps stale and
        the result is not comparable to a synchronous run; a chunk prefetched
        across an episode boundary is discarded.

        ``env_or_address`` is a bare address string the loop dials, an object
        with an ``address`` (:class:`~rlmesh.EnvServer`; an open ``RemoteEnv``
        is refused, as it holds its env's lease, while a
        :class:`~rlmesh.SandboxEnv` lends its container to the run and attaches
        again after), or a local env object -- served
        in this process for
        the duration of the run (tag it via
        :func:`rlmesh.adapters.tag` for a spec'd model). Both the model and a
        caller's env are borrowed: the run releases what it created (the
        in-process server, a factory-built env, the runtime session) and leaves
        them usable, on success, failure, or interrupt alike; ``close_env``
        opts a caller's env into shutdown, and :meth:`close` is the model's own
        release.

        ``episodes`` is the exact number of episodes the result holds: one by
        default, the length of ``seeds`` when only seeds are given (one reset
        seed per episode), and the two must agree when both are; ``0`` returns
        an empty result. On a vectorized env the budget bounds episode
        *starts*, so the scored set is fixed by the budget alone -- a lane the
        env rolls past it (``NEXT_STEP`` autoreset steps the whole vector in
        lockstep, so that lane cannot be paused) runs unscored: never counted,
        reported, or seen by ``hooks``. ``max_episode_steps`` /
        ``max_episode_seconds`` truncate an episode at a step / wall-clock cap,
        and a non-terminating env is bounded regardless: without an explicit
        cap the runtime truncates any episode at 100,000 steps (an
        autoresetting vector env owns its episodes, so there an episode that
        reaches it fails the run instead). Explicit
        seeds and caps need the runtime to own resets, so they are refused
        before any episode starts on an autoresetting vector env (drive it
        with ``episodes``; on a driver-reset vector env seeds are consumed one
        per episode start, in any count). Every episode walks a trial
        ordinal, ``trial_index_base + i`` for episode ``i``: the runtime
        delivers it as ``reset(options={"trial_index": ...})`` to an env that
        declared the key in :attr:`EnvFactory.reset_options
        <rlmesh.EnvFactory.reset_options>` (an env that did not never sees
        it) and reports it on :attr:`EpisodeResult.trial
        <rlmesh.EpisodeResult.trial>`, so a local eval walks the same states
        as a platform shard given the same base (a non-zero base needs the
        runtime to own resets, like ``seeds``). A per-call
        ``trust_entrypoints`` override applies to this run only. On the
        result, each episode's ``predict_ms`` / ``step_ms`` are that episode's
        own per-step means (the runtime times every predict and env step it
        issues; see :class:`~rlmesh.StepEvent` for how chunk replay and
        prefetch are charged), and ``RunResult.telemetry`` carries the run-wide
        aggregate -- every measured op/metric series with count and avg/p50/p95/p99
        (``print(result.format_telemetry())`` for a table) -- so a slow run can
        be attributed to the model forward, the env step, serialization, or
        queueing without a profiler.

        ``workflow_edition`` pins the semantics this run is evaluated under --
        the runtime's own declaration, above every other surface. It defaults to
        ``RLMESH_WORKFLOW_EDITION``, then :attr:`workflow_edition` on this model,
        then ``[tool.rlmesh] workflow_edition``; declaring none floats the run to
        this build's newest edition (reported once per process). An edition
        neither side can run is refused before any episode starts, naming what
        each tier wants and can do.

        ``hooks`` observes the loop: a :class:`~rlmesh.RunHooks` (or a list of
        them, each called in order) receives the
        runtime's own events -- episode starts, every step as a
        :class:`~rlmesh.StepEvent` with the terminal flags the runtime knows at
        that step, and episode ends -- in the same per-episode order as a
        :class:`~rlmesh.Binding`, with
        :meth:`RunHooks.on_run_start <rlmesh.RunHooks.on_run_start>` receiving a
        :class:`~rlmesh.RunContext` for role reads and frame discovery. Hooks
        never change the result: it is the runtime's report either way, and
        the :class:`~rlmesh.EpisodeResult` handed to ``on_episode_end`` is the
        same record the result holds. On a vectorized env episodes
        interleave. ``instruction`` overrides the text input of a spec'd model
        in its declared shape, exactly as on the session loop.

        ``view`` watches the run live (``True``, a backend name, or a
        :class:`~rlmesh.View`), the first lane of a vectorized env: skipping
        truncates the episodes in flight and the run goes on, and quitting
        truncates them and returns the partial result.
        """
        from ._eval import Interrupted, NativeRunContext, as_hooks
        from ._facade import EnvTarget, FacadeHookRelay, result_from_native
        from ._view import ViewerDriver, resolve_view

        self._calls.check_open("run()")
        self._require_device_support()
        context = NativeRunContext(
            trust=bool(
                self._trust_entrypoints
                if trust_entrypoints is None
                else trust_entrypoints
            )
        )
        resolved_view = resolve_view(view)
        viewer = ViewerDriver(resolved_view) if resolved_view is not None else None
        relay = FacadeHookRelay(as_hooks(hooks), context, viewer)
        target = EnvTarget(
            self,
            env_or_address,
            context,
            trust=trust_entrypoints,
            instruction=instruction,
        )
        failed = True
        try:
            report = self._install_worker().run_facade(
                target,
                relay,
                dict(
                    episodes=episodes,
                    seeds=list(seeds) if seeds is not None else None,
                    max_episode_steps=max_episode_steps,
                    max_episode_seconds=max_episode_seconds,
                    execution_horizon=self._execution_horizon(execution_horizon),
                    prefetch_lead=prefetch_lead,
                    trial_index_base=trial_index_base,
                    close_env=close_env,
                    workflow_edition=self._declared_workflow_edition(workflow_edition),
                ),
            )
            failed = False
            return result_from_native(report)
        except KeyboardInterrupt as exc:
            if isinstance(exc, Interrupted):
                raise
            raise Interrupted(relay.result) from exc
        finally:
            if viewer is not None:
                viewer.close(hold=not failed)

    def _serve_here(self, env_obj: object, workflow_edition: str | None) -> EnvServer:
        """Serve the local env ``env_obj`` in this process for one run or binding.

        In a session of this process's shared host, so the driver calls the
        env directly; or in a fresh session on :attr:`_env_session`'s host
        when set. A local single env's spec/tags pairing is pre-flighted
        here, so a mismatch surfaces as the typed ``AdapterResolutionError``
        (a vector env's local spaces are batched while tags describe one lane,
        so its check stays at the route resolve).
        """
        from .._editions import resolved_by_caller
        from .._load_native import load_native
        from .._server import EnvServer, VectorServerEnvLike

        if (
            not hasattr(env_obj, "single_observation_space")
            and self._served_address is None
            and not self._random
        ):
            from ._connect import local_contract
            from ._resolve import resolve_adapter

            resolve_adapter(
                self.spec,
                local_contract(env_obj),
                trust_entrypoints=self._trust_entrypoints,
            )
        # The in-process server is part of THIS call, so it takes the caller's
        # already-resolved declaration verbatim instead of re-resolving and
        # possibly landing on a different edition than the runtime tier.
        resolved = resolved_by_caller.set(True)
        try:
            address = (
                f"{self._env_session}/run-{uuid.uuid4().hex[:12]}/env"
                if self._env_session is not None
                else f"{load_native('local_run_session')()}/env"
            )
            server = EnvServer(
                cast("VectorServerEnvLike", env_obj),
                address,
                framework=getattr(env_obj, "_bridge", None),
                options=(
                    load_native("ServeOptions")(workflow_edition=workflow_edition)
                    if workflow_edition is not None
                    else None
                ),
                # The env is borrowed (or built here): its close() is the
                # caller's decision, not the in-process server's.
                close_env_on_shutdown=False,
            )
        finally:
            resolved_by_caller.reset(resolved)
        server.start()
        return server

    # The 0.1 shape, refused with its 0.2 form.
    session = model_session

    def bind(
        self,
        target: LocalEnvTarget,
        *,
        hooks: HooksArg = None,
        view: ViewArg = None,
        execution_horizon: int | None = None,
        close_env: bool = False,
        workflow_edition: str | None = None,
    ) -> Binding[ObsT, ActT]:
        """Bind this model to one env and return a :class:`Binding` to step by hand.

        The manual counterpart of :meth:`run`, on the same loop: ``reset`` starts
        an episode and ``step()`` steps it under this model's action (or
        ``step(action)`` under yours). ``target`` is an env object, an
        :class:`~rlmesh.EnvFactory`, an :class:`~rlmesh.EnvServer` or a session
        address (see :meth:`run`); a local env is served in this process for the
        binding's life, and a vector env is refused (a binding drives one lane).
        ``execution_horizon`` (> 1) replays that many actions of each chunk this
        model predicts. ``hooks`` and ``view`` see each reset and step as a
        run's would. Closing the binding never closes this model; ``close_env``
        opts a local env into closing with it.
        """
        from ._binding import Binding
        from ._eval import as_hooks
        from ._facade import EnvTarget
        from ._view import resolve_view

        self._calls.check_open("bind()")
        self._require_device_support()
        view = resolve_view(view)
        declared = self._declared_workflow_edition(workflow_edition)
        horizon = self._execution_horizon(execution_horizon)
        target_adapter = EnvTarget(self, target, binding=True)
        native = self._install_worker().bind_facade(
            target_adapter, horizon, declared, close_env
        )
        try:
            return Binding._create(  # pyright: ignore[reportPrivateUsage]
                native=native,
                bridge=self._bridge,
                hooks=as_hooks(hooks),
                view=view,
                target=target_adapter,
            )
        except BaseException:
            with contextlib.suppress(BaseException):
                native.close()
            raise

    def serve(
        self,
        address: str,
        *,
        name: str | None = None,
        options: ServeOptions | None = None,
        batching: BatchOptions | None = None,
    ) -> None:
        """Serve this model on a session (blocking).

        The model joins the session at ``address`` (``"lab:7447"``, ``":0"``,
        ``"bench"``) as the participant ``name`` and answers whoever drives
        it, until closed or interrupted: ``Model("lab:7447/openpi")`` reaches
        it. A run invites it into the session its env is in.

        Predicts from every session it answers in are coalesced into one
        grouped predict as ``batching`` says (:class:`BatchOptions`' defaults
        when ``None``).

        A spec'd model resolves its adapter per env from the env contract each
        route's resolve delivers, then applies it around predict; a spec-less /
        ``NO_ADAPTER`` model serves its own predict directly.

        A session checks no endpoint token yet, so a set
        ``RLMESH_MODEL_ENDPOINT_TOKEN`` is refused: keep the session on a
        private network.

        The model declares a workflow edition in the session's roster, where a
        driver negotiates the run's edition: ``options``' own
        ``workflow_edition`` if it sets one, else the resolved declaration
        (``RLMESH_WORKFLOW_EDITION``, :attr:`workflow_edition`, ``[tool.rlmesh]``).
        """
        from .._editions import serve_options_declaring

        self._calls.check_open("serve()")
        self._require_device_support()
        declared = serve_options_declaring(
            options,
            # The options' own edition is this call's value, ahead of
            # RLMESH_WORKFLOW_EDITION, as EnvServer adopts it (COR-6).
            call=options.workflow_edition if options is not None else None,
            declared=self._class_workflow_edition(),
        )
        if not name:
            removed(
                f"Model.serve({address!r}) without a name",
                "a model serves on a session as a named participant: "
                f"model.serve({address!r}, name='policy') is reached at "
                "'<host:port>/policy'",
            )
        worker = self._install_worker()
        # close() stops this serve (drain); the serve then returns and its own
        # close below is that one.
        token = self._calls.register_serve(worker.stop_serving)
        _serving_models[id(self)] = self
        try:
            from ..serve import (
                _handshake_describe,  # pyright: ignore[reportPrivateUsage]
            )

            envelope = _handshake_describe(
                self,
                "model",
                None,
                (declared.workflow_edition if declared is not None else None) or "",
            )
            batch = batching if batching is not None else BatchOptions()
            worker.serve(
                address,
                name,
                declared,
                envelope,
                max_batch_size=batch.max_batch_size,
                batch_window_ms=batch.batch_window_ms,
                batch_byte_budget=batch.batch_byte_budget,
                max_in_flight_batches=batch.max_in_flight_batches,
                batch_records=batch.batch_records,
            )
        except BaseException:
            self._calls.deregister_serve(token)
            with contextlib.suppress(Exception):
                self.close()
            raise
        self._calls.deregister_serve(token)
        self.close()

    def _run_local(
        self,
        env_address: str,
        *,
        execution_horizon: int | None = None,
        prefetch_lead: int = 0,
    ) -> dict[str, Any]:
        """Native worker loop against a remote env, for one episode.

        The native default budget, as for :meth:`run` with ``episodes=None``;
        :meth:`_run_local_for_episodes` takes an explicit count.

        Returns the runtime report as ``{"episodes", "telemetry"}``: one
        ``EpisodeResult``-shaped dict per completed episode, plus the session
        metric aggregate (``TelemetryRow``-shaped dicts). Drives vectorized (``num_envs > 1``) envs through the
        native engine: the route resolves at connect (adapter + batched
        predict corners), ``execution_horizon`` (> 1) enables action
        chunking exactly as the served path's ``ResolveAdapter`` pin does, and
        ``prefetch_lead`` (> 0) predicts the next chunk while that many replay
        frames of the current one remain (see :meth:`run`).
        """
        return self._install_worker().run_local(
            env_address, self._execution_horizon(execution_horizon), prefetch_lead
        )

    def _run_local_for_episodes(
        self,
        env_address: str,
        *,
        episodes: int,
        execution_horizon: int | None = None,
        seeds: list[int] | None = None,
        max_episode_steps: int | None = None,
        max_episode_seconds: float | None = None,
        close_env: bool = False,
        trial_index_base: int = 0,
        prefetch_lead: int = 0,
        workflow_edition: str | None = None,
        hooks: object | None = None,
        invite_at: str | None = None,
    ) -> dict[str, Any]:
        """Native worker loop against a remote env for a fixed episode count.

        Returns the report; see :meth:`_run_local` for the shape and what
        ``execution_horizon`` / ``prefetch_lead`` do. ``seeds`` / the episode caps /
        ``trial_index_base`` mirror :meth:`run` (explicit seeds, caps, and a
        non-zero base need runtime-owned resets, i.e. autoreset disabled).
        """
        return self._install_worker().run_local_for_episodes(
            env_address,
            episodes,
            self._execution_horizon(execution_horizon),
            seeds,
            max_episode_steps,
            max_episode_seconds,
            close_env,
            trial_index_base,
            prefetch_lead,
            workflow_edition,
            hooks,
            invite_at,
        )

    def _declared_workflow_edition(self, call: str | None) -> str | None:
        """This model's declaration for one run or session.

        The explicit call keyword, else the process / class / project surfaces
        (see :mod:`rlmesh._editions`).
        """
        from .._editions import resolve_workflow_edition

        return resolve_workflow_edition(
            call=call,
            declared=self._class_workflow_edition(),
            # A subclass or a wrapped policy has a home for the declaration; a
            # wrapped bare callable has none.
            authored=self._policy is not None,
        )

    def _class_workflow_edition(self) -> str | None:
        """The type-level edition declaration this model carries.

        A subclass's own; for a wrapping ``Model(policy)``, the wrapped policy
        type's (``describe`` reads the same one); for a bare callable, the
        callable's attribute.
        """
        if self._policy is self:
            owner: object = type(self)
        elif self._policy is not None:
            owner = type(self._policy)
        else:
            owner = self._edition_source
        declared = getattr(owner, "workflow_edition", None)
        return declared if isinstance(declared, str) else None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def as_model(model: object) -> ModelBase[Any, Any]:
    """Normalize a model source to a built :class:`ModelBase` instance.

    A ``Model`` instance is used as-is; a ``Model`` subclass *class* is instantiated
    once. Anything else is refused: the library never picks a framework for a bare
    predict callable or policy object, so the caller wraps it in the ``Model`` whose
    arrays it expects. (A served ``SandboxModel`` handle binds via
    its own ``.session`` and never reaches here.)
    """
    if isinstance(model, ModelBase):
        return cast("ModelBase[Any, Any]", model)
    if isinstance(model, type) and issubclass(model, ModelBase):
        return cast("ModelBase[Any, Any]", model())
    raise TypeError(
        "rlmesh.run() takes a Model; wrap a predict callable or "
        "policy object in the framework Model whose arrays it expects -- "
        "rlmesh.numpy.Model(fn), rlmesh.torch.Model(fn), rlmesh.jax.Model(fn) -- or "
        "rlmesh.Model(fn) for raw RLMesh values."
    )


def run(
    model: object,
    env: EnvTarget,
    *,
    seeds: Sequence[int] | None = None,
    episodes: int | None = None,
    max_episode_steps: int | None = None,
    max_episode_seconds: float | None = None,
    hooks: HooksArg = None,
    instruction: str | None = None,
    close_env: bool = False,
    trust_entrypoints: bool | None = None,
    execution_horizon: int | None = None,
    prefetch_lead: int = 0,
    view: ViewArg = None,
    workflow_edition: str | None = None,
    trial_index_base: int = 0,
) -> RunResult:
    """Drive ``model`` against ``env`` to completion and return a :class:`RunResult`.

    ``model`` is a :class:`Model` instance or subclass class (see
    :meth:`Model.run` for the parameters), or :data:`rlmesh.RANDOM_SAMPLE` for a
    random baseline that samples the env's action space. A bare predict
    callable or policy object is refused: wrap it in the framework ``Model``
    whose arrays it expects (``rlmesh.numpy.Model(fn)``), the library never
    picks one. A model instance is borrowed and stays usable afterwards; a
    model built here from a subclass class is closed when the call returns.
    """
    from ._eval import RANDOM_SAMPLE

    if _served_handle(model):
        removed(
            "rlmesh.run(SandboxModel(...), env)",
            "SandboxModel(...).run(env), with the sandbox on a session",
        )
    if model is RANDOM_SAMPLE:
        from .._native import Model

        model = Model(RANDOM_SAMPLE)
    local = as_model(model)
    try:
        return local.run(
            cast("LocalEnvTarget", env),
            seeds=seeds,
            episodes=episodes,
            max_episode_steps=max_episode_steps,
            max_episode_seconds=max_episode_seconds,
            hooks=hooks,
            instruction=instruction,
            close_env=close_env,
            trust_entrypoints=trust_entrypoints,
            execution_horizon=execution_horizon,
            prefetch_lead=prefetch_lead,
            workflow_edition=workflow_edition,
            trial_index_base=trial_index_base,
            view=view,
        )
    finally:
        # A model built here is this call's own; an instance is the caller's
        # and stays open.
        if local is not model:
            local.close()


__all__ = ["LifecycleCallback", "ModelBase", "PredictFn", "run"]
