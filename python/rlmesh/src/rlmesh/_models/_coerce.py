"""Model source coercion: any source -> a :class:`CoercedModel`.

Also the ``RANDOM_SAMPLE`` sentinel policy. The Model-rejection guard (a ``Model``
cannot be wrapped as a source -- it builds its own worker) lives in
:meth:`rlmesh._models.base.ModelBase.__init__`, the only construction gateway, so
this module stays free of a back-import to ``base``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple


class _RandomSample:
    """Sentinel policy: act by sampling the env's action space (a random baseline)."""

    def __repr__(self) -> str:
        return "RANDOM_SAMPLE"


RANDOM_SAMPLE = _RandomSample()
"""Pass as the model to :func:`rlmesh.session`/:func:`rlmesh.run` to sample actions."""


class CoercedModel(NamedTuple):
    """A model source normalized to its predict corners, spec, and lifecycle hooks.

    ``on_episode_end`` carries a duck-typed policy's ``reset()``, wired to the
    episode-END edge: the only per-episode boundary both the local loop and the
    served wire path signal, so a stateful policy clears its state identically
    either way. The three optional corners (``predict_chunk`` / ``predict_batch``
    / ``predict_chunk_batch``) are picked up from a duck-typed policy when it
    defines them, so they feed the same corner synthesis a ``Model`` subclass's do.
    """

    predict: Callable[[Any], Any]
    spec: object | None
    on_episode_end: Callable[[], None] | None
    on_close: Callable[[], None] | None
    policy: Any
    predict_chunk: Callable[..., Any] | None = None
    predict_batch: Callable[..., Any] | None = None
    predict_chunk_batch: Callable[..., Any] | None = None


def coerce_model(
    source: Any,
    *,
    spec: object | None,
) -> CoercedModel:
    """Resolve a model source into a :class:`CoercedModel`.

    The source is either a bare predict callable or a duck-typed policy object
    (class or instance) exposing ``predict`` plus optional ``spec``/``reset``/``close``.
    A :class:`~rlmesh._models.base.ModelBase` is rejected at construction
    (``ModelBase.__init__``) before reaching here: a ``Model`` builds its own
    worker, so instantiate the subclass directly rather than wrapping it again.
    """
    from .._bootstrap.loaders import construct_authored_model, looks_like_policy

    # A policy *class* is also callable, so check the policy shape first.
    if looks_like_policy(source):
        inst = construct_authored_model(source)
        return CoercedModel(
            inst.predict,
            spec if spec is not None else getattr(inst, "spec", None),
            getattr(inst, "reset", None),
            getattr(inst, "close", None),
            inst,
            predict_chunk=getattr(inst, "predict_chunk", None),
            predict_batch=getattr(inst, "predict_batch", None),
            predict_chunk_batch=getattr(inst, "predict_chunk_batch", None),
        )
    if callable(source):
        return CoercedModel(source, spec, None, None, None)
    raise TypeError(
        "Model source must be a predict callable or a policy object with predict(); "
        f"got {type(source).__name__}"
    )
