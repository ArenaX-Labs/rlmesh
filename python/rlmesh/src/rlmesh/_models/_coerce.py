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
    predict: Callable[[Any], Any]
    spec: object | None
    # A duck-typed policy's ``reset()`` is wired here, to the episode-END edge: it
    # is the only per-episode boundary both the local loop and the served wire path
    # signal, so a stateful policy clears its state identically either way.
    on_episode_end: Callable[[], None] | None
    on_close: Callable[[], None] | None
    policy: Any


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
        )
    if callable(source):
        return CoercedModel(source, spec, None, None, None)
    raise TypeError(
        "Model source must be a predict callable or a policy object with predict(); "
        f"got {type(source).__name__}"
    )
