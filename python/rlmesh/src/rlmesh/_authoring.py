"""Authoring base for environments: a thin runtime class (tags/make), NOT a build DSL."""

from __future__ import annotations

import functools
import inspect
import itertools
import numbers
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast, final

from rlmesh.types import EnvLike

if TYPE_CHECKING:
    from ._value_conversion import ValueBridge
    from .adapters import EnvTags
    from .params import ParamSpec

#: Cap on the contract-branch table; every branch is constructed once at describe time.
_MAX_TAG_BRANCHES = 8


class EnvFactory(ABC):
    """Authoring base that *builds* environment(s): set ``tags`` and implement ``make``.

    Subclass per obs/action contract -- the ``tags`` (a :class:`~rlmesh.adapters.EnvTags`)
    are that contract. ``make(**kwargs)`` is the factory and may return a single env or a
    vectorized batch; task selection and ``num_envs`` are its parameters, not separate
    subclasses. ``tags = None`` (the default) means a generic, un-adapted env.

    Optionally implement an ``enumerate_variants()`` classmethod that returns a
    list of :class:`~rlmesh.Variant` (or ``yield`` them lazily) -- the finite,
    named catalog of concrete sub-environments this factory contains (e.g. one per
    benchmark task). Each
    variant binds only its identity-defining ``params``; the remaining free dials
    stay in ``params`` (the ``ParamSpec`` is always the full validation surface --
    never describe the same dimension in both ``enumerate_variants`` and
    ``enumerate_params``). Import heavy/optional deps lazily inside the method, as
    ``make`` does, so ``describe`` stays off-GPU. See :func:`rlmesh.describe`.

    Reporting success: the env reports a task outcome in the ``info`` of the step
    that ends an episode (terminated or truncated; earlier steps are not read),
    under ``is_success``, ``success``, or ``task_success``, checked in that order.
    The first key holding a bool, integer, or float decides -- a number counts as
    success when non-zero -- and keys holding any other kind (a string such as
    ``"False"``, ``None``, a list) are skipped. With none of them the outcome is
    unknown, never inferred from ``terminated``: ``success_rate`` counts it as a
    failure, and is ``None`` when no episode reported an outcome.
    """

    tags: ClassVar[EnvTags | None] = None
    #: Optional declared construction-parameter surface validated against ``make``.
    params: ClassVar[ParamSpec | None] = None
    #: ``params`` names whose value selects the obs/action contract. Each must be a declared
    #: :class:`~rlmesh.Param` with ``choices``; the product of choices is capped at 8 branches.
    tag_params: ClassVar[tuple[str, ...]] = ()
    #: Reserved reset-option keys delivered on ``reset(seed=, options=)``. The only key
    #: defined is ``"trial_index"`` (see :func:`rlmesh.trial_index`).
    reset_options: ClassVar[tuple[str, ...]] = ()
    #: The two ``make`` params ``(height, width)`` that set camera render resolution; ``()`` is no dial.
    render_params: ClassVar[tuple[str, ...]] = ()
    #: ``True`` for an env with no episode horizon of its own; a run without
    #: ``max_episode_steps`` is refused before its first reset.
    requires_episode_cap: ClassVar[bool] = False
    #: Workflow edition this env was authored against (see :func:`rlmesh.current_workflow_edition`).
    #: ``None`` runs at the current edition; nothing at deploy time overrides it.
    workflow_edition: ClassVar[str | None] = None
    #: Framework bridge pinned by a framework-specific subclass; ``None`` is numpy.
    _bridge: ClassVar[ValueBridge | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Stamp ``tags`` onto every env ``make()`` returns so a locally made env carries its contract.
        super().__init_subclass__(**kwargs)
        _check_tag_params(cls)
        _check_render_params(cls)
        user_make = cls.__dict__.get("make")
        if user_make is None or getattr(user_make, "_rlmesh_tag_stamped", False):
            return

        @functools.wraps(user_make)
        def make(self: EnvFactory, *args: Any, **make_kwargs: Any) -> EnvLike[Any, Any]:
            cls = type(self)
            # Bind before make(): the branch is read from make's signature with defaults applied.
            branch = (
                _tag_binding(cls, user_make, self, args, make_kwargs)
                if cls.tag_params
                else None
            )
            env = user_make(self, *args, **make_kwargs)
            tags = cls.tags if branch is None else cls.tags_for(**branch)
            if tags is not None:
                from .adapters import tag

                # validate=False: a vectorized make() has batched spaces that differ from the per-lane tags.
                env = tag(env, tags, validate=False)
            fragment: dict[str, object] = {}
            if branch is not None:
                from .adapters.constants import ENV_BRANCH_METADATA_KEY

                fragment[ENV_BRANCH_METADATA_KEY] = branch
            if cls.reset_options:
                from ._rlmesh import ENV_RESET_OPTIONS_KEY

                fragment[ENV_RESET_OPTIONS_KEY] = list(cls.reset_options)
            if cls.render_params:
                from ._rlmesh import ENV_RENDER_PARAMS_KEY

                fragment[ENV_RENDER_PARAMS_KEY] = list(cls.render_params)
            if cls.requires_episode_cap:
                from ._rlmesh import ENV_REQUIRES_EPISODE_CAP_KEY

                fragment[ENV_REQUIRES_EPISODE_CAP_KEY] = True
            if fragment:
                _stamp_metadata(env, fragment)
            if cls._bridge is not None and getattr(env, "_bridge", None) is None:
                try:
                    env._bridge = cls._bridge  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            return env

        make._rlmesh_tag_stamped = True  # type: ignore[attr-defined]
        cls.make = make  # type: ignore[method-assign]

    @classmethod
    def tags_for(cls, **params: Any) -> EnvTags | None:
        """Return the :class:`~rlmesh.adapters.EnvTags` for one discriminant binding.

        Override it alongside ``tag_params`` when the declared discriminants pick
        between *different* contracts (e.g. an ``action_type`` that switches
        end-effector deltas for absolute targets); ``params`` carries every
        declared discriminant, defaults applied. The default returns ``tags``, so
        a ``tag_params`` that only moves the *spaces* needs no override.

        The binding whose values are ``make``'s own signature defaults is the
        **default branch**, and its result must be ``tags`` -- that is the contract
        a reader with no branch information sees.
        """
        return cls.tags

    def prepare(self) -> None:  # noqa: B027
        """Optional: one-time setup before ``make()``."""

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Return this factory's full metadata envelope (see :func:`rlmesh.describe`)."""
        from ._describe import describe

        return describe(cls, kind="env")

    @abstractmethod
    def make(self, **kwargs: Any) -> EnvLike[Any, Any]:
        """Construct and return the environment.

        Your override returns a plain env; the returned env is automatically
        stamped with this factory's ``tags`` (in ``env.metadata``), so the tag
        rides the environment -- a spec'd model can resolve its adapter from the
        env alone, whether it is served or driven locally via
        :func:`rlmesh.session`.
        """
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027
        """Optional: release resources."""

    @final
    def serve(
        self,
        address: str,
        *,
        num_envs: int = 1,
        vectorization_mode: str | None = None,
        framework: str | None = None,
        device: object | None = None,
        **make_kwargs: Any,
    ) -> None:
        """Serve this env at the participant address ``address`` (blocking): ``prepare()`` + ``make(**make_kwargs)``, publish ``tags``.

        The named keywords are *serving* options, forwarded to
        :func:`rlmesh.serve.serve_env` (``num_envs > 1`` fans ``make`` out into a
        vector env; ``framework``/``device`` type and place the served obs/action
        seam); every other keyword goes to ``make``. Naming them here keeps a make
        kwarg from silently binding to a serving option -- a ``make`` parameter that
        shares a serving option's name cannot ride through ``serve``.
        """
        from .serve import serve_env

        serve_env(
            self,
            address,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            framework=framework,
            device=device,
            **make_kwargs,
        )


def trial_index(
    options: Mapping[str, Any] | None, *, lane: int | None = None
) -> int | None:
    """The reserved ``trial_index`` reset option, or ``None`` when absent.

    The ordinal of the episode a ``reset`` starts, 0-based and walked in order by
    the runtime (from ``run(trial_index_base=...)``, 0 by default), so an env
    can sweep a fixed list of initial states / goals exactly as its upstream
    benchmark does instead of re-deriving one from a hashed seed. Delivered only
    to an env that declared ``"trial_index"`` in
    :attr:`EnvFactory.reset_options`; absent on a hand-driven ``Session.reset()``
    that passed none::

        class MyEnv(rlmesh.EnvFactory):
            reset_options = ("trial_index",)


        def reset(self, *, seed=None, options=None):
            trial = rlmesh.trial_index(options)
            n = len(self.init_states)
            state = self.init_states[(trial if trial is not None else seed or 0) % n]

    The option takes one of two forms. A reset covering one lane carries a bare
    integer; a reset covering several lanes of a vector env carries a list, one
    ordinal per lane being reset, in the same order as the reset's seeds. A
    vector env reads its lanes with ``lane=``, the position among the lanes
    this reset covers (not the env index, when only some lanes are reset)::

        for i in range(len(lanes_being_reset)):
            trial = rlmesh.trial_index(options, lane=i)

    Args:
        options: The ``options`` mapping ``reset`` received, or ``None``.
        lane: Position among the lanes this reset covers. ``None`` (the
            default) reads the single-lane form; a bare integer also answers
            ``lane=0``.

    Returns:
        The ordinal, or ``None`` when the key is absent or not an integer.

    Raises:
        ValueError: The option is the multi-lane list and ``lane`` was not
            given, or ``lane`` is outside the lanes the option covers.
    """
    value = options.get("trial_index") if options is not None else None
    if value is None:
        return None
    values: Sequence[object]
    if isinstance(value, (list, tuple)):
        values = cast("Sequence[object]", value)
        if lane is None:
            raise ValueError(
                f"trial_index carries one ordinal per lane ({len(values)} lanes); "
                "pass lane= to read one, e.g. rlmesh.trial_index(options, lane=0)"
            )
    else:
        values = [value]
        lane = 0 if lane is None else lane
    if not 0 <= lane < len(values):
        raise ValueError(
            f"lane={lane} is outside the {len(values)} lane(s) trial_index covers"
        )
    return _as_ordinal(values[lane])


def _as_ordinal(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer() and -(2**63) <= value < 2**63:
        return int(value)
    return None


def _stamp_metadata(env: object, fragment: Mapping[str, object]) -> None:
    existing = getattr(env, "metadata", None)
    merged: dict[str, object] = (
        dict(cast("Mapping[str, object]", existing))
        if isinstance(existing, Mapping)
        else {}
    )
    merged.update(fragment)
    env.metadata = merged  # type: ignore[attr-defined]


def _discriminant(cls: type[EnvFactory], name: str) -> Any:
    spec = cls.params
    for param in spec.params if spec is not None else ():
        if param.name == name:
            return param
    return None


def _tag_binding(
    cls: type[EnvFactory],
    user_make: Any,
    instance: EnvFactory,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the branch a ``make`` call selects, before the call runs."""
    from .params import ParamError

    bound = inspect.signature(user_make).bind(instance, *args, **kwargs)
    bound.apply_defaults()
    binding: dict[str, Any] = {}
    for name in cls.tag_params:
        value = bound.arguments[name]
        choices = _discriminant(cls, name).choices
        if value not in choices:
            raise ParamError(
                f"{cls.__name__}.make() got {name}={value!r}, which is not one of "
                f"the contract discriminant's choices {list(choices)!r}"
            )
        binding[name] = value
    return binding


def _tag_branches(cls: type[EnvFactory]) -> list[dict[str, Any]]:
    signature = inspect.signature(cls.make).parameters
    choices = {name: tuple(_discriminant(cls, name).choices) for name in cls.tag_params}
    defaults = {name: signature[name].default for name in cls.tag_params}
    branches = [
        dict(zip(choices, values, strict=True))
        for values in itertools.product(*choices.values())
    ]
    branches.sort(key=lambda binding: binding != defaults)
    return branches


def _check_render_params(cls: type[EnvFactory]) -> None:
    """Reject a camera dial that is not two distinct ``make`` params."""
    dial = cast("tuple[object, ...]", cls.render_params)
    if not dial:
        return
    if (
        len(dial) != 2
        or len(set(dial)) != 2
        or not all(isinstance(n, str) and n for n in dial)
    ):
        raise TypeError(
            f"{cls.__name__}.render_params must name two distinct make params, "
            f"(height, width); got {dial!r}"
        )
    parameters = inspect.signature(cls.make).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return
    missing = [name for name in dial if name not in parameters]
    if missing:
        raise TypeError(
            f"{cls.__name__}.render_params names {missing}, which make() does not take"
        )


def render_binding(source: object, size: tuple[int, int] | None) -> dict[str, int]:
    """``{height param: h, width param: w}`` for an env source's camera dial.

    ``{}`` when ``size`` is ``None`` or the source declares no
    :attr:`EnvFactory.render_params`.
    """
    dial = cast("tuple[str, ...]", getattr(source, "render_params", ()) or ())
    if size is None or len(dial) != 2:
        return {}
    return {dial[0]: int(size[0]), dial[1]: int(size[1])}


def _check_tag_params(cls: type[EnvFactory]) -> None:
    overrides_tags_for = cls.tags_for.__func__ is not EnvFactory.tags_for.__func__  # type: ignore[attr-defined]
    if not cls.tag_params:
        if overrides_tags_for:
            raise TypeError(
                f"{cls.__name__} overrides tags_for() but declares no tag_params, "
                "so nothing would ever call it; declare the discriminants "
                "(tag_params = ('action_type',)) or drop the override"
            )
        return

    signature = inspect.signature(cls.make).parameters
    for name in cls.tag_params:
        param = _discriminant(cls, name)
        if param is None:
            raise TypeError(
                f"{cls.__name__}.tag_params names {name!r}, which is not declared "
                "in params; a contract discriminant must be a declared Param"
            )
        if not param.choices:
            raise TypeError(
                f"{cls.__name__}.tag_params names {name!r}, whose Param declares "
                "no choices; a contract discriminant must be enumerable so the "
                "branch table is finite"
            )
        default = (
            signature[name].default if name in signature else inspect.Parameter.empty
        )
        if default is inspect.Parameter.empty or default not in param.choices:
            raise TypeError(
                f"{cls.__name__}.make() must give the contract discriminant "
                f"{name!r} a signature default drawn from its choices "
                f"{list(param.choices)!r}; the default branch is the contract a "
                "reader with no branch information sees"
            )

    branches = _tag_branches(cls)
    if len(branches) > _MAX_TAG_BRANCHES:
        raise TypeError(
            f"{cls.__name__}.tag_params declares {len(branches)} contract branches, "
            f"over the limit of {_MAX_TAG_BRANCHES}; discriminants are contract "
            "axes, not shape dials -- move the free knobs back to plain params"
        )

    from .adapters import EnvTags

    tags: list[Any] = [cls.tags_for(**binding) for binding in branches]
    for binding, branch_tags in zip(branches, tags, strict=True):
        if branch_tags is not None and not isinstance(branch_tags, EnvTags):
            raise TypeError(
                f"{cls.__name__}.tags_for({_kwargs_repr(binding)}) returned "
                f"{type(branch_tags).__name__}, not EnvTags or None"
            )
    if any(t is None for t in tags) and any(t is not None for t in tags):
        raise TypeError(
            f"{cls.__name__}.tags_for() returns EnvTags on some branches and None "
            "on others; a factory is adapted or generic, not both -- a "
            "branch-blind reader cannot tell which one it got"
        )
    if tags[0] != cls.tags:
        raise TypeError(
            f"{cls.__name__}.tags disagrees with tags_for({_kwargs_repr(branches[0])}), "
            "its default branch; they are the same contract read two ways, so set "
            "tags to what the default branch returns"
        )


def _kwargs_repr(binding: Mapping[str, Any]) -> str:
    return ", ".join(f"{name}={value!r}" for name, value in binding.items())
