"""Authoring base for environments: a thin runtime class (tags/make), NOT a build DSL.

Subclass :class:`EnvFactory` to describe an environment's *runtime*. There is no build
DSL here -- packaging stays in your Dockerfile. Models are authored by subclassing
``rlmesh.Model`` and overriding ``predict`` (no separate recipe noun); envs serve via
:class:`EnvServer`, ``EnvFactory.serve``, or ``python -m rlmesh.serve --env my_pkg:MyEnv``.
"""

from __future__ import annotations

import functools
import inspect
import itertools
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, cast, final

from rlmesh.types import EnvLike

if TYPE_CHECKING:
    from ._value_conversion import ValueBridge
    from .adapters import EnvTags
    from .params import ParamSpec

#: Ceiling on a factory's contract-branch table. Every branch is constructed and
#: tag-normalized once at describe time and lands in the image label, so the
#: product of the declared discriminants' choices is bounded rather than free.
_MAX_TAG_BRANCHES = 8


class EnvFactory(ABC):
    """Authoring base that *builds* environment(s): set ``tags`` and implement ``make``.

    Subclass per obs/action contract -- the ``tags`` (a :class:`~rlmesh.adapters.EnvTags`)
    are that contract. ``make(**kwargs)`` is the factory and may return a single env or a
    vectorized batch; task selection and ``num_envs`` are its parameters, not separate
    subclasses. ``tags = None`` (the default) means a generic, un-adapted env.

    Optionally set ``params`` to a :class:`~rlmesh.ParamSpec` declaring ``make``'s
    construction parameters; a managed dashboard then presents/validates/sweeps
    them, and a bad binding is rejected before construction (see
    :mod:`rlmesh.params`). ``params = None`` (default) keeps today's blind
    passthrough to ``make``.

    Optionally implement an ``enumerate_variants()`` classmethod that returns a
    list of :class:`~rlmesh.Variant` (or ``yield`` them lazily) -- the finite,
    named catalog of concrete sub-environments this factory contains (e.g. one per
    benchmark task). Each
    variant binds only its identity-defining ``params``; the remaining free dials
    stay in ``params`` (the ``ParamSpec`` is always the full validation surface --
    never describe the same dimension in both ``enumerate_variants`` and
    ``enumerate_params``). Import heavy/optional deps lazily inside the method, as
    ``make`` does, so ``describe`` stays off-GPU. See :func:`rlmesh.describe`.
    """

    tags: ClassVar[EnvTags | None] = None
    #: Optional declared construction-parameter surface validated against ``make``.
    params: ClassVar[ParamSpec | None] = None
    #: The declared ``params`` names whose value selects the obs/action contract --
    #: the *contract discriminants*. Each must be a declared :class:`~rlmesh.Param`
    #: with ``choices`` and a ``make`` signature default drawn from them, and the
    #: product of those choices is capped at 8 branches. ``()`` (the default) is
    #: one fixed contract: ``tags``.
    tag_params: ClassVar[tuple[str, ...]] = ()
    #: The reserved reset-option keys this env wants delivered on
    #: ``reset(seed=, options=)``. ``make`` publishes them in the env's metadata,
    #: and the runtime sends a reserved key only to an env that named it -- an env
    #: that forwards ``options`` into a third-party ``reset`` is never handed a key
    #: it cannot interpret. The one key this edition defines is ``"trial_index"``,
    #: the 0-based ordinal of the episode being started (see
    #: :func:`rlmesh.trial_index`); ``()`` (the default) receives none.
    reset_options: ClassVar[tuple[str, ...]] = ()
    #: Framework bridge pinned by a framework-specific subclass
    #: (``rlmesh.torch.EnvFactory`` / ``rlmesh.jax.EnvFactory``); ``serve_env``
    #: reads it to type the served env's obs/action seam. ``None`` serves numpy.
    _bridge: ClassVar[ValueBridge | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Stamp the factory's ``tags`` onto every env ``make()`` returns, so the tag
        # "rides the environment": a locally-made env (no server) still carries the
        # contract a spec'd model resolves against. The subclass's own ``make`` body
        # is untouched; serving an already-stamped env merges the same tags
        # idempotently. ``tags = None`` is a no-op (a generic, un-adapted env).
        super().__init_subclass__(**kwargs)
        _check_tag_params(cls)
        user_make = cls.__dict__.get("make")
        if user_make is None or getattr(user_make, "_rlmesh_tag_stamped", False):
            return

        @functools.wraps(user_make)
        def make(self: EnvFactory, *args: Any, **make_kwargs: Any) -> EnvLike[Any, Any]:
            cls = type(self)
            # Bind BEFORE make(): the branch is read off make's own signature with
            # defaults applied, so a positional or omitted discriminant resolves to
            # the same value the body will see, and a bad one fails pre-construction.
            branch = (
                _tag_binding(cls, user_make, self, args, make_kwargs)
                if cls.tag_params
                else None
            )
            env = user_make(self, *args, **make_kwargs)
            tags = cls.tags if branch is None else cls.tags_for(**branch)
            if tags is not None:
                from .adapters import tag

                # validate=False: the obs/action layout is validated against the
                # tags at adapter-resolution time (serve or session), so a make()
                # of a vectorized batch (whose spaces differ) is not rejected here.
                env = tag(env, tags, validate=False)
            fragment: dict[str, object] = {}
            if branch is not None:
                from .adapters.constants import ENV_BRANCH_METADATA_KEY

                fragment[ENV_BRANCH_METADATA_KEY] = branch
            if cls.reset_options:
                from ._rlmesh import ENV_RESET_OPTIONS_KEY

                fragment[ENV_RESET_OPTIONS_KEY] = list(cls.reset_options)
            if fragment:
                _stamp_metadata(env, fragment)
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

    def prepare(self) -> None:  # noqa: B027  optional no-op hook, not abstract
        """Optional: one-time setup before ``make()``."""

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Return this factory's full metadata envelope (see :func:`rlmesh.describe`)."""
        from ._describe import describe  # lazy: avoid an import cycle at module load

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

    def close(self) -> None:  # noqa: B027  optional no-op hook, not abstract
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
        """Host this env on ``address`` (blocking): ``prepare()`` + ``make(**make_kwargs)``, publish ``tags``.

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


def trial_index(options: Mapping[str, Any] | None) -> int | None:
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
    """
    value = options.get("trial_index") if options is not None else None
    return (
        int(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _stamp_metadata(env: object, fragment: Mapping[str, object]) -> None:
    """Merge a metadata fragment into ``env.metadata`` (mirrors ``adapters.tag``).

    ``metadata`` is typically a class attribute on gymnasium envs; copy-and-assign
    shadows it with an instance attribute rather than mutating every env of that
    class, which is the standard override path.
    """
    existing = getattr(env, "metadata", None)
    merged: dict[str, object] = (
        dict(cast("Mapping[str, object]", existing))
        if isinstance(existing, Mapping)
        else {}
    )
    merged.update(fragment)
    env.metadata = merged  # type: ignore[attr-defined]


def _discriminant(cls: type[EnvFactory], name: str) -> Any:
    """The declared :class:`~rlmesh.Param` for a ``tag_params`` name, or ``None``."""
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
    """Resolve the branch a ``make`` call selects, before the call runs.

    Binds the call against ``make``'s own signature and applies defaults, so an
    omitted or positionally-passed discriminant resolves to exactly the value the
    body will see. The binding (never a subset -- every declared discriminant is
    present) is what rides the env under ``ENV_BRANCH_METADATA_KEY``.
    """
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
    """The factory's contract-branch table: every discriminant binding it declares.

    The Cartesian product of each ``tag_params`` name's declared ``choices``,
    ordered so **index 0 is the default branch** -- the binding ``make``'s own
    signature defaults select, whose ``tags_for`` result is the factory's
    top-level ``tags``. ``tag_params = ()`` has no table (one fixed contract).
    """
    signature = inspect.signature(cls.make).parameters
    choices = {name: tuple(_discriminant(cls, name).choices) for name in cls.tag_params}
    defaults = {name: signature[name].default for name in cls.tag_params}
    branches = [
        dict(zip(choices, values, strict=True))
        for values in itertools.product(*choices.values())
    ]
    branches.sort(key=lambda binding: binding != defaults)
    return branches


def _check_tag_params(cls: type[EnvFactory]) -> None:
    """Reject an unusable contract-branch declaration at class creation.

    A branch table is baked into the image label and read by a platform that
    cannot re-run the author's code, so every way it can be wrong -- an
    undeclared, unenumerable or defaultless discriminant, an unbounded product, a
    ``tags_for`` nothing consults or that answers with the wrong kind of thing, a
    table that is adapted on some branches and generic on others, or a default
    branch that disagrees with the ``tags`` a branch-blind reader sees -- fails
    here, where the author is looking, rather than at push or run time.
    """
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
