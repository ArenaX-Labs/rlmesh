"""The three-phase environment Recipe schema (inert, JSON-round-trippable data).

A recipe is **inert data**: no callables anywhere, JSON-round-trippable, and
derivable to a Dockerfile by a language-neutral core with zero Python executed.
A non-serializable field would poison the sandbox / cross-language path, so the
invariant is enforced *eagerly* at construction (``__post_init__``) rather than
at the wire: you can never construct a ``Recipe`` that will not serialize and
render safely.

The three phases are read top to bottom:

* **make** (phase 3) -- the named factory (``gym`` | ``py`` | ``hf``).
* **build** (phase 1) -- derives a Dockerfile; shell/apt/git are allowed here
  and *only* here, because this phase runs exclusively inside ``docker build``.
* **setup** (phase 2) -- construct-time DATA only (env mutation + file writes);
  the safety boundary, because it runs in your process.

Canonical parse/serialize is single-implemented in the Rust serde core; these
dataclasses are typed views with identical field names and JSON shape. The
``to_json``/``from_json`` here emit and accept that canonical shape (snake_case
keys, a ``kind``-tagged ``make`` union).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Final, Literal, cast

if TYPE_CHECKING:
    # Typing-only: the core never depends on the adapters layer at runtime
    # (``from __future__ import annotations`` keeps the annotation a string).
    from rlmesh.adapters import EnvTags, ModelSpec

__all__ = [
    "ArtifactInput",
    "Build",
    "Fetch",
    "FileWrite",
    "GymMake",
    "HfMake",
    "Make",
    "PipInstall",
    "ProjectInstall",
    "PyMake",
    "Recipe",
    "RecipeKind",
    "RecipeValidationError",
    "Requires",
    "RuntimeReserved",
    "Setup",
]

# The recipe schema version. New fields default, so a document missing them still
# loads; the version is a soft signal for readers, not a hard gate.
RECIPE_VERSION: Final = 1

# The single discriminator that distinguishes an env recipe document from a model
# recipe document. Selects the container ENTRYPOINT/bootstrap and gates a small set
# of cross-kind validations. Deliberately excluded from build_hash.
RecipeKind = Literal["env", "model"]


def _empty_json_map() -> dict[str, object]:
    return {}


def _empty_str_map() -> dict[str, str]:
    return {}


class RecipeValidationError(ValueError):
    """Raised when a recipe is constructed with an unserializable or unsafe field.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers keep
    working; the dedicated type lets callers distinguish recipe-shape failures
    from other value errors.
    """


# Per-field allowlists, each with its own legal charset. The *primary* safety
# boundary is exec-form argument passing in the renderer; these allowlists are
# belt-and-suspenders that reject shell metacharacters, newlines, and option
# injection (a leading "-") at construction time.

_APT_NAME: Final = re.compile(r"^[a-z0-9][a-z0-9+.\-]*$")
_GIT_REF: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]*$")
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_URL: Final = re.compile(r"^https?://[^\s'\"\\]+$")
_POSIX_PATH: Final = re.compile(r"^[A-Za-z0-9._/\-]+$")
# A relative path glob for ProjectInstall.include: like a POSIX path but also
# permits the only wildcards the Rust include matcher implements -- '*' and '**'.
# '..' rides on the '.' already in the charset (the renderer enforces the
# context_root boundary). The other glob metacharacters ('?', '[', ']', '{', '}')
# are deliberately *rejected*: the matcher treats them as literals, so an entry
# like 'file?.json' would pass check() then silently match nothing. Shell
# metacharacters, whitespace, and a leading '/' are excluded too.
_INCLUDE_GLOB: Final = re.compile(r"^[A-Za-z0-9._/\-*]+$")
_ENV_NAME: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# The name half of "namespace/name"; '@' is reserved for @variant addressing.
_RECIPE_NAME: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]*$")
# A gymnasium env id. Unlike a recipe name it may contain ':' (the
# ``module:Name-vN`` load form, e.g. ``sai_pygame:SquidHunt-v0``), so
# ``rlmesh.make`` stays a strict superset of ``gymnasium.make``.
_GYM_ENV_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:\-]*$")


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise RecipeValidationError(
            f"{field_name} must be a str, got {type(value).__name__}"
        )
    return value


def _check_token(value: str, pattern: re.Pattern[str], field_name: str) -> None:
    # fullmatch, not match: with a `$`-anchored pattern, match() accepts a value
    # with a trailing newline (`$` matches before the final \n), which then breaks
    # the Dockerfile line / recipe name at the Rust boundary. fullmatch requires
    # the whole string to match, so a trailing \n (or any control char) is rejected.
    if not pattern.fullmatch(value):
        raise RecipeValidationError(
            f"{field_name} {value!r} is not a valid {field_name} token"
        )


def _check_apt_name(value: str, field_name: str) -> None:
    _check_token(value, _APT_NAME, field_name)


def _check_include_glob(value: str, field_name: str) -> None:
    """Validate a ProjectInstall.include glob entry (spec 7.1; staged by the deriver).

    Each entry is a glob relative to the project root that may use ``..`` to reach
    siblings (e.g. ``../assets/**``); the Rust deriver enforces the context_root
    boundary at staging time. The only supported wildcards are ``*`` and ``**`` --
    the Rust include matcher implements just those and treats ``?``/``[``/``]``/
    ``{``/``}`` as *literals*, so those metacharacters are rejected here rather than
    let an entry pass check() and then silently match nothing. The load-bearing
    checks are: non-empty, no absolute path (leading ``/``), and only path/``*``
    tokens -- no other glob metacharacters, shell metacharacters, whitespace, or
    control characters.
    """
    if not value:
        raise RecipeValidationError(f"{field_name} entry must be non-empty")
    if value[0] == "/":
        raise RecipeValidationError(
            f"{field_name} {value!r} must be a relative glob, not an absolute path"
        )
    _check_token(value, _INCLUDE_GLOB, field_name)


def _check_pip_package(value: str, field_name: str) -> None:
    """Validate a pip requirement string (PEP 508-ish, exec-form safe).

    Full PEP 508 is brittle to re-derive; the load-bearing checks are: non-empty,
    no newlines/control characters, and no leading ``-`` (which would smuggle a
    ``pip`` option such as ``--index-url`` past the package list).
    """
    if not value or value[0] == "-":
        raise RecipeValidationError(
            f"{field_name} {value!r} must be a package spec, not empty or an option"
        )
    if any(ch in value for ch in "\n\r\x00"):
        raise RecipeValidationError(
            f"{field_name} {value!r} must not contain newlines or null bytes"
        )


# ``kwargs`` are inherited by the Rust serde boundary as serde_json::Value, so
# they must be JSON-only. Scalars are checked by *exact* type (not isinstance) so
# numpy scalars -- np.float64 subclasses float -- are rejected, not silently cast.

_JSON_SCALARS: Final = (bool, int, float, str)


def _clean_json(value: object, path: str) -> object:
    """Validate a value is JSON-only and return a canonical copy.

    Tuples become lists and mappings become plain dicts so that
    ``from_json(to_json(recipe)) == recipe`` holds regardless of the container
    types the author passed in.
    """
    if type(value) is float and not math.isfinite(value):
        # float('nan')/float('inf') survive construction and json.dumps emits them as
        # the NaN/Infinity tokens, which Python's json.loads accepts but the Rust
        # serde_json boundary REJECTS -- a check()-passes-then-fails-at-launch trap.
        # Reject at construction (`is float` so bool/int are unaffected; numpy scalars
        # already fall through to the non-JSON rejection below).
        raise RecipeValidationError(
            f"{path}: non-finite floats (NaN/Infinity) are not valid JSON; "
            f"got {value!r}"
        )
    if value is None or type(value) in _JSON_SCALARS:
        return value
    if isinstance(value, Mapping):
        cleaned: dict[str, object] = {}
        for key, item in cast("Mapping[object, object]", value).items():
            if not isinstance(key, str):
                raise RecipeValidationError(
                    f"{path}: kwargs mapping keys must be str, got {type(key).__name__}"
                )
            cleaned[key] = _clean_json(item, f"{path}.{key}")
        return cleaned
    if isinstance(value, (list, tuple)):
        items = cast("Sequence[object]", value)
        return [
            _clean_json(item, f"{path}[{index}]") for index, item in enumerate(items)
        ]
    raise RecipeValidationError(
        f"{path}: kwargs must be JSON-only (str/int/float/bool/None/list/dict), "
        f"got {type(value).__name__}"
    )


def _clean_json_kwargs(value: Mapping[str, object], path: str) -> dict[str, object]:
    cleaned = _clean_json(dict(value), path)
    # _clean_json on a Mapping always returns a dict; narrow for the type checker.
    assert isinstance(cleaned, dict)
    return cast("dict[str, object]", cleaned)


def _clean_str_map(value: Mapping[str, str], field_name: str) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, item in value.items():
        name = _require_str(key, f"{field_name} key")
        _check_token(name, _ENV_NAME, f"{field_name} key")
        cleaned[name] = _require_str(item, f"{field_name}[{name}]")
    return cleaned


def _as_str_tuple(value: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise RecipeValidationError(
            f"{field_name} expects a sequence of strings, not a bare str; "
            f"pass [{value!r}] for a single entry"
        )
    return tuple(_require_str(item, f"{field_name}[]") for item in value)


@dataclass(frozen=True)
class GymMake:
    """A ``gymnasium.make`` / ``gym.make`` factory -- the 90% case.

    Covers any id constructible via ``gymnasium.make``: a built-in like
    ``CartPole-v1``, or one that a ``requires.imports`` package registers into the
    Gymnasium registry on import (e.g. ``ale_py`` for Atari). For an environment
    with its *own* ``make`` or one that needs a wrapper (e.g. ``safety_gymnasium``),
    or any custom construction, use :class:`PyMake` or ``rlmesh.EnvRecipe`` instead.
    """

    env_id: str
    kwargs: Mapping[str, object] = field(default_factory=_empty_json_map)
    kind: Final = "gym"

    def __post_init__(self) -> None:
        """Validate the env id and JSON-only kwargs."""
        _check_token(
            _require_str(self.env_id, "GymMake.env_id"), _GYM_ENV_ID, "GymMake.env_id"
        )
        object.__setattr__(
            self, "kwargs", _clean_json_kwargs(self.kwargs, "GymMake.kwargs")
        )


@dataclass(frozen=True)
class PyMake:
    """A ``module:callable`` Python factory referenced by string.

    The right tool whenever ``gymnasium.make`` is not (an env with its own ``make``
    plus a wrapper like ``safety_gymnasium``, or heavy construction like LIBERO /
    Isaac). The callable can be a function or a class (its ``__init__`` is the
    initializer). The factory *body* is the sole import sequencer; the envelope
    never pre-runs ``requires.imports`` for a py recipe (see ``Recipe`` validation).
    For the cohesive class form (factory + build + ``prepare()`` together), subclass
    ``rlmesh.EnvRecipe``, which projects to a ``PyMake`` recipe.
    """

    entrypoint: str
    kwargs: Mapping[str, object] = field(default_factory=_empty_json_map)
    kind: Final = "py"

    def __post_init__(self) -> None:
        """Validate the ``module:callable`` entrypoint and JSON-only kwargs."""
        from rlmesh._bootstrap.entrypoint import parse_entrypoint

        entrypoint = _require_str(self.entrypoint, "PyMake.entrypoint")
        # Use the canonical parser so the same malformed shapes the loader rejects
        # (``mod:``, ``:fn``, ``mod:fn.``) fail at construction, not at image build.
        try:
            parse_entrypoint(entrypoint, label="PyMake.entrypoint")
        except ValueError as exc:
            raise RecipeValidationError(str(exc)) from exc
        object.__setattr__(
            self, "kwargs", _clean_json_kwargs(self.kwargs, "PyMake.kwargs")
        )


@dataclass(frozen=True)
class HfMake:
    """A Hugging Face-materialized factory.

    ``revision`` pins a full 40-char SHA by default (the ``allow_unpinned_hf``
    gate relaxes it).
    """

    repo: str
    revision: str | None = None
    suite: str | None = None
    task: str | None = None
    kwargs: Mapping[str, object] = field(default_factory=_empty_json_map)
    kind: Final = "hf"

    def __post_init__(self) -> None:
        """Validate the repo / revision and JSON-only kwargs."""
        _require_str(self.repo, "HfMake.repo")
        if self.revision is not None:
            _check_token(
                _require_str(self.revision, "HfMake.revision"),
                _GIT_REF,
                "HfMake.revision",
            )
        object.__setattr__(
            self, "kwargs", _clean_json_kwargs(self.kwargs, "HfMake.kwargs")
        )


Make = GymMake | PyMake | HfMake


@dataclass(frozen=True)
class PipInstall:
    """One ``pip install`` step with its own index URLs.

    The per-step model is the only faithful transcription of pyproject's
    per-package ``[tool.uv.sources]``: torch+torchvision via one step pinned to a
    cu118 ``--index-url``, isaacsim via another pinned to ``pypi.nvidia.com``,
    pure-PyPI deps in a third. ``packages`` accepts bare strings for the common
    case.
    """

    packages: Sequence[str]
    index_url: str | None = None
    extra_index_urls: Sequence[str] = ()
    no_deps: bool = False
    pre: bool = False
    requirements: str | None = None

    def __post_init__(self) -> None:
        """Normalize sequences and validate package specs and index URLs."""
        packages = _as_str_tuple(self.packages, "PipInstall.packages")
        if not packages:
            raise RecipeValidationError("PipInstall.packages must be non-empty")
        for package in packages:
            _check_pip_package(package, "PipInstall.packages")
        object.__setattr__(self, "packages", packages)

        extras = _as_str_tuple(self.extra_index_urls, "PipInstall.extra_index_urls")
        for url in extras:
            _check_token(url, _URL, "PipInstall.extra_index_urls")
        object.__setattr__(self, "extra_index_urls", extras)

        if self.index_url is not None:
            _check_token(
                _require_str(self.index_url, "PipInstall.index_url"),
                _URL,
                "PipInstall.index_url",
            )
        if self.requirements is not None:
            _check_token(
                _require_str(self.requirements, "PipInstall.requirements"),
                _POSIX_PATH,
                "PipInstall.requirements",
            )


@dataclass(frozen=True)
class Fetch:
    """A third-party build-time acquisition: a git clone or url download.

    Confined to ``docker build``; never runs in-process. Pinned by ``ref`` (git)
    or ``sha256`` (url). Installing the recipe *author's own* tree is
    ``ProjectInstall``, not ``Fetch``.
    """

    kind: str
    repo: str | None = None
    ref: str | None = None
    dest: str = ""
    pip_install: bool = False
    pip_requirements: str | None = None
    url: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate the fetch kind and its pinning/destination tokens."""
        kind = _require_str(self.kind, "Fetch.kind")
        if kind not in ("git", "url"):
            raise RecipeValidationError(
                f"Fetch.kind must be 'git' or 'url', got {kind!r}"
            )
        if kind == "git":
            if self.repo is None:
                raise RecipeValidationError("Fetch(kind='git') requires repo=")
            _check_token(_require_str(self.repo, "Fetch.repo"), _URL, "Fetch.repo")
            if self.ref is not None:
                _check_token(_require_str(self.ref, "Fetch.ref"), _GIT_REF, "Fetch.ref")
        else:
            if self.url is None:
                raise RecipeValidationError("Fetch(kind='url') requires url=")
            _check_token(_require_str(self.url, "Fetch.url"), _URL, "Fetch.url")
            if self.sha256 is not None:
                _check_token(
                    _require_str(self.sha256, "Fetch.sha256"), _SHA256, "Fetch.sha256"
                )
        # dest is required for both kinds: the deriver clones/downloads INTO it,
        # and an empty dest fails the Rust deriver (MissingField) only at build
        # time -- reject it eagerly here instead.
        dest = _require_str(self.dest, "Fetch.dest")
        if not dest:
            raise RecipeValidationError(
                f"Fetch(kind={kind!r}) requires a non-empty dest="
            )
        _check_token(dest, _POSIX_PATH, "Fetch.dest")
        if self.pip_requirements is not None:
            _check_token(
                _require_str(self.pip_requirements, "Fetch.pip_requirements"),
                _POSIX_PATH,
                "Fetch.pip_requirements",
            )


@dataclass(frozen=True)
class ProjectInstall:
    """Install the recipe *author's own* package source tree, editable.

    Carries the package's non-code assets and preserves parent-dir layout so
    ``__file__``-relative paths (the Isaac/USD ``Path(__file__).parent/'../../assets'``
    pattern) resolve at construct time. Rejected for Remote provenance (no host
    tree to read).
    """

    src: str = "."
    dest: str = ""
    editable: bool = True
    include: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Validate the source/destination paths and include globs."""
        src = _require_str(self.src, "ProjectInstall.src")
        if src != "." and not _POSIX_PATH.fullmatch(src):
            raise RecipeValidationError(
                f"ProjectInstall.src {src!r} is not a valid path"
            )
        if self.dest:
            _check_token(
                _require_str(self.dest, "ProjectInstall.dest"),
                _POSIX_PATH,
                "ProjectInstall.dest",
            )
        include = _as_str_tuple(self.include, "ProjectInstall.include")
        for entry in include:
            _check_include_glob(entry, "ProjectInstall.include")
        object.__setattr__(self, "include", include)


@dataclass(frozen=True)
class Build:
    """Phase 1 -- the typed Dockerfile.

    An empty ``Build()`` reproduces today's OSS behavior (base + pip rlmesh +
    gymnasium). Every field renders to a Dockerfile instruction, *or*
    ``dockerfile`` supplies the body verbatim (the strict-superset-of-a-Dockerfile
    trapdoor). The full ``FROM``-chain is derivable from the document alone.

    The default renderer installs ``system``/``system_runtime`` with **apt**, so a
    structured build targets a **Debian/Ubuntu** base (the defaults --
    ``python:3.11-slim`` and the ``nvidia/cuda`` images -- are). For another distro,
    set ``dockerfile`` to a verbatim Dockerfile, or use ``commands``.
    """

    base: str | None = None
    from_recipe: str | None = None
    system: Sequence[str] = ()
    system_runtime: Sequence[str] = ()
    pip: Sequence[PipInstall] = ()
    project: ProjectInstall | None = None
    fetch: Sequence[Fetch] = ()
    env: Mapping[str, str] = field(default_factory=_empty_str_map)
    pythonpath: Sequence[str] = ()
    gpu: bool = False
    installer: str = "pip"
    run_as: int | None = None
    toolchain: bool | None = None
    commands: Sequence[str] = ()
    dockerfile: str | None = None

    def __post_init__(self) -> None:
        """Normalize sequences and enforce mutual-exclusivity + token rules."""
        if self.base is not None and self.from_recipe is not None:
            raise RecipeValidationError(
                "Build.base and Build.from_recipe are mutually exclusive"
            )
        if self.base is not None:
            _require_str(self.base, "Build.base")
        if self.from_recipe is not None:
            _check_token(
                _require_str(self.from_recipe, "Build.from_recipe"),
                _RECIPE_NAME,
                "Build.from_recipe",
            )
        if self.installer not in ("pip", "uv"):
            raise RecipeValidationError(
                f"Build.installer must be 'pip' or 'uv', got {self.installer!r}"
            )

        system = _as_str_tuple(self.system, "Build.system")
        runtime = _as_str_tuple(self.system_runtime, "Build.system_runtime")
        for name in (*system, *runtime):
            _check_apt_name(name, "Build.system")
        object.__setattr__(self, "system", system)
        object.__setattr__(self, "system_runtime", runtime)

        pythonpath = _as_str_tuple(self.pythonpath, "Build.pythonpath")
        for entry in pythonpath:
            _check_token(entry, _POSIX_PATH, "Build.pythonpath")
        object.__setattr__(self, "pythonpath", pythonpath)

        object.__setattr__(self, "pip", tuple(self.pip))
        object.__setattr__(self, "fetch", tuple(self.fetch))
        object.__setattr__(
            self, "commands", _as_str_tuple(self.commands, "Build.commands")
        )
        object.__setattr__(self, "env", _clean_str_map(self.env, "Build.env"))

        # The verbatim-Dockerfile trapdoor is mutually exclusive with
        # every field that only affects the *derived* Dockerfile. The Rust deriver's
        # verbatim trapdoor emits the body as-is and IGNORES the resolved base_image
        # and installer, so base/installer/env/pythonpath/run_as would be silently
        # dropped just like the structured build steps -- pairing dockerfile with
        # base=... would even build a different FROM than the hash/wheel-compat were
        # computed against. ``gpu`` is the lone exception: it independently drives
        # the runtime --gpus flag, so a verbatim Dockerfile legitimately pairs with
        # gpu=True.
        if self.dockerfile is not None:
            _require_str(self.dockerfile, "Build.dockerfile")
            structured = (
                system,
                runtime,
                self.pip,
                self.fetch,
                self.commands,
                self.project,
                self.env,
                pythonpath,
            )
            if (
                any(structured)
                or self.base is not None
                or self.from_recipe is not None
                or self.run_as is not None
                or self.installer != "pip"
            ):
                raise RecipeValidationError(
                    "Build.dockerfile is mutually exclusive with structured build "
                    "fields (base/system/pip/fetch/project/commands/from_recipe/env/"
                    "pythonpath/run_as/installer); put those directives in the "
                    "verbatim Dockerfile body"
                )


@dataclass(frozen=True)
class FileWrite:
    """A construct-time file write.

    In-container the path is unrestricted; in-process it is tempdir-only (the
    safety boundary).
    """

    path: str
    contents: str
    if_absent: bool = False

    def __post_init__(self) -> None:
        """Validate the path and contents are strings."""
        _require_str(self.path, "FileWrite.path")
        _require_str(self.contents, "FileWrite.contents")


@dataclass(frozen=True)
class Setup:
    """Construct-time DATA: ``os.environ`` updates plus file writes.

    Applied before ``requires.imports`` (gym/hf only).

    Per-construction parameters for env-var-driven legacy envs (LIBERO's
    parameterless ``class Environment`` that reads ``LIBERO_TASK``) ride
    ``setup.env`` -- not ``make.kwargs`` -- so they do not invalidate ``build_hash``.
    ``setup.env`` is *not* isolation-safe in-process (constructed envs read vars
    lazily); the sandbox is the blessed isolation path.
    """

    env: Mapping[str, str] = field(default_factory=_empty_str_map)
    files: Sequence[FileWrite] = ()
    # Allowlist of setup.env keys a member may override at runtime via
    # RLMESH_PARAMS_JSON; build-hash-excluded, so one image serves every member.
    params: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Validate env var names / params and normalize the file-write sequence."""
        object.__setattr__(self, "env", _clean_str_map(self.env, "Setup.env"))
        object.__setattr__(self, "files", tuple(self.files))
        params = _as_str_tuple(self.params, "Setup.params")
        for name in params:
            _check_token(name, _ENV_NAME, "Setup.params[]")
        object.__setattr__(self, "params", params)


@dataclass(frozen=True)
class Requires:
    """Registration imports, run *before* ``make`` for ``GymMake``/``HfMake`` only.

    For ``PyMake`` this field is forbidden (spec 7.1D): the factory body owns its
    own import sequence, so a non-empty ``imports`` would be a silently-ignored
    lie. There is no ``requires.packages`` -- the single dependency surface is
    ``build.pip``.
    """

    imports: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Normalize the imports sequence to a tuple of strings."""
        object.__setattr__(
            self, "imports", _as_str_tuple(self.imports, "Requires.imports")
        )


_ARTIFACT_URI_SCHEMES: Final = (
    "hf://",
    "gs://",
    "s3://",
    "https://",
    "http://",
    "file://",
)


@dataclass(frozen=True)
class ArtifactInput:
    """A runtime weight/asset mount for a model recipe.

    Weights are ALWAYS a runtime mount -- never ``build.fetch``, never baked into the
    image. ``build_hash`` excludes runtime params, so one image serves every checkpoint.
    Resolve the mounted path inside ``load()`` via ``ModelRecipe.input_path(name)``.

    A ``uri`` is fetched by the rlmesh artifact resolver into a content-addressed cache
    (root ``$RLMESH_CACHE_DIR``, default ``~/.cache/rlmesh/artifacts``); ``local_dir``
    overrides with an explicit host dir for the local (non-sandbox) path. ``None`` uri
    means the mount is bound at run time (a ``SandboxModel``/``Model`` launch arg).
    """

    name: str
    target_path: str
    uri: str | None = None
    local_dir: str | None = None
    include: Sequence[str] = ()
    required: bool = True

    def __post_init__(self) -> None:
        """Validate the handle name, mount path, uri scheme, and include globs."""
        _check_token(
            _require_str(self.name, "ArtifactInput.name"),
            _RECIPE_NAME,
            "ArtifactInput.name",
        )
        _check_token(
            _require_str(self.target_path, "ArtifactInput.target_path"),
            _POSIX_PATH,
            "ArtifactInput.target_path",
        )
        if self.uri is not None:
            uri = _require_str(self.uri, "ArtifactInput.uri")
            if not uri.startswith(_ARTIFACT_URI_SCHEMES):
                raise RecipeValidationError(
                    f"ArtifactInput.uri {uri!r} must use one of {_ARTIFACT_URI_SCHEMES}"
                )
        includes = _as_str_tuple(self.include, "ArtifactInput.include")
        for glob in includes:
            _check_include_glob(glob, "ArtifactInput.include")
        object.__setattr__(self, "include", includes)


@dataclass(frozen=True)
class RuntimeReserved:
    """Reserved, inert home for every deferred feature.

    Every default is ``None`` (a no-op today); populating any field later is additive
    and excluded from ``build_hash``. Serializes to absent/null when empty, so existing
    env recipe JSON and build hashes stay byte-identical. Lives on ``Recipe.runtime``.
    """

    # action chunking -- a MODEL fact vs a pinned eval knob vs the loop mode
    chunk_size: int | None = None
    execute_horizon: int | None = None
    loop_mode: Literal["step", "chunk", "receding", "open_loop"] | None = None
    # batching / scheduling -- opt-in is the industry standard
    batching: Literal["off", "utilization", "fusion"] | None = None
    max_batch: int | None = None
    # eval-determinism mode
    determinism: Literal["off", "seeded", "strict"] | None = None
    # multi-modal perturbation taxonomy (a JSON bag until/unless it earns a schema)
    perturbation: Mapping[str, object] | None = None
    # per-lane stateful-adapter affinity
    lane_affinity: bool | None = None

    def __post_init__(self) -> None:
        """Clean the perturbation bag to JSON scalars (the only non-scalar field)."""
        if self.perturbation is not None:
            object.__setattr__(
                self,
                "perturbation",
                _clean_json_kwargs(self.perturbation, "RuntimeReserved.perturbation"),
            )

    def is_empty(self) -> bool:
        """True when every field is ``None`` (the default, inert state)."""
        return all(getattr(self, f.name) is None for f in fields(self))

    def to_dict(self) -> dict[str, object] | None:
        """JSON form, or ``None`` when empty (so the recipe envelope omits the key)."""
        if self.is_empty():
            return None
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> RuntimeReserved:
        """Rehydrate from JSON, ignoring unknown keys (forward-compat)."""
        if not data:
            return cls()
        loop_mode = data.get("loop_mode")
        if loop_mode is not None and loop_mode not in (
            "step",
            "chunk",
            "receding",
            "open_loop",
        ):
            raise RecipeValidationError(
                f"RuntimeReserved.loop_mode invalid: {loop_mode!r}"
            )
        batching = data.get("batching")
        if batching is not None and batching not in ("off", "utilization", "fusion"):
            raise RecipeValidationError(
                f"RuntimeReserved.batching invalid: {batching!r}"
            )
        determinism = data.get("determinism")
        if determinism is not None and determinism not in ("off", "seeded", "strict"):
            raise RecipeValidationError(
                f"RuntimeReserved.determinism invalid: {determinism!r}"
            )
        return cls(
            chunk_size=_opt_int(data.get("chunk_size"), "RuntimeReserved.chunk_size"),
            execute_horizon=_opt_int(
                data.get("execute_horizon"), "RuntimeReserved.execute_horizon"
            ),
            loop_mode=loop_mode,
            batching=batching,
            max_batch=_opt_int(data.get("max_batch"), "RuntimeReserved.max_batch"),
            determinism=determinism,
            perturbation=_opt_map(data.get("perturbation")),
            lane_affinity=_opt_bool(
                data.get("lane_affinity"), "RuntimeReserved.lane_affinity"
            ),
        )


@dataclass(frozen=True)
class Recipe:
    """An inert environment recipe.

    No callables, JSON-round-trippable, and derivable to a Dockerfile by a
    language-neutral core with zero Python executed.

    ``make=None`` is a build-only base (the honest shape for ``from_recipe`` reuse);
    constructing such a recipe directly is rejected by ``rlmesh.make``/``build()``.
    """

    name: str
    make: Make | None = None
    build: Build = field(default_factory=Build)
    setup: Setup = field(default_factory=Setup)
    requires: Requires = field(default_factory=Requires)
    summary: str | None = None
    # The published adapter content: an env recipe's EnvTags or a model recipe's
    # ModelSpec (or a raw JSON Mapping after from_dict). A dataclass instance is
    # accepted at construction and lazily type-checked against ``kind`` in
    # __post_init__; serde flattens it to a bare JSON dict.
    adapter: EnvTags | ModelSpec | Mapping[str, object] | None = None
    recipe_version: int = RECIPE_VERSION
    # Appended, keyword-friendly, defaulted; never inserted mid-list so positional
    # construction + the Rust serde golden order stay stable.
    kind: RecipeKind = "env"
    inputs: tuple[ArtifactInput, ...] = ()
    runtime: RuntimeReserved = field(default_factory=RuntimeReserved)

    def __post_init__(self) -> None:
        """Validate the name and enforce the cross-cutting PyMake import rule."""
        name = _require_str(self.name, "Recipe.name")
        if "@" in name:
            raise RecipeValidationError(
                f"Recipe.name {name!r} must not contain '@' (reserved for @variant)"
            )
        _check_token(name, _RECIPE_NAME, "Recipe.name")

        # requires.imports is a hard error for PyMake.
        if isinstance(self.make, PyMake) and self.requires.imports:
            raise RecipeValidationError(
                "requires.imports is forbidden for PyMake; the py factory body owns "
                "its own import sequence (spec 7.1D)"
            )

        # Cross-kind rules.
        if self.kind == "model" and not (
            self.make is None or isinstance(self.make, PyMake)
        ):
            raise RecipeValidationError(
                "a model recipe's make must be a PyMake (to ModelRecipe._rlmesh_load) "
                "or None; gym/hf factories are env-only"
            )
        # inputs (runtime artifact mounts) are kind-agnostic -- weights for a model,
        # assets for an env -- but only an authored (PyMake/None) recipe has an
        # input_path to resolve them; a gym/hf SOURCE env cannot, so reject them there.
        if self.inputs and not (self.make is None or isinstance(self.make, PyMake)):
            raise RecipeValidationError(
                "Recipe.inputs require an authored (PyMake) recipe; a gym/hf source "
                "env has no input_path to resolve them"
            )
        object.__setattr__(self, "inputs", tuple(self.inputs))

        # adapter: a raw Mapping is JSON-cleaned; a dataclass instance is type-checked
        # against the kind (and serialized to a bare dict later).
        adapter = self.adapter
        if adapter is None:
            pass
        elif isinstance(adapter, Mapping):
            object.__setattr__(
                self, "adapter", _clean_json_kwargs(adapter, "Recipe.adapter")
            )
        else:
            from rlmesh.adapters import EnvTags, ModelSpec

            if self.kind == "env" and not isinstance(adapter, EnvTags):
                raise RecipeValidationError(
                    f"Recipe.adapter must be an EnvTags for kind='env'; "
                    f"got {type(adapter).__name__}"
                )
            if self.kind == "model" and not isinstance(adapter, ModelSpec):
                raise RecipeValidationError(
                    f"Recipe.adapter must be a ModelSpec for kind='model'; "
                    f"got {type(adapter).__name__}"
                )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-shaped mapping for this recipe."""
        adapter = _adapter_to_dict(self.adapter)
        return {
            "name": self.name,
            "make": _make_to_dict(self.make),
            "build": _build_to_dict(self.build),
            "setup": _setup_to_dict(self.setup),
            "requires": {"imports": list(self.requires.imports)},
            "summary": self.summary,
            "adapter": adapter,
            "recipe_version": self.recipe_version,
            "kind": self.kind,
            "inputs": [_artifact_to_dict(a) for a in self.inputs],
            "runtime": _runtime_to_dict(self.runtime),
        }

    def to_json(self) -> str:
        """Serialize to the canonical JSON wire format consumed by the Rust core."""
        import json

        # allow_nan=False is defense-in-depth: _clean_json already rejects non-finite
        # floats at construction, so the wire can never carry the NaN/Infinity tokens
        # that the Rust serde_json boundary rejects, even if some path bypasses it.
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Recipe:
        """Build a recipe from a canonical JSON-shaped mapping (executes nothing)."""
        return cls(
            name=_expect_str(payload.get("name"), "name"),
            make=_make_from_dict(payload.get("make")),
            build=_build_from_dict(payload.get("build")),
            setup=_setup_from_dict(payload.get("setup")),
            requires=Requires(imports=_str_list(_get(payload, "requires", "imports"))),
            summary=_opt_str(payload.get("summary"), "summary"),
            adapter=_opt_map(payload.get("adapter")),
            recipe_version=_expect_int(
                payload.get("recipe_version"), "recipe_version", RECIPE_VERSION
            ),
            kind=_recipe_kind_from(payload.get("kind")),
            inputs=_artifact_list(payload.get("inputs")),
            runtime=RuntimeReserved.from_dict(_opt_map(payload.get("runtime"))),
        )

    @classmethod
    def from_json(cls, payload: str) -> Recipe:
        """Parse a recipe from canonical JSON. Parsing is inert: executes nothing."""
        import json

        loaded: object = json.loads(payload)
        return cls.from_dict(_require_map(loaded, "recipe JSON"))


def _recipe_kind_from(value: object) -> RecipeKind:
    """Coerce a JSON ``kind`` to the sealed literal; absent/None defaults to ``env``."""
    if value is None:
        return "env"
    if value not in ("env", "model"):
        raise RecipeValidationError(
            f"Recipe.kind must be 'env' or 'model', got {value!r}"
        )
    return value


def _artifact_list(value: object) -> tuple[ArtifactInput, ...]:
    """Coerce a JSON ``inputs`` array (or absent) to a tuple of artifact mounts."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(_artifact_from_dict(item) for item in cast("Sequence[object]", value))


def _adapter_to_dict(
    adapter: EnvTags | ModelSpec | Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Serialize ``Recipe.adapter`` to a BARE JSON dict.

    A raw Mapping passes through verbatim; a dataclass (``EnvTags``/``ModelSpec``)
    serializes via its ``to_dict()`` -- NOT ``to_metadata()`` (that double-nests under
    the wire key; the recipe envelope carries the bare spec).
    """
    if adapter is None:
        return None
    if isinstance(adapter, Mapping):
        return dict(adapter)
    return adapter.to_dict()


def _artifact_to_dict(artifact: ArtifactInput) -> dict[str, object]:
    """Serialize one runtime weight mount; omit None/empty optionals."""
    data: dict[str, object] = {
        "name": artifact.name,
        "target_path": artifact.target_path,
        "required": artifact.required,
    }
    if artifact.uri is not None:
        data["uri"] = artifact.uri
    if artifact.local_dir is not None:
        data["local_dir"] = artifact.local_dir
    if artifact.include:
        data["include"] = list(artifact.include)
    return data


def _artifact_from_dict(value: object) -> ArtifactInput:
    data = _require_map(value, "ArtifactInput")
    return ArtifactInput(
        name=_expect_str(data.get("name"), "ArtifactInput.name"),
        target_path=_expect_str(data.get("target_path"), "ArtifactInput.target_path"),
        uri=_opt_str(data.get("uri"), "ArtifactInput.uri"),
        local_dir=_opt_str(data.get("local_dir"), "ArtifactInput.local_dir"),
        include=_str_list(data.get("include")),
        required=_expect_bool(data.get("required"), "ArtifactInput.required", True),
    )


def _runtime_to_dict(runtime: RuntimeReserved) -> dict[str, object] | None:
    """The reserved-features struct serializes to ``None`` when inert."""
    return runtime.to_dict()


def _make_to_dict(make: Make | None) -> dict[str, object] | None:
    if make is None:
        return None
    if isinstance(make, GymMake):
        return {"kind": "gym", "env_id": make.env_id, "kwargs": dict(make.kwargs)}
    if isinstance(make, PyMake):
        return {
            "kind": "py",
            "entrypoint": make.entrypoint,
            "kwargs": dict(make.kwargs),
        }
    return {
        "kind": "hf",
        "repo": make.repo,
        "revision": make.revision,
        "suite": make.suite,
        "task": make.task,
        "kwargs": dict(make.kwargs),
    }


def _make_from_dict(value: object) -> Make | None:
    if value is None:
        return None
    value = _require_map(value, "make")
    kind = _expect_str(value.get("kind"), "make.kind")
    kwargs = _opt_map(value.get("kwargs")) or {}
    if kind == "gym":
        return GymMake(
            env_id=_expect_str(value.get("env_id"), "make.env_id"), kwargs=kwargs
        )
    if kind == "py":
        return PyMake(
            entrypoint=_expect_str(value.get("entrypoint"), "make.entrypoint"),
            kwargs=kwargs,
        )
    if kind == "hf":
        return HfMake(
            repo=_expect_str(value.get("repo"), "make.repo"),
            revision=_opt_str(value.get("revision"), "make.revision"),
            suite=_opt_str(value.get("suite"), "make.suite"),
            task=_opt_str(value.get("task"), "make.task"),
            kwargs=kwargs,
        )
    raise RecipeValidationError(f"unknown make.kind {kind!r}")


def _pip_to_dict(step: PipInstall) -> dict[str, object]:
    return {
        "packages": list(step.packages),
        "index_url": step.index_url,
        "extra_index_urls": list(step.extra_index_urls),
        "no_deps": step.no_deps,
        "pre": step.pre,
        "requirements": step.requirements,
    }


def _fetch_to_dict(step: Fetch) -> dict[str, object]:
    return {
        "kind": step.kind,
        "repo": step.repo,
        "ref": step.ref,
        "dest": step.dest,
        "pip_install": step.pip_install,
        "pip_requirements": step.pip_requirements,
        "url": step.url,
        "sha256": step.sha256,
    }


def _project_to_dict(project: ProjectInstall | None) -> dict[str, object] | None:
    if project is None:
        return None
    return {
        "src": project.src,
        "dest": project.dest,
        "editable": project.editable,
        "include": list(project.include),
    }


def _build_to_dict(build: Build) -> dict[str, object]:
    return {
        "base": build.base,
        "from_recipe": build.from_recipe,
        "system": list(build.system),
        "system_runtime": list(build.system_runtime),
        "pip": [_pip_to_dict(step) for step in build.pip],
        "project": _project_to_dict(build.project),
        "fetch": [_fetch_to_dict(step) for step in build.fetch],
        "env": dict(build.env),
        "pythonpath": list(build.pythonpath),
        "gpu": build.gpu,
        "installer": build.installer,
        "run_as": build.run_as,
        "toolchain": build.toolchain,
        "commands": list(build.commands),
        "dockerfile": build.dockerfile,
    }


def _setup_to_dict(setup: Setup) -> dict[str, object]:
    return {
        "env": dict(setup.env),
        "files": [
            {"path": fw.path, "contents": fw.contents, "if_absent": fw.if_absent}
            for fw in setup.files
        ],
        "params": list(setup.params),
    }


def _build_from_dict(value: object) -> Build:
    if value is None:
        return Build()
    value = _require_map(value, "build")
    return Build(
        base=_opt_str(value.get("base"), "build.base"),
        from_recipe=_opt_str(value.get("from_recipe"), "build.from_recipe"),
        system=_str_list(value.get("system")),
        system_runtime=_str_list(value.get("system_runtime")),
        pip=tuple(_pip_from_dict(item) for item in _seq(value.get("pip"))),
        project=_project_from_dict(value.get("project")),
        fetch=tuple(_fetch_from_dict(item) for item in _seq(value.get("fetch"))),
        env=_str_str_map(value.get("env")),
        pythonpath=_str_list(value.get("pythonpath")),
        gpu=_expect_bool(value.get("gpu"), "build.gpu", False),
        installer=_opt_str(value.get("installer"), "build.installer") or "pip",
        run_as=_opt_int(value.get("run_as"), "build.run_as"),
        toolchain=_opt_bool(value.get("toolchain"), "build.toolchain"),
        commands=_str_list(value.get("commands")),
        dockerfile=_opt_str(value.get("dockerfile"), "build.dockerfile"),
    )


def _pip_from_dict(value: object) -> PipInstall:
    value = _require_map(value, "build.pip[]")
    return PipInstall(
        packages=_str_list(value.get("packages")),
        index_url=_opt_str(value.get("index_url"), "pip.index_url"),
        extra_index_urls=_str_list(value.get("extra_index_urls")),
        no_deps=_expect_bool(value.get("no_deps"), "pip.no_deps", False),
        pre=_expect_bool(value.get("pre"), "pip.pre", False),
        requirements=_opt_str(value.get("requirements"), "pip.requirements"),
    )


def _fetch_from_dict(value: object) -> Fetch:
    value = _require_map(value, "build.fetch[]")
    return Fetch(
        kind=_expect_str(value.get("kind"), "fetch.kind"),
        repo=_opt_str(value.get("repo"), "fetch.repo"),
        ref=_opt_str(value.get("ref"), "fetch.ref"),
        dest=_opt_str(value.get("dest"), "fetch.dest") or "",
        pip_install=_expect_bool(value.get("pip_install"), "fetch.pip_install", False),
        pip_requirements=_opt_str(
            value.get("pip_requirements"), "fetch.pip_requirements"
        ),
        url=_opt_str(value.get("url"), "fetch.url"),
        sha256=_opt_str(value.get("sha256"), "fetch.sha256"),
    )


def _project_from_dict(value: object) -> ProjectInstall | None:
    if value is None:
        return None
    value = _require_map(value, "build.project")
    return ProjectInstall(
        src=_opt_str(value.get("src"), "project.src") or ".",
        dest=_opt_str(value.get("dest"), "project.dest") or "",
        editable=_expect_bool(value.get("editable"), "project.editable", True),
        include=_str_list(value.get("include")),
    )


def _setup_from_dict(value: object) -> Setup:
    if value is None:
        return Setup()
    value = _require_map(value, "setup")
    files: list[FileWrite] = []
    for raw in _seq(value.get("files")):
        item = _require_map(raw, "setup.files[]")
        files.append(
            FileWrite(
                path=_expect_str(item.get("path"), "setup.files[].path"),
                contents=_expect_str(item.get("contents"), "setup.files[].contents"),
                if_absent=_expect_bool(
                    item.get("if_absent"), "setup.files[].if_absent", False
                ),
            )
        )
    return Setup(
        env=_str_str_map(value.get("env")),
        files=tuple(files),
        params=_str_list(value.get("params")),
    )


def _expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RecipeValidationError(f"{label} must be a string")
    return value


def _opt_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _expect_str(value, label)


def _expect_int(value: object, label: str, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise RecipeValidationError(f"{label} must be an integer")
    return value


def _opt_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _expect_int(value, label, 0)


def _expect_bool(value: object, label: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise RecipeValidationError(f"{label} must be a boolean")
    return value


def _opt_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    return _expect_bool(value, label, False)


def _require_map(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecipeValidationError(f"{label} must be a JSON object")
    # JSON object keys are always strings once parsed by json.loads.
    return cast("Mapping[str, object]", value)


def _seq(value: object) -> Sequence[object]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RecipeValidationError("expected a JSON array")
    return cast("Sequence[object]", value)


def _str_list(value: object) -> list[str]:
    return [_expect_str(item, "array item") for item in _seq(value)]


def _str_str_map(value: object) -> dict[str, str]:
    if value is None:
        return {}
    return {
        key: _expect_str(item, "map value")
        for key, item in _require_map(value, "object of strings").items()
    }


def _opt_map(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    return dict(_require_map(value, "object"))


def _get(payload: Mapping[str, object], outer: str, inner: str) -> object:
    section = payload.get(outer)
    if section is None:
        return None
    return _require_map(section, outer).get(inner)


# Surface every dataclass field name for tooling/conformance cross-checks without
# importing dataclasses at call sites.
RECIPE_FIELD_NAMES: Final = tuple(f.name for f in fields(Recipe))
