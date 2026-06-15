"""``SandboxModel``: a model recipe running in its own container.

The model-side sibling of ``SandboxEnv``. The recipe builds to an image whose
ENTRYPOINT is the model bootstrap (the kind-aware deriver selects it). Like
:class:`~rlmesh.models.base.ModelBase`, construction is inert -- it resolves the
recipe but starts nothing. ``run(env)`` drives a one-shot container against an env
and returns a :class:`RunResult`; ``serve()`` (or use as a context manager) starts
a long-lived container serving the policy as a model endpoint. A spec'd model can
only be driven with ``run(env)``, not served (see ``serve``).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ._export import normalize_rlmesh_package

if TYPE_CHECKING:
    from ..models._eval import RunResult
    from ..recipes import Recipe
    from ..recipes._schema import ArtifactInput

__all__ = ["SandboxModel"]


def resolve_model_recipe(source: object) -> tuple[Recipe, str | None]:
    from ..recipes import Recipe, resolve
    from ..recipes._registry import class_origin_dir, recipe_origin_dir
    from ..recipes._schema import PyMake
    from ..recipes.authoring.model import as_authored_model_recipe, is_model_recipe

    context_root: str | None = None
    if isinstance(source, str):
        recipe = resolve(source)
        context_root = recipe_origin_dir(source)
    elif isinstance(source, Recipe):
        recipe = source
    else:
        recipe = as_authored_model_recipe(source)
        if is_model_recipe(source):
            context_root = class_origin_dir(source)
    if recipe is None or recipe.kind != "model":
        raise TypeError(
            "SandboxModel requires a model recipe (a ModelRecipe subclass, a "
            "kind='model' Recipe, or a registered model name)"
        )
    # A flat registration (register(name, hf=...|load=...)) synthesizes its loader
    # class at register() time and binds it onto this module in the current
    # interpreter only. A fresh container imports the module from disk and never
    # sees that class, so the bootstrap would fail late with an opaque ImportError.
    # Fail early, on the host, with the actionable fix.
    if isinstance(recipe.make, PyMake) and recipe.make.entrypoint.split(":", 1)[0] == (
        "rlmesh.models._registry"
    ):
        raise TypeError(
            f"model {recipe.name!r} is flat-registered (register(name, hf=...|load=...)), "
            "which is in-process only and cannot be launched in a container. "
            "Subclass rlmesh.ModelRecipe so its loader lives in an importable module."
        )
    return recipe, context_root


def _resolve_env_address(env: object) -> str:
    """The address a drive container dials to reach ``env``.

    A ``SandboxEnv`` exposes ``env.sandbox.address``; a remote-env-like object an
    ``address``; a bare string is taken verbatim.
    """
    sandbox = getattr(env, "sandbox", None)
    if sandbox is not None and getattr(sandbox, "address", None):
        return sandbox.address
    address = getattr(env, "address", None)
    if isinstance(address, str) and address:
        return address
    if isinstance(env, str) and env:
        return env
    raise TypeError(
        "SandboxModel.run() expects a SandboxEnv, an object with an 'address', or "
        f"an address string; got {type(env).__name__}"
    )


def _parse_run_result(stdout: str) -> RunResult:
    """Reconstruct the :class:`RunResult` printed by the drive bootstrap."""
    from ..models._eval import EpisodeResult, RunResult

    for line in stdout.splitlines():
        if line.startswith("RLMESH_RUN_RESULT "):
            payload = json.loads(line[len("RLMESH_RUN_RESULT ") :])
            return RunResult(
                episodes=tuple(EpisodeResult(**e) for e in payload["episodes"])
            )
    raise RuntimeError(
        "drive container produced no RLMESH_RUN_RESULT line; container stdout:\n"
        + stdout
    )


class SandboxModel:
    """A model recipe run in an isolated container -- driven one-shot or served."""

    def __init__(
        self,
        source: object,
        *,
        base_image: str | None = None,
        rlmesh_package: str | None = None,
        packages: Sequence[str] = (),
        artifacts: Sequence[ArtifactInput] = (),
        trust_remote_code: bool = False,
        allow_unpinned_hf: bool = False,
        build_memory: str | None = None,
    ) -> None:
        from dataclasses import replace

        from ..recipes._artifacts import local_dir_mounts, merged_inputs

        recipe, context_root = resolve_model_recipe(source)
        # A declared input with a host local_dir is bind-mounted at its container
        # target; uri-backed inputs still resolve in-container. artifacts= supplies
        # or overrides a local_dir by name (the run-time checkpoint).
        self._mounts = local_dir_mounts(recipe.inputs, artifacts)
        # Reflect the overrides into the recipe the container reconstructs from, so
        # an overridden input carries a local_dir and input_path() resolves to the
        # bind-mounted target instead of re-fetching the original uri.
        if artifacts:
            recipe = replace(
                recipe,
                inputs=tuple(merged_inputs(recipe.inputs, artifacts).values()),
            )
        self._recipe = recipe
        self._context_root = context_root
        self._base_image = base_image
        self._rlmesh_package = normalize_rlmesh_package(rlmesh_package)
        self._packages = list(packages)
        self._trust_remote_code = trust_remote_code
        self._allow_unpinned_hf = allow_unpinned_hf
        self._build_memory = build_memory
        self._address: str | None = None
        self._container_id: str | None = None
        self._closed = False

    def _build_image(self) -> str:
        from .._rlmesh import sandbox_build_image

        info = sandbox_build_image(
            self._recipe.name,
            recipe_json=self._recipe.to_json(),
            recipe_provenance="installed",
            base_image=self._base_image,
            rlmesh_package=self._rlmesh_package,
            packages=self._packages,
            trust_remote_code=self._trust_remote_code,
            allow_unpinned_hf=self._allow_unpinned_hf,
            context_root=self._context_root,
            build_memory=self._build_memory,
        )
        return info["image"]

    def _bootstrap_json(self) -> str:
        """The run-time bootstrap payload, mirroring the native runner.

        Sets ``RLMESH_BOOTSTRAP_JSON`` to the current recipe's runtime half (the
        build phase is stripped -- the image already shaped it). This overrides the
        image's baked ``recipe.json``, so a cached image (keyed only by the build
        phase) still drives *this* recipe's policy and inputs, not a previously
        baked one.
        """
        document = json.loads(self._recipe.to_json())
        document.pop("build", None)
        return json.dumps({"spec": {"kind": "recipe", "document": document}})

    def _drive_run_args(self) -> list[str]:
        """``docker run`` args the one-shot drive shares with the native runner.

        Container hardening, the bootstrap payload, GPU access for ``build.gpu``
        recipes, and read-only artifact bind-mounts -- the same setup
        ``sandbox_start_env`` applies, minus the serve-only ``-d``/published port.
        ``--network host`` lets the container dial the env's host-loopback address.
        ponytail: --network host is Linux-only; add a host.docker.internal mapping
        if macOS support is ever needed.
        """
        # TODO(tech-debt): duplicates Rust's docker_run_args/render_bootstrap_json
        # (can drift). Clean form: a native sandbox_drive_env that reuses them.
        args = [
            "--network",
            "host",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--label",
            "rlmesh.sandbox=1",
            "-e",
            f"RLMESH_BOOTSTRAP_JSON={self._bootstrap_json()}",
        ]
        if self._recipe.build.gpu:
            args += ["--gpus", "all", "-e", "NVIDIA_VISIBLE_DEVICES=all"]
        for host, target in self._mounts:
            # docker --mount has no escape for a comma inside a value; reject it
            # loudly, mirroring the native runner.
            if "," in host or "," in target:
                raise ValueError(
                    f"artifact mount path must not contain a comma: {host!r} {target!r}"
                )
            args += ["--mount", f"type=bind,source={host},target={target},readonly"]
        return args

    def run(
        self,
        env: object,
        *,
        seeds: Sequence[int] = (0, 1),
        max_episodes: int | None = None,
    ) -> RunResult:
        """Build this model's image and drive ``env`` from a one-shot container.

        The containerized sibling of :meth:`Model.run`: builds the recipe image,
        runs it once in drive-mode against ``env`` (the policy and its adapter
        resolve in-container, where the weights and deps live), and returns the
        :class:`RunResult` the run reports. ``seeds`` sets the per-episode seeds
        and the episode count unless ``max_episodes`` is given.
        """
        address = _resolve_env_address(env)
        image = self._build_image()
        # Drive-mode env vars. RLMESH_BOOTSTRAP_JSON carries the *current* recipe's
        # runtime doc so the run reflects self._recipe even when a cached image
        # (keyed only by the build phase) bakes a different recipe's policy.
        cmd = ["docker", "run", "--rm", *self._drive_run_args()]
        cmd += ["-e", f"RLMESH_DRIVE_ENV_ADDRESS={address}"]
        cmd += ["-e", "RLMESH_SEEDS=" + ",".join(str(s) for s in seeds)]
        if max_episodes is not None:
            cmd += ["-e", f"RLMESH_MAX_EPISODES={max_episodes}"]
        cmd.append(image)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"drive container exited with code {proc.returncode}:\n{proc.stderr}"
            )
        return _parse_run_result(proc.stdout)

    def serve(self) -> SandboxModel:
        """Start a long-lived container serving the policy as a model endpoint.

        Spec-less / ``DELEGATED`` models only; a spec'd model's adapter resolves
        from the env contract on dial-in, so drive it with :meth:`run` instead (an
        in-container serve of a spec'd model exits at startup -- see ``Model.serve``).

        Idempotent: a second call returns the already-running handle. The endpoint
        is reachable at :attr:`address` until :meth:`shutdown`.
        """
        # TODO: serve a spec'd model once dial-in adapter resolution lands in
        # Model.serve; today it raises, so this container would exit at startup.
        if self._address is not None:
            return self
        from .._rlmesh import sandbox_start_env

        info = sandbox_start_env(
            self._recipe.name,
            recipe_json=self._recipe.to_json(),
            recipe_provenance="installed",
            base_image=self._base_image,
            rlmesh_package=self._rlmesh_package,
            packages=self._packages,
            trust_remote_code=self._trust_remote_code,
            allow_unpinned_hf=self._allow_unpinned_hf,
            context_root=self._context_root,
            mounts_json=json.dumps(self._mounts) if self._mounts else None,
            build_memory=self._build_memory,
        )
        self._address = info["address"]
        self._container_id = info["container_id"]
        self._closed = False
        return self

    @property
    def address(self) -> str:
        if self._address is None:
            raise RuntimeError(
                "SandboxModel is not serving; call serve() (or use it as a context "
                "manager) before reading address"
            )
        return self._address

    @property
    def container_id(self) -> str:
        if self._container_id is None:
            raise RuntimeError(
                "SandboxModel is not serving; call serve() before reading container_id"
            )
        return self._container_id

    def shutdown(self) -> None:
        """Stop the served container, if any. Idempotent; safe to call repeatedly."""
        if self._closed or self._container_id is None:
            return
        from .._rlmesh import sandbox_stop_env

        # Mark closed only once the stop succeeds, so a transient failure can be
        # retried by a later shutdown()/__exit__/__del__ instead of leaking the
        # container (mirrors SandboxSessionBase).
        sandbox_stop_env(container_id=self._container_id)
        self._closed = True
        self._address = None
        self._container_id = None

    def __enter__(self) -> SandboxModel:
        return self.serve()

    def __exit__(self, *exc: object) -> bool:
        self.shutdown()
        return False

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
