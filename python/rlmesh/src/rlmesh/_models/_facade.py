"""Python target, framework and presentation conversion for the Rust facade."""

from __future__ import annotations

import contextlib
import weakref
from functools import partial
from typing import Any, cast

from ._eval import (
    EpisodeResult,
    RunHooks,
    RunResult,
    StepEvent,
    TelemetryRow,
    artifact_tap,
    episode_outcome,
    overrides_transforms,
)


def result_from_native(report: dict[str, Any]) -> RunResult:
    return RunResult(
        episodes=tuple(EpisodeResult(**e) for e in report["episodes"]),
        telemetry=tuple(TelemetryRow(**row) for row in report["telemetry"]),
        advisories=tuple(report.get("advisories", ())),
    )


class FacadeHookRelay:
    """Convert shared facade events; never count steps or reconstruct results."""

    def __init__(
        self,
        hooks: RunHooks | None,
        context: Any,
        view: Any = None,
        *,
        weak_context: bool = False,
    ) -> None:
        self.hooks = hooks
        # The native binding owns its observer: holding the Python Binding here
        # would introduce a cycle through an opaque extension object.
        self._weak_context: weakref.ReferenceType[Any] | None = (
            weakref.ref(context) if weak_context else None
        )
        self._context: Any = None if weak_context else context
        self.view = view
        self.result = RunResult()
        self._outcome = ""

    @property
    def context(self) -> Any:
        return self._weak_context() if self._weak_context is not None else self._context

    def start(self, core: object) -> None:
        self.context._core = core
        if self.hooks is not None:
            with self.context._dispatch():
                self.hooks.on_run_start(self.context)

    def episode_start(self, episode: int, seed: int | None) -> None:
        if self.hooks is not None:
            with self.context._dispatch():
                self.hooks.on_episode_start(episode=episode, seed=seed)

    def _viewer(self) -> Any:
        return getattr(self.context, "_view", self.view)

    def wants_frame(self) -> bool:
        return self._viewer() is not None

    def wants_step(self) -> bool:
        return self.hooks is not None or self._viewer() is not None

    def wants_transforms(self) -> bool:
        return self.hooks is not None and overrides_transforms(self.hooks)

    def artifacts(self) -> Any:
        return artifact_tap(self.hooks)

    def transform_observation(
        self, observation: object, episode: int, step: int
    ) -> Any:
        assert self.hooks is not None
        with self.context._dispatch():
            out = self.hooks.transform_observation(
                self.context._decode(observation), episode=episode, step=step
            )
        return None if out is None else self.context._bridge.encode(out)

    def transform_action(self, action: object, episode: int, step: int) -> Any:
        assert self.hooks is not None
        with self.context._dispatch():
            out = self.hooks.transform_action(
                self.context._decode(action), episode=episode, step=step
            )
        return None if out is None else self.context._bridge.encode(out)

    def frame(
        self,
        observation: object,
        episode: int,
        seed: int | None,
        steps: int,
        reward: float,
        outcome: str | None = None,
        predict_ms: float = 0.0,
        step_ms: float = 0.0,
        elapsed_s: float = 0.0,
    ) -> None:
        viewer = self._viewer()
        if viewer is None:
            return
        with self.context._dispatch():
            viewer.feed(
                contract=self.context.contract,
                client=self.context._client,
                obs=self.context._decode(observation),
                read=self.context.read,
                steps=steps,
                reward=reward,
                outcome=self._outcome if outcome is None else outcome,
                episode=episode,
                seed=seed,
                model_ms=predict_ms,
                env_ms=step_ms,
                elapsed_s=elapsed_s,
            )

    def step(self, data: dict[str, Any]) -> None:
        self._outcome = episode_outcome(
            data["terminated"], data["truncated"], data["info"]
        )
        data["observation"] = self.context._decode(data["observation"])
        data["action"] = self.context._decode(data["action"])
        if self.hooks is not None:
            with self.context._dispatch():
                self.hooks.on_step(
                    StepEvent(
                        **data, read=partial(self.context.read, data["observation"])
                    )
                )
        viewer = self._viewer()
        if viewer is not None:
            viewer.pace()

    def directive(self) -> str | None:
        viewer = self._viewer()
        if viewer is None:
            return None
        if viewer.quit_requested():
            return "quit"
        return "skip" if viewer.consume_skip() else None

    def episode_end(self, data: dict[str, Any]) -> None:
        if self.hooks is not None:
            with self.context._dispatch():
                self.hooks.on_episode_end(EpisodeResult(**data))

    def end(self, report: dict[str, Any]) -> None:
        self.result = result_from_native(report)
        if self.hooks is not None:
            with self.context._dispatch():
                self.hooks.on_run_end(self.result)


class EnvTarget:
    """Convert Python env/server/Docker objects when the core opens its target."""

    def __init__(
        self,
        model: Any,
        target: object,
        context: Any = None,
        *,
        binding: bool = False,
        trust: bool | None = None,
        instruction: str | None = None,
    ) -> None:
        from ._connect import classify_env_target

        self.model = model
        self.kind, self.payload = classify_env_target(target)
        self.owned = self.kind == "factory"
        self.context = context
        self.binding = binding
        self.trust = trust
        self.instruction = instruction
        self.env: Any = None
        self.server: Any = None
        self.client: Any = None
        self._lent = contextlib.ExitStack()
        self._previous: tuple[Any, Any, Any] | None = None

    def open(self, edition: str | None) -> tuple[str, str | None, bool]:
        from .._sandbox.session import SandboxLifecycle, peer_session
        from ._connect import factory_env

        model = self.model
        self._previous = model._trust_entrypoints, model._instruction, model._native_run
        if self.trust is not None:
            model._trust_entrypoints = self.trust
        model._instruction = self.instruction
        model._native_run = self.context
        kind, payload = self.kind, self.payload
        self.env = None if self.owned else payload
        invite = None
        if kind == "address":
            return cast(str, payload), None, True
        if kind == "handle":
            if isinstance(payload, SandboxLifecycle):
                if self.context is not None:
                    self.context.bind_client(payload, local=False)
                address = self._lent.enter_context(payload._lent())  # pyright: ignore[reportPrivateUsage]
                if model._env_session is not None:
                    invite = peer_session(payload.sandbox)
                return address, invite, False
            # A SandboxEnv lends its container, as to a run; any other open
            # handle holds its env's lease.
            if self.binding:
                from .._upgrade import removed

                removed(
                    "bind(RemoteEnv(address))",
                    "an open RemoteEnv holds its env's lease, so close it and pass the address: bind(address)",
                )
            from .._client._remote_base import RemoteClientBase
            from .._rlmesh import is_v2_address

            if isinstance(payload, RemoteClientBase) and is_v2_address(payload.address):
                from .._upgrade import removed

                removed(
                    "run(RemoteEnv(address))",
                    "an open RemoteEnv holds its env's lease, so close it and pass the address: run(address)",
                )
            address = getattr(cast(object, payload), "address", None)
            if not isinstance(address, str):
                raise TypeError(
                    "this env handle exposes env_contract but no dialable address; pass the env's address string instead"
                )
            if self.context is not None:
                self.context.bind_client(payload, local=False)
            return address, None, False
        if kind == "factory":
            self.env = factory_env(payload)
        self.server = model._serve_here(self.env, edition)
        self.client = self.env
        if self.context is not None:
            self.context.bind_client(self.env)
            self.context.contract = self.server.env_contract
        return self.server.address, None, False

    def release(self, close_env: bool) -> None:
        try:
            try:
                if self.server is not None:
                    self.server.shutdown()
                    self.server = None
            finally:
                if close_env and self.env is not None and self.kind != "address":
                    from ._connect import shutdown_env

                    shutdown_env(self.env)
        finally:
            try:
                self._lent.close()
            finally:
                if self._previous is not None:
                    (
                        self.model._trust_entrypoints,
                        self.model._instruction,
                        self.model._native_run,
                    ) = self._previous
                    self._previous = None
