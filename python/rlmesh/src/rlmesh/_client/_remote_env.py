"""Shared single-environment remote client base."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from ..spaces import Space
from . import _gym
from ._remote_base import ActionT, RemoteClientBase, ValueT

if TYPE_CHECKING:
    from rlmesh._rlmesh import ResetInfo, StepInfo


class RemoteEnvBase(RemoteClientBase[ValueT, ActionT]):
    """Base class for backend-specific single-environment remote clients.

    Backend modules such as ``rlmesh.numpy`` and ``rlmesh.torch`` configure the
    value bridge. User code should normally instantiate those concrete
    backends instead of this base class.

    A client owns one wire session and is not thread-safe: it serves one
    caller at a time, so dial one client per thread.

    Args:
        address: Endpoint address such as ``"tcp://127.0.0.1:5555"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
        connect_timeout_seconds: Optional dial timeout in seconds.
        request_timeout_seconds: Optional per-request timeout in seconds.
        workflow_edition: The workflow edition this client declares.
        api_key: Key sent as ``authorization: Bearer <key>`` on every request.
            ``None`` (default) reads ``RLMESH_API_KEY`` for an ``https://``
            address only; ``""`` sends none. Never shown in errors or repr.
        tls_ca_file: PEM bundle an ``https://`` dial trusts instead of the
            system roots.
    """

    _observation_space: Space[ValueT] | None = None
    _action_space: Space[ActionT] | None = None
    _gym_codecs: tuple[_gym.Convert | None, _gym.Convert] | None = None

    def _make_client(
        self,
        address: str,
        connect_timeout_seconds: float | None,
        request_timeout_seconds: float | None,
        workflow_edition: str | None,
        **credentials: str,
    ) -> Any:
        from .._load_native import load_native

        return load_native("PyEnvClient")(
            address,
            connect_timeout_seconds=connect_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
            workflow_edition=workflow_edition,
            **credentials,
            # A Gymnasium face's info is the env's own, exactly.
            **({"runtime_info": False} if self._gym_active else {}),
        )

    @property
    def observation_space(self) -> Space[ValueT]:
        """Observation space loaded from the remote environment contract."""
        if self._observation_space is None:
            self._observation_space = self._load_observation_space()
        return self._observation_space

    @property
    def action_space(self) -> Space[ActionT]:
        """Action space loaded from the remote environment contract."""
        if self._action_space is None:
            self._action_space = self._load_action_space()
        return self._action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[ValueT, ResetInfo]:
        """Reset the remote environment and decode the observation.

        Args:
            seed: Optional environment reset seed.
            options: Optional reset options forwarded to the environment.

        Returns:
            A decoded observation and reset info dictionary. Besides the env's
            own keys, the info carries the runtime-reserved ``episode_ids``
            (the runtime-minted episode id, as a one-element list). An env key
            of the same name wins and shadows it; the client logs that once.
        """
        seeds = [seed] if seed is not None else None
        if self._gym_active:
            # Seeds this env's own ``np_random``, as ``gymnasium.Env.reset`` does.
            cast("Any", _gym.gym).Env.reset(self, seed=seed)
        obs, info = self._client.reset(seeds=seeds, options=options)
        return self._observation(obs), info

    def step(self, action: ActionT) -> tuple[ValueT, float, bool, bool, StepInfo]:
        """Step the remote environment with one encoded action.

        Args:
            action: Action accepted by the remote environment action space.

        Returns:
            Observation, reward, terminated flag, truncated flag, and step info.
            Besides the env's own keys, the info carries two runtime-reserved
            keys on every step: ``episode_ids`` (the runtime-minted episode id,
            as a one-element list) and ``completed_episodes`` (``1`` on the
            step that ends an episode, else ``0``). An env key of the same name
            wins and shadows the runtime's; the client logs that once.
        """
        if self._gym_values:
            action = cast("ActionT", self._codecs()[1](action))
        obs, reward, terminated, truncated, info = self._client.step(
            self._bridge.encode(action)
        )
        return self._observation(obs), reward, terminated, truncated, info

    def _codecs(self) -> tuple[_gym.Convert | None, _gym.Convert]:
        if self._gym_codecs is None:
            self._gym_codecs = (
                _gym.observation_converter(self.observation_space),
                _gym.action_converter(self.action_space, self._space_spec("action")),
            )
        return self._gym_codecs

    def _observation(self, wire: object) -> ValueT:
        obs = self._bridge.decode(cast("Any", wire))
        if self._gym_values:
            convert = self._codecs()[0]
            if convert is not None:
                obs = convert(obs)
        return cast("ValueT", obs)

    def __enter__(self) -> RemoteEnvBase[ValueT, ActionT]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _ = exc_type, exc_val, exc_tb
        self.close()


__all__ = ["ActionT", "RemoteEnvBase", "ValueT"]
