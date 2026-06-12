"""Shared Python model wrapper."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast

from ._values import ValueBridge
from .types import Value

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModel, ServeOptions

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
LifecycleCallback = Callable[[], None]
PredictFn = Callable[[ObsT], ActT]


class ModelBase(Generic[ObsT, ActT]):
    """Wrap a Python prediction function as an RLMesh model worker.

    A model worker receives observations from an RLMesh environment endpoint,
    calls ``predict_fn`` once per observation, and returns the encoded action to
    the runtime loop.

    Args:
        predict_fn: Callable that maps one decoded observation to one action.
        on_reset: Optional callback invoked when the environment resets.
        on_episode_end: Optional callback invoked when an episode ends.
        on_close: Optional callback invoked when the model worker closes.

    Examples:
        >>> from rlmesh.numpy import Model
        >>> model = Model(lambda observation: 0)
        >>> model.run("127.0.0.1:5555", max_episodes=1)
    """

    _bridge: ClassVar[ValueBridge]

    def __init__(
        self,
        predict_fn: PredictFn[ObsT, ActT],
        *,
        on_reset: LifecycleCallback | None = None,
        on_episode_end: LifecycleCallback | None = None,
        on_close: LifecycleCallback | None = None,
    ) -> None:
        try:
            from rlmesh._rlmesh import PyModel
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError("Failed to import _rlmesh native module.") from e

        self._bridge.ensure_available()

        def wrapped_predict(observation: Value) -> Value:
            decoded = cast(ObsT, self._bridge.decode(observation))
            action = predict_fn(decoded)
            return self._bridge.encode(action)

        self._worker: PyModel = PyModel(
            predict_fn=wrapped_predict,
            on_reset=on_reset,
            on_episode_end=on_episode_end,
            on_close=on_close,
        )

    def run_local(self, env_address: str, *, token: str = "") -> None:
        """Run against a remote environment endpoint until interrupted.

        Args:
            env_address: Environment endpoint address.
            token: Optional endpoint token.
        """
        self._worker.run_local(env_address, token)

    def run_local_for_episodes(
        self, env_address: str, *, token: str = "", max_episodes: int
    ) -> None:
        """Run against a remote environment endpoint for a fixed episode count.

        Args:
            env_address: Environment endpoint address.
            token: Optional endpoint token.
            max_episodes: Number of episodes to run before returning.
        """
        self._worker.run_local_for_episodes(env_address, token, max_episodes)

    def serve(
        self, address: str, *, token: str = "", options: ServeOptions | None = None
    ) -> None:
        """Serve this model as a model endpoint.

        Args:
            address: Address where the model endpoint should listen.
            token: Optional endpoint token.
            options: Optional serve lifecycle options.
        """
        self._worker.serve(address, token, options)

    def run(
        self,
        env_or_address: object,
        *,
        token: str = "",
        max_episodes: int | None = None,
        close_env: bool = False,
    ) -> None:
        """Run the model against an environment object or address.

        Args:
            env_or_address: Remote environment object or endpoint address.
            token: Optional endpoint token.
            max_episodes: Optional number of episodes to run before returning.
            close_env: If ``True``, request environment shutdown after the run.
        """
        address = _env_address(env_or_address)
        if max_episodes is None:
            self._worker.run_local(address, token)
        else:
            self._worker.run_local_for_episodes(address, token, max_episodes)
        if close_env:
            _shutdown_env(env_or_address, address)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _env_address(env_or_address: object) -> str:
    if isinstance(env_or_address, str):
        return env_or_address
    address = getattr(env_or_address, "_address", None)
    if isinstance(address, str):
        return address
    address_method = getattr(env_or_address, "address", None)
    if callable(address_method):
        value = address_method()
        if isinstance(value, str):
            return value
    raise TypeError("Model.run() expects a remote env object or address string")


def _shutdown_env(env_or_address: object, address: str) -> None:
    shutdown = getattr(env_or_address, "shutdown", None)
    if callable(shutdown):
        shutdown("local model run complete")
        return

    from rlmesh._rlmesh import PyEnvClient

    PyEnvClient(address).shutdown("local model run complete")


__all__ = ["LifecycleCallback", "ModelBase", "PredictFn"]
