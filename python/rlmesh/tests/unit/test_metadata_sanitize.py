"""rlmesh.sanitize_metadata: the lossy mirror of the native metadata codec.

Pins that sanitize accepts exactly what the codec accepts (passthrough), unwraps
wrappers the way ``normalization.rs`` does, matches the codec's enum handling
(``value`` first, ``name`` fallback, scalar extraction first for mixins), and
stringifies everything else -- so a sanitized dict always survives the served
info path.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum, IntEnum
from typing import Any, TypeVar, cast

import pytest
import rlmesh


class _Pose:
    """A sapien.Pose-like rich object the codec rejects."""

    def __repr__(self) -> str:
        return "Pose([0, 0, 0], [1, 0, 0, 0])"

    def __str__(self) -> str:
        return "Pose([0, 0, 0], [1, 0, 0, 0])"


def test_accepted_shapes_pass_through_unchanged() -> None:
    info: dict[str, Any] = {
        "none": None,
        "flag": True,
        "big": 2**53 + 1,
        "lr": 2.0,
        "name": "cartpole",
        "blob": b"\x00\x01\xff",
        "nested": {"a": [1, "b", {"c": False}]},
    }
    assert rlmesh.sanitize_metadata(info) == info


def test_tuples_become_lists_like_the_codec() -> None:
    assert rlmesh.sanitize_metadata({"t": (1, 2, (3,))}) == {"t": [1, 2, [3]]}


def test_numpy_wrappers_unwrap_via_tolist_and_item() -> None:
    np = pytest.importorskip("numpy")
    out = rlmesh.sanitize_metadata(
        {
            "scalar": np.float32(1.5),
            "count": np.int64(7),
            "arr": np.arange(3, dtype=np.int32),
        }
    )
    assert out == {"scalar": 1.5, "count": 7, "arr": [0, 1, 2]}
    assert type(out["scalar"]) is float
    assert type(out["count"]) is int


def test_namedtuple_unwraps_via_asdict() -> None:
    from collections import namedtuple

    Point = namedtuple("Point", ["x", "y"])
    assert rlmesh.sanitize_metadata({"p": Point(1, 2)}) == {"p": {"x": 1, "y": 2}}


def test_enum_handling_matches_the_codec() -> None:
    class Color(Enum):
        RED = "red"

    class Count(IntEnum):
        ONE = 1

    class Opaque(Enum):
        X = _Pose()

    out = rlmesh.sanitize_metadata({"c": Color.RED, "n": Count.ONE, "o": Opaque.X})
    assert out == {"c": "red", "n": 1, "o": "X"}


def test_rich_leaf_becomes_str_and_non_str_keys_are_stringified() -> None:
    info = cast("dict[str, Any]", {"pose": _Pose(), 5: "five"})
    out = rlmesh.sanitize_metadata(info)
    assert out == {"pose": "Pose([0, 0, 0], [1, 0, 0, 0])", "5": "five"}


def test_reference_cycle_raises_naming_the_key_path() -> None:
    info: dict[str, Any] = {}
    info["outer"] = {"inner": info}
    with pytest.raises(ValueError, match=r"outer\.inner"):
        rlmesh.sanitize_metadata(info)


def test_non_mapping_input_raises_type_error() -> None:
    with pytest.raises(TypeError, match="mapping"):
        rlmesh.sanitize_metadata([("a", 1)])  # type: ignore[arg-type]


class _RichInfoEnv:
    """An env whose reset info only serves after sanitize_metadata."""

    def __init__(self) -> None:
        self.observation_space = rlmesh.spaces.Discrete(2)
        self.action_space = rlmesh.spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[int, dict[str, Any]]:
        raw: dict[Any, Any] = {
            "pose": _Pose(),
            "big": 2**53 + 1,
            "blob": b"\x00\xff",
            "nested": {"t": (1, 2)},
            5: "five",
        }
        return 0, rlmesh.sanitize_metadata(raw)

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, Any]]:
        return 0, 1.0, True, False, {}

    def close(self) -> None:
        return None


RemoteT = TypeVar("RemoteT")


def _connect_with_retry(factory: Callable[[str], RemoteT], address: str) -> RemoteT:
    deadline = time.monotonic() + 3.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return factory(address)
        except Exception as exc:
            last_error = exc
            time.sleep(0.05)
    raise AssertionError(f"failed to connect to {address}") from last_error


def test_sanitized_info_survives_a_served_reset() -> None:
    try:
        server = rlmesh.EnvServer(_RichInfoEnv(), host="127.0.0.1", port=0)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    try:
        remote = _connect_with_retry(rlmesh.RemoteEnv, server.address)
        try:
            _obs, info = remote.reset(seed=1)
            assert info["pose"] == "Pose([0, 0, 0], [1, 0, 0, 0])"
            assert info["big"] == 2**53 + 1
            assert info["blob"] == b"\x00\xff"
            assert info["nested"] == {"t": [1, 2]}
            assert info["5"] == "five"
        finally:
            remote.close()
    finally:
        server.shutdown()
