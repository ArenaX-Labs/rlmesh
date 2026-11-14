"""Leaving a ``with`` block never swaps the exception in flight for a close failure."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("numpy")

from rlmesh.realtime import Policy


class _SlowHost:
    """A native controller whose close finds the host not answering."""

    def close(self) -> None:
        raise TimeoutError("no reply from the session host within 2s")


def _policy() -> Policy:
    policy = Policy({"hz": 1, "inputs": {}, "actions": {}}, session="127.0.0.1:1")
    policy._native = _SlowHost()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    return policy


def test_a_failed_close_does_not_replace_the_exception_in_flight(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="rlmesh.realtime"):
        with pytest.raises(ValueError, match="the model failed"):
            with _policy():
                raise ValueError("the model failed")
    assert "no reply from the session host" in caplog.text


def test_a_failed_close_is_raised_when_nothing_else_is() -> None:
    with pytest.raises(TimeoutError, match="no reply"):
        with _policy():
            pass
