"""The result ``TypedDict``s have exactly the keys the extension's dicts are built with.

Read from the extension's source, so a key added, renamed or dropped there
fails here without a session; ``integration/test_v2_builders.py`` checks the
dicts a live session returns, key for key and type for type.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, get_args, get_type_hints

import pytest

pytest.importorskip("numpy")

from rlmesh.realtime import results

SESSION_RS = Path(__file__).resolve().parents[2] / "rust" / "src" / "session.rs"
# A literal key, or one set through a loop over ``("key", row.field)`` pairs.
KEY = re.compile(r'\.set_item\(\s*"([a-z0-9_]+)"|\(\s*"([a-z0-9_]+)",\s*row\.\w+\s*\)')
KIND = re.compile(r'\.set_item\("kind", "([a-z_]+)"\)')


def body(signature: str) -> str:
    """The body of the one function whose source starts with ``signature``."""
    source = SESSION_RS.read_text()
    assert source.count(signature) == 1, (
        f"{signature!r} is not one function in session.rs"
    )
    start = source.index("{", source.index(signature))
    depth = 0
    for end in range(start, len(source)):
        depth += {"{": 1, "}": -1}.get(source[end], 0)
        if depth == 0:
            return source[start : end + 1]
    raise AssertionError(f"{signature!r} never closes")


def built(source: str) -> set[str]:
    return {literal or paired for literal, paired in KEY.findall(source)}


def keys(*typed: Any) -> set[str]:
    return {key for t in typed for key in get_type_hints(t)}


@pytest.mark.parametrize(
    ("signature", "typed"),
    [
        (
            "fn health_dict<'py>(",
            (
                results.Health,
                results.Link,
                results.ClockRefusal,
                results.Published,
                results.LinkRun,
            ),
        ),
        (
            "fn health<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>>",
            (
                results.HostLink,
                results.ChunkTiming,
                results.DatagramEnd,
                results.PathTime,
                results.DatagramRelay,
            ),
        ),
        ("fn edge_stats<'py>(", (results.EdgeStats, results.EdgeClockRefusal)),
        ("fn held_list<'py>(", (results.Held,)),
        ("fn stats<'py>(", (results.PolicyStats,)),
        ("fn release<'py>(", (results.HandBack,)),
    ],
)
def test_a_result_has_the_keys_its_dicts_are_built_with(
    signature: str, typed: tuple[Any, ...]
) -> None:
    made = built(body(signature))
    declared = keys(*typed)
    assert made - declared == set(), f"built, not typed: {sorted(made - declared)}"
    assert declared - made == set(), f"typed, not built: {sorted(declared - made)}"


def test_each_event_kind_has_exactly_its_keys() -> None:
    events = body("fn events<'py>(")
    arms = events.split("RunEvent::")[1:]
    assert arms, "no RunEvent arms in events()"
    typed = {
        kind: event
        for event in get_args(results.Event)
        for kind in get_args(get_type_hints(event)["kind"])
    }
    seen: set[str] = set()
    for arm in arms:
        [kind] = KIND.findall(arm)
        assert kind in typed, f"event {kind!r} has no TypedDict in results.Event"
        seen.add(kind)
        made = built(arm) | {"message"}
        declared = set(get_type_hints(typed[kind]))
        assert made == declared, (
            f"{kind}: built {sorted(made)}, typed {sorted(declared)}"
        )
    assert seen == set(typed), f"typed, never built: {sorted(set(typed) - seen)}"
