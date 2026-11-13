"""The calls refusals suggest exist, with the keywords they use."""

from __future__ import annotations

import ast
import inspect
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from rlmesh import realtime

REPO_ROOT = Path(__file__).resolve().parents[4]
GOLDEN = REPO_ROOT / "crates" / "rlmesh-session-core" / "tests" / "golden"
SOURCES = [
    REPO_ROOT / "crates" / "rlmesh-session-core" / "src",
    REPO_ROOT / "crates" / "rlmesh-session" / "src",
]

#: A dotted call, to the end of its line.
CALL = re.compile(r"\b[a-z_]+\.[a-z_]+\(.*\)")
#: What a hint's receiver stands for.
RECEIVERS: dict[str, type] = {"policy": realtime.Policy, "robot": realtime.Robot}


def fixes() -> Iterator[tuple[str, str, str]]:
    """Every call a golden refusal suggests: ``(golden, its text, the call)``."""
    for path in sorted(GOLDEN.glob("*__refused.txt")):
        text = path.read_text()
        for match in CALL.finditer(text):
            yield path.stem, text, match.group()


def parsed(fix: str) -> ast.Call:
    call = ast.parse(fix, mode="eval").body
    assert isinstance(call, ast.Call), fix
    return call


def signature(call: ast.Call) -> inspect.Signature:
    assert isinstance(call.func, ast.Attribute), ast.dump(call.func)
    assert isinstance(call.func.value, ast.Name), ast.dump(call.func.value)
    receiver = RECEIVERS.get(call.func.value.id)
    assert receiver is not None, f"{call.func.value.id} is not a receiver a user holds"
    method: Callable[..., object] | None = getattr(receiver, call.func.attr, None)
    assert method is not None, f"{receiver.__name__} has no {call.func.attr}"
    return inspect.signature(method)


def test_the_goldens_cover_every_keyword_of_bind() -> None:
    used = {keyword.arg for _, _, fix in fixes() for keyword in parsed(fix).keywords}
    assert {"under", "at", "names", "allow"} <= used


@pytest.mark.parametrize(
    ("name", "text", "fix"), list(fixes()), ids=lambda value: value[:40]
)
def test_a_refusals_fix_is_a_real_call(name: str, text: str, fix: str) -> None:
    call = parsed(fix)
    assert ast.unparse(call.func) == "policy.bind", f"{name}: {fix}"
    parameters = signature(call).parameters
    for keyword in call.keywords:
        assert keyword.arg is not None, fix
        # A keyword bind does not name is a capture: `{side}` in a selector.
        assert keyword.arg in parameters or f"{{{keyword.arg}}}" in text, (
            f"{name}: {fix}: bind() has no {keyword.arg}"
        )
    positional = [None for _ in call.args]
    keywords = {k.arg: None for k in call.keywords if k.arg is not None}
    signature(call).bind(None, *positional, **keywords)


def string_literal_calls(line: str) -> Iterator[str]:
    """The ``policy.x(...)`` and ``robot.x(...)`` calls inside a Rust line's strings."""
    for match in re.finditer(r"\b(policy|robot)\.([a-z_]+)\(", line):
        before = line[: match.start()].replace('\\"', "")
        if before.count('"') % 2 == 1:
            yield f"{match.group(1)}.{match.group(2)}"


def test_every_call_a_message_names_exists() -> None:
    named: set[str] = set()
    for root in SOURCES:
        for path in root.rglob("*.rs"):
            for line in path.read_text().splitlines():
                if not line.lstrip().startswith("//"):
                    named.update(string_literal_calls(line))
    assert "robot.rearm" in named, "the scan found no message at all"
    for call in sorted(named):
        receiver, method = call.split(".")
        assert hasattr(RECEIVERS[receiver], method), f"{call} does not exist"
