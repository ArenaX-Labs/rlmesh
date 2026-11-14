"""``python -m rlmesh.serve`` sets a served model's batching from flags or env vars.

A container is configured through its environment alone, so every
``BatchOptions`` field has a flag and an ``RLMESH_*`` variable: the flag wins,
unset takes the default, and a bad value stops startup naming the variable.
"""

from __future__ import annotations

from typing import Any

import pytest
from rlmesh import BatchOptions, serve

_VARS = (
    "RLMESH_MAX_BATCH_SIZE",
    "RLMESH_BATCH_WINDOW_MS",
    "RLMESH_BATCH_BYTE_BUDGET",
    "RLMESH_MAX_IN_FLIGHT_BATCHES",
    "RLMESH_BATCH_RECORDS",
)


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(serve, "resolve_entrypoint", lambda *a, **k: object())

    def fake_serve_model(model: object, address: str, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(serve, "serve_model", fake_serve_model)
    monkeypatch.setattr(serve, "serve_env", lambda *a, **k: None)
    return captured


def test_unset_keeps_the_defaults(captured: dict[str, Any]) -> None:
    assert serve.main(["pkg:Model"]) == 0
    assert captured["batching"] is None


def test_every_field_has_a_flag(captured: dict[str, Any]) -> None:
    argv = [
        "pkg:Model",
        "--max-batch-size=8",
        "--batch-window-ms=0",
        "--batch-byte-budget=1024",
        "--max-in-flight-batches=1",
        "--batch-records=/tmp/batches.jsonl",
    ]
    assert serve.main(argv) == 0
    assert captured["batching"] == BatchOptions(
        max_batch_size=8,
        batch_window_ms=0,
        batch_byte_budget=1024,
        max_in_flight_batches=1,
        batch_records="/tmp/batches.jsonl",
    )


def test_every_field_has_a_variable(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLMESH_MAX_BATCH_SIZE", "16")
    monkeypatch.setenv("RLMESH_BATCH_WINDOW_MS", "3")
    monkeypatch.setenv("RLMESH_BATCH_BYTE_BUDGET", "4096")
    monkeypatch.setenv("RLMESH_MAX_IN_FLIGHT_BATCHES", "2")
    monkeypatch.setenv("RLMESH_BATCH_RECORDS", "batches.jsonl")
    assert serve.main(["pkg:Model"]) == 0
    assert captured["batching"] == BatchOptions(
        max_batch_size=16,
        batch_window_ms=3,
        batch_byte_budget=4096,
        max_in_flight_batches=2,
        batch_records="batches.jsonl",
    )


def test_a_flag_wins_over_its_variable_and_the_rest_stay_unset(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLMESH_MAX_BATCH_SIZE", "16")
    monkeypatch.setenv("RLMESH_BATCH_RECORDS", "from-env.jsonl")
    argv = ["pkg:Model", "--max-batch-size", "4", "--batch-records", "flag.jsonl"]
    assert serve.main(argv) == 0
    assert captured["batching"] == BatchOptions(
        max_batch_size=4, batch_records="flag.jsonl"
    )


@pytest.mark.parametrize(
    ("var", "raw"),
    [
        ("RLMESH_MAX_BATCH_SIZE", "0"),
        ("RLMESH_MAX_BATCH_SIZE", "lots"),
        ("RLMESH_BATCH_WINDOW_MS", "-1"),
        ("RLMESH_BATCH_WINDOW_MS", "2.5"),
        ("RLMESH_BATCH_BYTE_BUDGET", "0"),
        ("RLMESH_MAX_IN_FLIGHT_BATCHES", "0"),
        ("RLMESH_MAX_IN_FLIGHT_BATCHES", str(2**64)),
    ],
)
def test_a_bad_variable_stops_startup_naming_it(
    captured: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    var: str,
    raw: str,
) -> None:
    monkeypatch.setenv(var, raw)
    with pytest.raises(SystemExit) as exited:
        serve.main(["pkg:Model"])
    assert exited.value.code == 2
    assert var in capsys.readouterr().err
    assert "batching" not in captured


def test_a_bad_flag_stops_startup_naming_it(
    captured: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        serve.main(["pkg:Model", "--max-batch-size", "0"])
    assert "--max-batch-size" in capsys.readouterr().err
    assert "batching" not in captured


def test_a_good_flag_covers_a_bad_variable(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLMESH_MAX_BATCH_SIZE", "lots")
    assert serve.main(["pkg:Model", "--max-batch-size", "2"]) == 0
    assert captured["batching"] == BatchOptions(max_batch_size=2)


def test_batching_is_refused_for_an_env(
    captured: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RLMESH_MAX_BATCH_SIZE", "4")
    with pytest.raises(SystemExit):
        serve.main(["--env", "pkg:Env"])
    assert "apply only to a model" in capsys.readouterr().err


def test_serve_model_hands_batching_to_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh._models.base import ModelBase

    served: dict[str, Any] = {}

    def fake_serve(self: object, address: str, **kwargs: Any) -> None:
        served.update(kwargs)

    monkeypatch.setattr(ModelBase, "serve", fake_serve)
    monkeypatch.setattr(serve, "_stamp_startup", lambda *a, **k: None)
    batching = BatchOptions(max_batch_size=3)
    serve.serve_model(lambda observation: 0, "127.0.0.1:0", batching=batching)
    assert served["batching"] is batching
