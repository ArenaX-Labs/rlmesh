"""The serve entrypoint stamps startup phase marks on the serving line and the
handshake, so a container's model wait splits into imports, construction, and
listen without a profiler."""

from __future__ import annotations

import pytest
from rlmesh import serve


def test_marks_are_monotonic_and_prefixed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "_marks", {})
    first = serve._mark("imports")
    second = serve._mark("model")
    assert 0 <= first <= second

    marks = serve.startup_marks()
    assert marks["rlmesh.startup.imports_ms"] == str(first)
    assert marks["rlmesh.startup.model_ms"] == str(second)
    assert all(key.startswith("rlmesh.startup.") for key in marks)
    if serve._PROCESS_AGE_MS is not None:
        assert int(marks["rlmesh.startup.process_ms"]) <= first


def test_serving_a_model_stamps_the_handshake_and_prints_the_phases(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import rlmesh._peer_info as peer_info

    monkeypatch.setattr(serve, "_marks", {})
    stamped: dict[str, str] = {}
    monkeypatch.setattr(
        peer_info,
        "register_python_peer_info",
        lambda extra=None: stamped.update(extra or {}),
    )

    class Served:
        def serve(self, address: str, *, options: object = None) -> None:
            self.address = address
            self.options = options

    served = Served()
    monkeypatch.setattr(serve, "_resolve_model", lambda source, binding: served)

    serve.serve_model(object(), "127.0.0.1:0")

    assert served.address == "127.0.0.1:0"
    assert {"rlmesh.startup.model_ms", "rlmesh.startup.listen_ms"} <= stamped.keys()
    line = capsys.readouterr().out
    assert line.startswith("RLMesh serving model on 127.0.0.1:0 (startup: ")
    assert "listen " in line


def test_serving_a_model_puts_its_describe_envelope_on_the_handshake(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    import rlmesh
    import rlmesh._peer_info as peer_info

    monkeypatch.setattr(serve, "_marks", {})
    stamped: dict[str, str] = {}
    monkeypatch.setattr(
        peer_info,
        "register_python_peer_info",
        lambda extra=None: stamped.update(extra or {}),
    )

    class Tiny(rlmesh.Model):
        def predict(self, observation: object) -> int:
            return 0

        def serve(self, address: str, *, options: object = None) -> None:
            pass

    monkeypatch.setattr(serve, "_resolve_model", lambda source, binding: Tiny())

    serve.serve_model(Tiny, "127.0.0.1:0")

    envelope = json.loads(stamped[rlmesh.DESCRIBE_METADATA_KEY])
    assert envelope["kind"] == "model"
    assert envelope["target"]["qualname"].endswith("Tiny")
    assert capsys.readouterr().err == ""


def test_an_undescribable_target_still_serves(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import rlmesh
    import rlmesh._peer_info as peer_info

    monkeypatch.setattr(serve, "_marks", {})
    stamped: dict[str, str] = {}
    monkeypatch.setattr(
        peer_info,
        "register_python_peer_info",
        lambda extra=None: stamped.update(extra or {}),
    )
    served = type(
        "Served", (), {"serve": lambda self, address, *, options=None: None}
    )()
    monkeypatch.setattr(serve, "_resolve_model", lambda source, binding: served)

    serve.serve_model(object(), "127.0.0.1:0")

    assert rlmesh.DESCRIBE_METADATA_KEY not in stamped
    assert "carries no describe envelope" in capsys.readouterr().err
