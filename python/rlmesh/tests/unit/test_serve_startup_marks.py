"""The serve entrypoint stamps startup phase marks on the serving line and the
handshake, so a container's model wait splits into imports, construction, and
listen without a profiler."""

from __future__ import annotations

from typing import Any, cast

import pytest
import rlmesh
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
    # Tiny declares no spec: the managed probe cannot drive it, and serve says so.
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "declares no spec" in err, err


def test_serving_a_specced_model_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import rlmesh
    import rlmesh._peer_info as peer_info
    import rlmesh.adapters as adapt

    monkeypatch.setattr(serve, "_marks", {})
    monkeypatch.setattr(peer_info, "register_python_peer_info", lambda extra=None: None)

    class Specced(rlmesh.Model):
        spec = adapt.ModelSpec(
            input=adapt.Text(role=adapt.INSTRUCTION),
            output=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1)),
        )

        def predict(self, observation: object) -> int:
            return 0

        def serve(self, address: str, *, options: object = None) -> None:
            pass

    monkeypatch.setattr(serve, "_resolve_model", lambda source, binding: Specced())

    serve.serve_model(Specced, "127.0.0.1:0")

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


class _Declared(rlmesh.Model):
    """Declares an edition on the class; a served option must override it."""

    workflow_edition = rlmesh.current_workflow_edition()

    def predict(self, observation: object) -> int:
        return 0

    def serve(self, address: str, *, options: object = None) -> None:
        self.options = options


@pytest.mark.parametrize("env_var", [None, ""])
def test_envelope_preferred_edition_is_what_the_serve_options_declare(
    monkeypatch: pytest.MonkeyPatch, env_var: str | None
) -> None:
    """The envelope reports the declaration the handshake sends, not a re-resolve.

    The wire `preferred_workflow_edition` is `ServeOptions.workflow_edition`
    (else this build's newest); the platform admits on the envelope, so the
    two must agree on every rung: an explicit option, and an empty
    `RLMESH_WORKFLOW_EDITION` that silences the class declaration.
    """
    import json

    import rlmesh._peer_info as peer_info
    from rlmesh._rlmesh import build_info

    monkeypatch.setattr(serve, "_marks", {})
    stamped: dict[str, str] = {}
    monkeypatch.setattr(
        peer_info,
        "register_python_peer_info",
        lambda extra=None: stamped.update(extra or {}),
    )
    if env_var is None:
        monkeypatch.delenv("RLMESH_WORKFLOW_EDITION", raising=False)
    else:
        monkeypatch.setenv("RLMESH_WORKFLOW_EDITION", env_var)
    served = _Declared()
    monkeypatch.setattr(serve, "_resolve_model", lambda source, binding: served)

    option = None if env_var == "" else build_info().workflow_edition
    serve.serve_model(_Declared, "127.0.0.1:0", workflow_edition=option)

    envelope = json.loads(stamped[rlmesh.DESCRIBE_METADATA_KEY])
    options = cast("Any", served.options)
    # What the handshake declares: the option's value, else this build's newest.
    wire = (
        options.workflow_edition if options is not None else None
    ) or build_info().workflow_edition
    assert envelope["runtime"]["preferred_workflow_edition"] == wire
    if env_var == "":
        # The deliberate float: the class declaration is not consulted.
        assert options is None or options.workflow_edition is None
    else:
        assert options.workflow_edition == option
    assert envelope["runtime"]["supported_workflow_editions"] == list(
        build_info().supported_workflow_editions
    )
