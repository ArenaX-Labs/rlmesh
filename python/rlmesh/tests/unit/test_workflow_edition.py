from __future__ import annotations

import re

import pytest
import rlmesh
from rlmesh._editions import (
    current_workflow_edition,
    declared_edition,
    declaring,
    resolve_workflow_edition,
)

UNKNOWN_EDITION = "2099.01"

EDITION = current_workflow_edition()


def test_every_build_reports_the_bare_edition() -> None:
    assert re.fullmatch(r"\d{4}\.\d{2}", EDITION)
    assert rlmesh.current_workflow_edition() == EDITION
    assert rlmesh.build_info().workflow_edition == EDITION


@pytest.mark.parametrize("declared", [None, "  ", f" {EDITION} "])
def test_the_declaration_else_the_current_edition(declared: str | None) -> None:
    assert resolve_workflow_edition(declared) == EDITION


def test_the_environment_declares_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLMESH_WORKFLOW_EDITION", UNKNOWN_EDITION)
    assert resolve_workflow_edition(None) == EDITION


def test_an_unknown_edition_is_refused_naming_the_known_ones_and_the_remedy() -> None:
    pattern = (
        rf'workflow edition "{UNKNOWN_EDITION}" is unknown to this rlmesh build, '
        rf"which knows {re.escape(EDITION)}; upgrade the rlmesh runtime"
    )
    with pytest.raises(ValueError, match=pattern):
        _ = resolve_workflow_edition(UNKNOWN_EDITION)


def test_serve_options_declare_nothing() -> None:
    with pytest.raises(TypeError):
        _ = rlmesh.ServeOptions(workflow_edition=EDITION)  # type: ignore[call-arg]


def test_a_server_built_inside_declaring_takes_its_declaration() -> None:
    assert declared_edition() is None
    with declaring(EDITION):
        assert declared_edition() == EDITION
        with declaring(None):
            assert declared_edition() is None
    assert declared_edition() is None
