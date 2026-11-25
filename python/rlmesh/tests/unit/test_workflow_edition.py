from __future__ import annotations

import re

import pytest
import rlmesh
from rlmesh._editions import (
    current_workflow_edition,
    resolve_workflow_edition,
    serve_options_declaring,
)
from rlmesh._rlmesh import ServeOptions

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
    assert serve_options_declaring().workflow_edition == EDITION


def test_an_unknown_edition_is_refused_naming_the_known_ones_and_the_remedy() -> None:
    pattern = (
        rf'workflow edition "{UNKNOWN_EDITION}" is unknown to this rlmesh build, '
        rf"which knows {re.escape(EDITION)}; upgrade the rlmesh runtime"
    )
    with pytest.raises(ValueError, match=pattern):
        _ = resolve_workflow_edition(UNKNOWN_EDITION)
    with pytest.raises(ValueError, match=pattern):
        _ = serve_options_declaring(ServeOptions(workflow_edition=UNKNOWN_EDITION))


def test_serve_options_carry_the_resolved_edition() -> None:
    assert serve_options_declaring().workflow_edition == EDITION
    assert serve_options_declaring(declared=EDITION).workflow_edition == EDITION
    options = ServeOptions(close_timeout_seconds=2.0)
    stamped = serve_options_declaring(options)
    assert stamped.workflow_edition == EDITION
    assert stamped.close_timeout_seconds == 2.0
    with pytest.raises(ValueError, match=UNKNOWN_EDITION):
        _ = serve_options_declaring(declared=UNKNOWN_EDITION)
    assert ServeOptions(workflow_edition=" ").workflow_edition is None
    assert (
        serve_options_declaring(
            ServeOptions(workflow_edition=EDITION), declared=UNKNOWN_EDITION
        ).workflow_edition
        == EDITION
    )
