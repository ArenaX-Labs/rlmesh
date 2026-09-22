"""``rlmesh.build_info()`` reports this build's identity; ``__build__`` is its alias."""

from __future__ import annotations

import re

import pytest
import rlmesh

#: A sealed workflow edition base: a year and a month.
BASE = re.compile(r"^\d{4}\.(0[1-9]|1[0-2])$")

STRING_FIELDS = (
    "version",
    "protocol_generation",
    "workflow_edition",
    "workflow_edition_base",
    "build_cohort",
    "build_source",
)


def test_fields_are_typed() -> None:
    info = rlmesh.build_info()

    assert isinstance(info, rlmesh.BuildInfo)
    for name in STRING_FIELDS:
        value = getattr(info, name)
        assert isinstance(value, str) and value, name
    assert info.git is None or isinstance(info.git, str)


def test_version_and_generation() -> None:
    info = rlmesh.build_info()

    assert info.version == rlmesh._rlmesh.__version__
    assert info.protocol_generation == "rlmesh-wire-v1"


def test_base_is_year_month_and_prefixes_edition() -> None:
    info = rlmesh.build_info()

    assert BASE.match(info.workflow_edition_base)
    assert info.workflow_edition_base == rlmesh.current_workflow_edition()
    assert info.workflow_edition.startswith(info.workflow_edition_base)
    suffix = info.workflow_edition.removeprefix(info.workflow_edition_base)
    if info.build_cohort == "stable":
        assert suffix == ""
    else:
        assert suffix == f"-{info.build_cohort}"


def test_git_is_the_dev_cohort_token() -> None:
    info = rlmesh.build_info()

    if info.build_source == "git":
        assert info.git is not None
        assert info.build_cohort == f"dev.{info.git}"
    else:
        assert info.build_source in ("release", "package")
        assert info.git is None


def test_repr_is_stable() -> None:
    info = rlmesh.build_info()
    git = "None" if info.git is None else f'"{info.git}"'

    assert repr(info) == (
        f'BuildInfo(version="{info.version}", '
        f'protocol_generation="{info.protocol_generation}", '
        f'workflow_edition="{info.workflow_edition}", '
        f'workflow_edition_base="{info.workflow_edition_base}", '
        f'build_cohort="{info.build_cohort}", '
        f'build_source="{info.build_source}", '
        f"git={git})"
    )
    assert repr(info) == repr(rlmesh.build_info())


def test_dunder_build_is_a_deprecated_alias() -> None:
    with pytest.warns(DeprecationWarning, match=r"build_info\(\)\.workflow_edition"):
        value = rlmesh.__build__

    assert value == rlmesh.build_info().workflow_edition
    assert "__build__" in dir(rlmesh)
