from __future__ import annotations

from collections.abc import Sequence

import pytest

from scripts.dev.python_synced import CHECK, HINT, SYNC, main


class Runner:
    """Answers the sync check with `checked`, recording every command."""

    def __init__(self, checked: int) -> None:
        self.checked = checked
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> int:
        self.commands.append(list(command))
        return self.checked if list(command) == CHECK else 0


def test_a_synced_environment_passes_without_a_sync() -> None:
    runner = Runner(checked=0)
    assert main([], run=runner, sync=runner) == 0
    assert main(["--sync"], run=runner, sync=runner) == 0
    assert runner.commands == [CHECK, CHECK]


def test_an_unsynced_environment_stops_with_one_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = Runner(checked=1)
    assert main([], run=runner, sync=runner) == 1
    assert runner.commands == [CHECK]
    assert capsys.readouterr().err == f"{HINT}\n"


def test_a_task_holding_the_heavy_lock_syncs_it() -> None:
    runner = Runner(checked=1)
    assert main(["--sync"], run=runner, sync=runner) == 0
    assert runner.commands == [CHECK, SYNC]


def test_the_check_reads_the_lock_and_leaves_out_the_workspace_packages() -> None:
    # Offline (the lock as it is) and blind to the editable rlmesh build,
    # which setup:python rebuilds whenever its Rust sources move.
    assert {"--check", "--frozen", "--no-install-workspace", "--inexact"} <= set(CHECK)
    assert SYNC == ["mise", "run", "setup:python"]
