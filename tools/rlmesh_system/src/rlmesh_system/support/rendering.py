from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover - optional presentation dependency
    box = None
    Console = None
    Panel = None
    Table = None

from rlmesh_system.support.command import CommandError, output_tail
from rlmesh_system.support.manifest import (
    EnvironmentSpec,
    ProfileSpec,
    ScenarioSpec,
)
from rlmesh_system.support.reports import EnvironmentResult, RegressionResult


class Renderer:
    def __init__(self, *, plain: bool) -> None:
        self.rich = not plain and Console is not None
        self.console = Console() if self.rich and Console is not None else None

    def print(self, text: str = "") -> None:
        if self.console is not None:
            self.console.print(text)
        else:
            print(text)

    def print_profiles(
        self,
        profiles: dict[str, ProfileSpec],
        environments: dict[str, EnvironmentSpec],
        scenarios: dict[str, ScenarioSpec],
    ) -> None:
        if self.console is None or Table is None or box is None:
            print_profiles_plain(profiles, environments, scenarios)
            return

        profile_table = Table(title="Validation Profiles", box=box.SIMPLE)
        profile_table.add_column("Profile", style="bold")
        profile_table.add_column("Description")
        profile_table.add_column("Environments")
        for profile in profiles.values():
            profile_table.add_row(
                profile.name,
                profile.description,
                "\n".join(profile.environments),
            )
        self.console.print(profile_table)

        env_table = Table(title="Validation Environments", box=box.SIMPLE)
        env_table.add_column("Environment", style="bold")
        env_table.add_column("Tier")
        env_table.add_column("Python")
        env_table.add_column("Dependencies")
        env_table.add_column("Scenarios")
        for environment in environments.values():
            env_table.add_row(
                environment.name,
                environment.tier,
                environment.python,
                ", ".join(environment.dependencies) or "<none>",
                "\n".join(environment.scenarios),
            )
        self.console.print(env_table)

        scenario_table = Table(title="System Scenario Kinds", box=box.SIMPLE)
        scenario_table.add_column("Kind", style="bold")
        scenario_table.add_column("Scenarios")
        for kind in ("trace", "artifact", "external"):
            names = [
                scenario.name
                for scenario in scenarios.values()
                if scenario.kind == kind
            ]
            if names:
                scenario_table.add_row(kind, ", ".join(names))
        self.console.print(scenario_table)

    def environment_start(
        self, environment: EnvironmentSpec, scenarios: list[ScenarioSpec]
    ) -> None:
        names = ", ".join(scenario.name for scenario in scenarios)
        if self.rich:
            self.print(
                f"\n[bold cyan]==> {environment.name}[/] "
                f"(python {environment.python}; {names})"
            )
        else:
            self.print(
                f"\n==> {environment.name} (python {environment.python}; {names})"
            )

    def environment_skip(self, environment: EnvironmentSpec, reason: str) -> None:
        self.print(f"{environment.name}: skipped ({reason})")

    def step(self, label: str) -> None:
        self.print(f"  [cyan]-[/] {label}" if self.rich else f"  - {label}")

    def command(self, command: list[str]) -> None:
        text = "$ " + " ".join(command)
        self.print(f"[dim]{text}[/dim]" if self.rich else text)

    def command_output(self, output: str) -> None:
        if output:
            self.print(output.rstrip())

    def command_running(self, label: str, elapsed: float, log_path: Path) -> None:
        text = f"    still running {label} ({elapsed:.0f}s); log: {log_path}"
        self.print(f"[dim]{text}[/dim]" if self.rich else text)

    def environment_done(
        self, environment: EnvironmentSpec, elapsed: float, report_path: Path
    ) -> None:
        if self.rich:
            self.print(
                f"[green]{environment.name}: ok[/] "
                f"[dim]({elapsed:.2f}s, report {report_path})[/dim]"
            )
        else:
            self.print(f"{environment.name}: ok ({elapsed:.2f}s, report {report_path})")

    def environment_failed(
        self, environment: EnvironmentSpec, elapsed: float, error: CommandError
    ) -> None:
        if self.rich:
            self.print(
                f"[red]{environment.name}: failed[/] "
                f"[dim]({elapsed:.2f}s, exit {error.returncode})[/dim]"
            )
        else:
            self.print(
                f"{environment.name}: failed ({elapsed:.2f}s, exit {error.returncode})"
            )
        self.failure_details(error)

    def failure_details(self, error: CommandError) -> None:
        tail = output_tail(error.output)
        details = f"log: {error.log_path}"
        if tail:
            details = f"{details}\n\n{tail}"

        if self.console is not None and Panel is not None:
            self.console.print(
                Panel(
                    details,
                    title=f"failure exit {error.returncode}",
                    border_style="red",
                )
            )
            return

        self.print(details)

    def run_summary(
        self, results: list[EnvironmentResult], report_path: Path | None
    ) -> None:
        if self.console is None or Table is None or box is None:
            print("\nSummary:")
            for result in results:
                status = "ok" if result.ok else "FAIL"
                print(f"  {status} {result.name} ({result.elapsed:.2f}s)")
            if report_path is not None:
                print(f"Report: {report_path}")
            return

        table = Table(title="Validation Summary", box=box.SIMPLE)
        table.add_column("Status")
        table.add_column("Environment")
        table.add_column("Python")
        table.add_column("Elapsed", justify="right")
        table.add_column("Report")
        for result in results:
            status = "[green]ok[/green]" if result.ok else "[red]FAIL[/red]"
            table.add_row(
                status,
                result.name,
                result.python,
                f"{result.elapsed:.2f}s",
                str(result.report_path or result.log_path or ""),
            )
        self.console.print(table)
        if report_path is not None:
            self.console.print(f"[bold]Report:[/] {report_path}")

    def measurement_summary(self, measurements: list[dict[str, Any]]) -> None:
        if not measurements:
            return
        if self.console is None or Table is None or box is None:
            print("\nMeasurements:")
            for measurement in measurements:
                throughput = measurement.get("throughput_mib_s")
                suffix = (
                    f" throughput={throughput:.2f} MiB/s"
                    if isinstance(throughput, int | float)
                    else ""
                )
                print(
                    f"  {measurement['environment']} {measurement['name']} "
                    f"median={measurement['median_ms']:.4f}ms "
                    f"p95={measurement['p95_ms']:.4f}ms{suffix}"
                )
            return

        table = Table(title="Validation Measurements", box=box.SIMPLE)
        table.add_column("Environment")
        table.add_column("Measurement")
        table.add_column("Median", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("Throughput", justify="right")
        for measurement in measurements:
            throughput = measurement.get("throughput_mib_s")
            table.add_row(
                str(measurement["environment"]),
                str(measurement["name"]),
                f"{measurement['median_ms']:.4f}ms",
                f"{measurement['p95_ms']:.4f}ms",
                f"{throughput:.2f} MiB/s"
                if isinstance(throughput, int | float)
                else "",
            )
        self.console.print(table)

    def regression_summary(self, results: list[RegressionResult]) -> None:
        if self.console is None or Table is None or box is None:
            print("\nBaseline Comparison:")
            for result in results:
                print(
                    f"  {result.status} {result.environment} {result.measurement}: "
                    f"{result.reason}"
                )
            return

        table = Table(title="Baseline Comparison", box=box.SIMPLE)
        table.add_column("Status")
        table.add_column("Environment")
        table.add_column("Measurement")
        table.add_column("Current", justify="right")
        table.add_column("Baseline", justify="right")
        table.add_column("Reason")
        for result in results:
            status = (
                "[red]REGRESSION[/red]"
                if result.status == "regression"
                else result.status
            )
            table.add_row(
                status,
                result.environment,
                result.measurement,
                f"{result.current_ms:.4f}ms" if result.current_ms is not None else "",
                f"{result.baseline_ms:.4f}ms" if result.baseline_ms is not None else "",
                result.reason,
            )
        self.console.print(table)


def print_profiles_plain(
    profiles: dict[str, ProfileSpec],
    environments: dict[str, EnvironmentSpec],
    scenarios: dict[str, ScenarioSpec],
) -> None:
    print("Profiles:")
    for profile in profiles.values():
        print(f"  {profile.name}: {profile.description}")
        for environment_name in profile.environments:
            environment = environments[environment_name]
            print(
                "    "
                f"{environment.name}: tier={environment.tier} "
                f"python={environment.python} "
                f"scenarios={','.join(environment.scenarios)}"
            )
    print()
    print("Environments:")
    for environment in environments.values():
        deps = ", ".join(environment.dependencies) or "<none>"
        print(
            f"  {environment.name}: tier={environment.tier} "
            f"python={environment.python} deps={deps}"
        )
    print()
    print("Scenario Kinds:")
    for kind in ("trace", "artifact", "external"):
        names = [
            scenario.name for scenario in scenarios.values() if scenario.kind == kind
        ]
        if names:
            print(f"  {kind}: {', '.join(names)}")
