from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Measurement:
    name: str
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    samples: tuple[float, ...]
    iterations: int
    bytes_per_iter: int | None = None
    throughput_mib_s: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["samples"] = list(self.samples)
        return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RLMesh installed-wheel artifact checks."
    )
    parser.add_argument("--environment", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--dependency", action="append", default=[])
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assert_installed_wheel()

    measurements: list[Measurement] = []
    for artifact in args.artifact:
        measurements.extend(
            run_artifact(
                artifact,
                tier=args.tier,
                samples=args.samples,
                warmups=args.warmups,
            )
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": args.environment,
        "python": args.python_version,
        "tier": args.tier,
        "rlmesh": version("rlmesh"),
        "dependencies": list(args.dependency),
        "artifacts": list(args.artifact),
        "warmups": args.warmups,
        "samples": args.samples,
        "measurements": [measurement.to_json() for measurement in measurements],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"report={args.output}")
    for measurement in measurements:
        print(
            f"measurement={measurement.name} "
            f"median_ms={measurement.median_ms:.6f} "
            f"p95_ms={measurement.p95_ms:.6f}"
        )
    return 0


def assert_installed_wheel() -> None:
    wheel = distribution("rlmesh").read_text("WHEEL")
    if not wheel:
        raise AssertionError("installed rlmesh distribution has no WHEEL metadata")


def run_artifact(
    name: str, *, tier: str, samples: int, warmups: int
) -> list[Measurement]:
    if name == "tensor-numpy-view":
        return tensor_numpy_view(tier=tier, samples=samples, warmups=warmups)
    if name == "tensor-torch-view":
        return tensor_torch_view(tier=tier, samples=samples, warmups=warmups)
    if name == "tensor-export-copy":
        return tensor_export_copy(tier=tier, samples=samples, warmups=warmups)
    if name == "tensor-torch-export-copy":
        return tensor_torch_export_copy(tier=tier, samples=samples, warmups=warmups)
    if name == "tensor-jax-view":
        return tensor_jax_view(tier=tier, samples=samples, warmups=warmups)
    if name == "tensor-dlpack-import":
        return tensor_dlpack_import(tier=tier, samples=samples, warmups=warmups)
    if name == "tensor-numpy-encode":
        return tensor_numpy_encode(tier=tier, samples=samples, warmups=warmups)
    raise SystemExit(f"unknown artifact {name!r}")


def tensor_numpy_view(*, tier: str, samples: int, warmups: int) -> list[Measurement]:
    import numpy as np
    from rlmesh import Tensor
    from rlmesh import numpy as rlmesh_numpy

    measurements = []
    for size in sizes_for_tier(tier):
        tensor = Tensor(bytes(size), [size], "uint8")
        array = rlmesh_numpy.asarray(tensor)
        assert isinstance(array, np.ndarray)
        assert array.shape == (size,)
        assert array.dtype == np.uint8
        assert not array.flags.writeable
        assert not array.flags.owndata
        iterations = view_iterations_for_size(size)
        measurements.append(
            measure(
                f"tensor.numpy.asarray/{size_label(size)}",
                lambda tensor=tensor: rlmesh_numpy.asarray(tensor),
                samples=samples,
                warmups=warmups,
                iterations=iterations,
                bytes_per_iter=size,
                metadata={"path": "zero-copy-view"},
            )
        )
    return measurements


def tensor_torch_view(*, tier: str, samples: int, warmups: int) -> list[Measurement]:
    import torch
    from rlmesh import Tensor
    from rlmesh import torch as rlmesh_torch

    measurements = []
    for size in sizes_for_tier(tier):
        tensor = Tensor(bytes(size), [size], "uint8")
        view = rlmesh_torch.as_tensor(tensor)
        assert isinstance(view, torch.Tensor)
        assert tuple(view.shape) == (size,)
        assert view.dtype == torch.uint8
        iterations = view_iterations_for_size(size)
        measurements.append(
            measure(
                f"tensor.torch.as_tensor/{size_label(size)}",
                lambda tensor=tensor: rlmesh_torch.as_tensor(tensor),
                samples=samples,
                warmups=warmups,
                iterations=iterations,
                bytes_per_iter=size,
                metadata={"path": "zero-copy-view"},
            )
        )
    return measurements


def tensor_export_copy(*, tier: str, samples: int, warmups: int) -> list[Measurement]:
    from rlmesh import Tensor

    measurements = []
    for size in sizes_for_tier(tier):
        tensor = Tensor(bytes(size), [size], "uint8")
        iterations = copy_iterations_for_size(size)
        measurements.append(
            measure(
                f"tensor.export.bytes/{size_label(size)}",
                lambda tensor=tensor: bytes(memoryview(tensor)),
                samples=samples,
                warmups=warmups,
                iterations=iterations,
                bytes_per_iter=size,
                metadata={"path": "copy"},
            )
        )
    return measurements


def tensor_torch_export_copy(
    *, tier: str, samples: int, warmups: int
) -> list[Measurement]:
    import torch
    from rlmesh import torch as rlmesh_torch

    measurements = []
    for size in sizes_for_tier(tier):
        tensor = torch.zeros((size,), dtype=torch.uint8)
        iterations = copy_iterations_for_size(size)
        measurements.append(
            measure(
                f"tensor.torch.export/{size_label(size)}",
                lambda tensor=tensor: rlmesh_torch.as_tensor(
                    rlmesh_torch.from_tensor(tensor)
                ),
                samples=samples,
                warmups=warmups,
                iterations=iterations,
                bytes_per_iter=size,
                metadata={"path": "torch-roundtrip"},
            )
        )
    return measurements


def tensor_jax_view(*, tier: str, samples: int, warmups: int) -> list[Measurement]:
    import jax
    from rlmesh import Tensor
    from rlmesh import jax as rlmesh_jax

    measurements = []
    for size in sizes_for_tier(tier):
        tensor = Tensor(bytes(size), [size], "uint8")
        array = rlmesh_jax.asarray(tensor)
        assert isinstance(array, jax.Array)
        assert array.shape == (size,)
        iterations = view_iterations_for_size(size)
        measurements.append(
            measure(
                f"tensor.jax.asarray/{size_label(size)}",
                # block_until_ready keeps async dispatch honest in the timing.
                lambda tensor=tensor: rlmesh_jax.asarray(tensor).block_until_ready(),
                samples=samples,
                warmups=warmups,
                iterations=iterations,
                bytes_per_iter=size,
                metadata={"path": "dlpack-shared"},
            )
        )
    return measurements


def tensor_dlpack_import(*, tier: str, samples: int, warmups: int) -> list[Measurement]:
    import numpy as np
    from rlmesh import Tensor

    measurements = []
    for size in sizes_for_tier(tier):
        array = np.zeros(size, dtype=np.uint8)
        imported = Tensor.from_dlpack(array)
        assert imported.shape == [size]
        iterations = copy_iterations_for_size(size)
        measurements.append(
            measure(
                f"tensor.from_dlpack/{size_label(size)}",
                lambda array=array: Tensor.from_dlpack(array),
                samples=samples,
                warmups=warmups,
                iterations=iterations,
                bytes_per_iter=size,
                metadata={"path": "copy-import"},
            )
        )
    return measurements


def tensor_numpy_encode(*, tier: str, samples: int, warmups: int) -> list[Measurement]:
    import numpy as np
    from rlmesh import Tensor
    from rlmesh import numpy as rlmesh_numpy

    measurements = []
    for size in sizes_for_tier(tier):
        array = np.zeros(size, dtype=np.uint8)
        encoded = rlmesh_numpy.from_array(array)
        assert isinstance(encoded, Tensor)
        iterations = copy_iterations_for_size(size)
        measurements.append(
            measure(
                f"tensor.numpy.from_array/{size_label(size)}",
                lambda array=array: rlmesh_numpy.from_array(array),
                samples=samples,
                warmups=warmups,
                iterations=iterations,
                bytes_per_iter=size,
                metadata={"path": "copy"},
            )
        )
    return measurements


def measure(
    name: str,
    func: Callable[[], object],
    *,
    samples: int,
    warmups: int,
    iterations: int,
    bytes_per_iter: int | None = None,
    metadata: dict[str, object] | None = None,
) -> Measurement:
    for _ in range(warmups):
        run_iterations(func, iterations)
    values = [run_iterations(func, iterations) for _ in range(samples)]
    median_ms = statistics.median(values)
    throughput_mib_s = None
    if bytes_per_iter is not None and median_ms > 0:
        throughput_mib_s = bytes_per_iter / (1024 * 1024) / (median_ms / 1000)
    return Measurement(
        name=name,
        median_ms=median_ms,
        p95_ms=percentile(values, 0.95),
        min_ms=min(values),
        max_ms=max(values),
        samples=tuple(values),
        iterations=iterations,
        bytes_per_iter=bytes_per_iter,
        throughput_mib_s=throughput_mib_s,
        metadata=metadata or {},
    )


def run_iterations(func: Callable[[], object], iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        func()
    return (time.perf_counter() - started) * 1000.0 / iterations


def percentile(samples: list[float], percentile_value: float) -> float:
    values = sorted(samples)
    index = int((len(values) - 1) * percentile_value + 0.999999)
    return values[min(index, len(values) - 1)]


def sizes_for_tier(tier: str) -> tuple[int, ...]:
    _ = tier
    return (1024, 1024 * 1024, 8 * 1024 * 1024)


def view_iterations_for_size(size: int) -> int:
    if size <= 1024:
        return 2000
    return 100


def copy_iterations_for_size(size: int) -> int:
    if size <= 1024:
        return 500
    if size <= 1024 * 1024:
        return 25
    return 5


def size_label(size: int) -> str:
    if size >= 1024 * 1024:
        mib = size / (1024 * 1024)
        return f"{mib:g}MiB"
    if size >= 1024:
        kib = size / 1024
        return f"{kib:g}KiB"
    return f"{size}B"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
