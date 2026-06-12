# RLMesh System Tests

System tests validate installed RLMesh Python wheels in clean `uv` environments. The runner lives in
`tools/rlmesh_system`; this directory holds scenario profiles and deterministic trace baselines.

## Layout

- `profiles/*.toml`: profile, environment, dependency, and scenario definitions.
- `traces/`: committed deterministic trace baselines.

The private env/model fixture package installed into each clean venv lives in
`tools/rlmesh_system_fixtures`.

## Commands

```bash
mise run test:system:list
mise run test:system -- --dry-run
mise run test:system
mise run test:system:clean
```

The base task defaults to `basic` and forwards runner arguments:

```bash
mise run test:system -- --profile gymnasium
mise run test:system -- --profile torch
mise run test:system -- --profile mujoco
mise run test:system -- --profile heavy
mise run test:system -- --kind trace
mise run test:system -- --kind artifact --baseline target/python-validation/reports/baseline.json
mise run test:system -- --baseline baseline.json --fail-on-regression
```

Direct usage:

```bash
rlmesh-system list
rlmesh-system run --profile basic --dry-run
rlmesh-system run --profile basic
rlmesh-system compare baseline.json current.json
rlmesh-system clean
```

Reports are written to `target/python-validation/reports`.

Profiles can reference fixture envs and models by registry key:

```toml
model = "discrete.zero"
env = { fixture = "counter", kwargs = { limit = 3 } }
```

Real Gymnasium environments stay explicit:

```toml
model = "gymnasium.pendulum_zero_numpy"
env = { gym = "Pendulum-v1", packages = ["gymnasium"] }
```

IsaacSim and longer soak hooks are intentionally not part of this profile set yet; add them later as
explicit profiles when the simulator launch contract is ready.

## Performance Drift

The `perf` profile times the tensor conversion paths (NumPy/Torch/JAX views, encode/import copies)
at 1 KiB / 1 MiB / 8 MiB in clean installed-wheel venvs:

```bash
mise run perf:baseline   # run the perf profile and store the local baseline
mise run perf:check      # re-run and fail on drift against that baseline
```

Baselines are machine-specific and live untracked at `target/python-validation/perf-baseline.json`.
Capture a fresh baseline on a quiet machine before a refactor; run `perf:check` after. Thresholds
are warn-first (see `thresholds_for` in `rlmesh_system.support.reports`): views gate at ~15-25%
relative drift with small absolute floors, copy paths also watch MiB/s throughput.

Version drift: `mise run test:python:floors --perf` runs the same benchmarks inside the
dependency-floor environment (python 3.10, numpy 1.22, torch 1.11, jax 0.4.24) and compares against
the local baseline warn-only, so a framework version that suddenly costs milliseconds shows up in
the same report format.

Expected shape of results (what "healthy" looks like):

- `tensor.numpy.asarray` and `tensor.torch.as_tensor` are zero-copy views: median times stay flat as
  size grows 8000x; reported throughput therefore _rises_ with size. If view times start scaling
  with size, a copy snuck in.
- `tensor.jax.asarray` imports over DLPack; XLA shares RLMesh's 64-byte-aligned storage, which every
  Python-visible tensor uses (constructor, wire decode, and codec handoff). Expect flat times equal
  to jax's fixed dispatch overhead (~20 us); if this row starts scaling with size, an unaligned
  buffer crept back in and XLA is copying.
- `tensor.export.bytes`, `tensor.numpy.from_array`, `tensor.from_dlpack`, and `tensor.torch.export`
  are copies: times scale linearly with size and throughput plateaus at memory bandwidth.
