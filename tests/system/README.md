# RLMesh System Tests

System tests validate installed RLMesh Python wheels in clean `uv`
environments. The runner lives in `tools/rlmesh_system`; this directory holds
scenario profiles and deterministic trace baselines.

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

IsaacSim and longer soak hooks are intentionally not part of this profile set
yet; add them later as explicit profiles when the simulator launch contract is
ready.
