# RLMesh System Fixtures

This is the private fixture package installed into clean system-test venvs. It contains envs,
models, artifact checks, and trace drivers used by `tools/rlmesh_system`.

## Adding Fixtures

Add envs under `src/rlmesh_system_fixtures/envs/` and register the factory:

```python
from rlmesh_system_fixtures.registry import env_fixture


@env_fixture("my-env")
def make_env() -> object:
    ...
```

Add models under `src/rlmesh_system_fixtures/models/` and register the policy:

```python
from rlmesh_system_fixtures.registry import model_fixture


@model_fixture("my_model.zero")
def zero_policy(observation: object) -> object:
    ...
```

Then reference the keys from `tests/system/profiles/*.toml`:

```toml
model = "my_model.zero"
env = { fixture = "my-env" }
```

Keep heavy optional imports inside the fixture function body so basic profiles remain lightweight.
