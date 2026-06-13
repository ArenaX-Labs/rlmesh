# Recipes

Recipes are experimental in this beta. A recipe is an inert, JSON-serializable description of how to
_construct_ an environment: the factory that builds it, the dependencies it needs, and the
construct-time setup it expects. Name a recipe once and anyone can build the same environment
locally or in a sandbox by referring to that name.

A recipe has three phases:

- **make** — the factory: a Gymnasium id (`GymMake`), a `module:callable` entrypoint (`PyMake`), or
  a Hugging Face source (`HfMake`).
- **build** — the image: base image, apt packages, pip steps, git fetches, and your own package. It
  renders to a Dockerfile; nothing in this phase runs in your process.
- **setup** — construct-time data: environment variables and file writes applied when the
  environment is created.

For the exact classes and methods, see {doc}`../api/recipes`.

## Make an environment

`rlmesh.make` is a strict superset of `gymnasium.make`. Pass a Gymnasium id, a registered recipe
name, or a `Recipe`. It constructs the environment in the current process:

```python
import rlmesh

env = rlmesh.make("CartPole-v1")
obs, info = env.reset(seed=0)
```

## Name a recipe

Register a recipe once, then refer to it by name everywhere:

```python
from rlmesh.recipes import Build, GymMake, PipInstall, Recipe, Requires, register

register(Recipe(
    name="safety/point-goal",
    make=GymMake(env_id="SafetyPointGoal1-v0"),
    build=Build(pip=[PipInstall(packages=["safety-gymnasium==1.0.0"])]),
    requires=Requires(imports=["safety_gymnasium"]),
))

env = rlmesh.make("safety/point-goal")
```

The same name works in a sandbox, which builds the recipe's image and runs the environment in its
own container:

```python
from rlmesh.numpy import SandboxEnv

with SandboxEnv("safety/point-goal") as env:
    obs, info = env.reset(seed=0)
```

## Heavier environments

For an environment with native dependencies, GPU access, a cloned upstream repo, or your own
package, write a `module:callable` factory and declare the build as data:

```python
from rlmesh.recipes import (
    Build, Fetch, PipInstall, ProjectInstall, PyMake, Recipe, Setup, register,
)

register(Recipe(
    name="libero/object",
    make=PyMake(entrypoint="libero.env:Environment"),
    build=Build(
        system=["cmake", "g++", "libegl1-mesa-dev"],     # apt build dependencies
        system_runtime=["libgl1", "libglib2.0-0"],       # apt runtime libraries
        fetch=[Fetch(kind="git", repo="https://github.com/.../LIBERO.git",
                     ref="<full-commit-sha>", dest="/opt/LIBERO",
                     pip_requirements="requirements.txt", pip_install=True)],
        project=ProjectInstall(src=".", dest="/opt/robot_env", include=["assets/**"]),
        pip=[PipInstall(["torch", "torchvision"],
                        index_url="https://download.pytorch.org/whl/cu124"),
             PipInstall(["robosuite==1.4.1"])],
        env={"MUJOCO_GL": "egl"}, pythonpath=["/opt/LIBERO"], gpu=True, run_as=1000,
    ),
    setup=Setup(env={"LIBERO_TASK": "libero_object/0"}),
))
```

The procedural construction logic lives in your factory function. The recipe stays declarative data,
so it can be validated at construction, shipped as JSON, and rendered to a Dockerfile by the
language-neutral core without executing anything.

## Share one build across tasks

When many tasks share a build, declare the build once as a base recipe and reference it with
`from_recipe`. Every task in the family resolves to the same image:

```python
from rlmesh.recipes import Build, PipInstall, PyMake, Recipe, register

register(Recipe(name="droid/base", build=Build(
    base="nvidia/cuda:12.4.1-runtime-ubuntu22.04", gpu=True,
    pip=[PipInstall(["torch==2.4.0"])],
)))

for scene in (1, 2, 3):
    register(Recipe(
        name=f"droid/scene{scene}",
        make=PyMake(entrypoint=f"robot_env:scene{scene}"),
        build=Build(from_recipe="droid/base"),
    ))
```

Per-task differences belong in `make` and `setup`. They ride the runtime payload and never
invalidate the shared image, so the whole family builds once.

## Serve a recipe

`EnvServer` accepts a recipe directly; it builds the environment and serves it:

```python
from rlmesh import EnvServer
from rlmesh.recipes import GymMake, Recipe

EnvServer(Recipe(name="cart/pole", make=GymMake(env_id="CartPole-v1"))).serve()
```

## Migrate an existing environment

`scaffold_from_pyproject` reads a project's `pyproject.toml` and emits a recipe to review: pip steps
from the dependencies and per-package indices, a base image and GPU guess, and `TODO` markers where
it cannot infer:

```python
from pathlib import Path

from rlmesh.recipes import scaffold_from_pyproject

result = scaffold_from_pyproject(
    "acme/robot", "robot_env:make", Path("pyproject.toml").read_text()
)
Path("recipes.py").write_text(result.recipe_source)
```

## Safety

The build phase runs `apt`, `git`, and `pip` only inside `docker build`, never in your process; the
`setup` phase is data-only. A recipe resolved from a local `register` call or an installed package
is trusted to build anything. A recipe handed in as raw data is treated as untrusted: its build must
pin every git fetch and download, version-pin and allowlist its pip steps, and may not carry raw
shell commands. Keep recipe sources reviewed and pinned for reproducible evaluations.
