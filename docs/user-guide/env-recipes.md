# Environment Recipes

Recipes are experimental in this beta. A recipe is an inert, JSON-serializable description of how to
construct an environment, decoupled from where it runs. You can author a recipe for an environment
you cannot run locally (an IsaacSim recipe written on a Mac, say). The recipe only references your
construction code by name, so the code runs later, in a container. See {doc}`../api/env-recipes` for the
full class and method contract.

For authoring a model, see {doc}`model-recipes`.

## Choosing an entry point

- Already pip-installed the env? `rlmesh.make("id")`.
- Want it installed for you in a clean container? `SandboxEnv("id", packages=[...])`.
- Naming one for reuse? `rlmesh.register("name", gym=...)` or `register("name", factory=...)`.
- Your own construction code? Subclass `rlmesh.EnvRecipe`.

Each one lowers to the same inert recipe form. You rarely write that form by hand.

## Run a built-in

For a Gymnasium id, `rlmesh.make` is a drop-in for `gymnasium.make` (it forwards the same kwargs and
adds recipe sources). If the package is already installed, this is the whole story:

```python
import rlmesh

env = rlmesh.make("CartPole-v1")
obs, info = env.reset(seed=0)
```

## Run it isolated, with the dependency installed for you

When an env needs a `pip install` you don't want in your process, the one-liner builds a container
and hands you a remote env. `packages=` are installed in the container; `imports=` are module names
imported so the env registers itself:

```python
from rlmesh.numpy import SandboxEnv  # pick your array backend (numpy / torch / jax)

with SandboxEnv("ALE/Breakout-v5", packages=["ale-py"], imports=["ale_py"]) as env:
    obs, info = env.reset(seed=0)
```

## Name it for reuse

`rlmesh.register` and `rlmesh.make` are the pair. Register a recipe once, then refer to it by name,
locally or in a sandbox. Pass exactly one of `gym=` (a gym id) or `factory=` (a `module:callable`):

```python
import rlmesh

rlmesh.register("atari/breakout", gym="ALE/Breakout-v5", packages=["ale-py"], imports=["ale_py"])

env = rlmesh.make("atari/breakout")                       # in-process
# with SandboxEnv("atari/breakout") as env: ...           # in a container
```

## Point at a factory you already have

`GymMake` only covers ids that `gymnasium.make` can build. An environment with its own `make`, or one
that needs a wrapper like `safety_gymnasium`, is constructed by a factory you reference by name:

```python
# safety_env.py  (in your package; the recipe references it as a string)
import safety_gymnasium
from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium


def make_point_goal(env_id="SafetyHalfCheetahVelocity-v1", **kwargs):
    return SafetyGymnasium2Gymnasium(safety_gymnasium.make(env_id, **kwargs))
```

```python
rlmesh.register(
    "safety/half-cheetah",
    factory="safety_env:make_point_goal",
    packages=["safety-gymnasium==1.0.0"],
)
```

## Author with a class: `EnvRecipe`

When your `make()` is your own construction code (a build step, GPU, your own package), subclass
`rlmesh.EnvRecipe`. The class attributes (`name`/`build`/`setup`) hold the recipe data; `make()` and
the optional `prepare()` hook hold the construction code. The class lowers to an inert recipe.

```python
from __future__ import annotations

import rlmesh
from rlmesh.recipes import Build, Fetch, PipInstall, ProjectInstall


class LiberoObject(rlmesh.EnvRecipe):
    name = "libero/object"
    build = Build(
        system=["cmake", "g++", "libegl1-mesa-dev"],
        fetch=[Fetch(kind="git", repo="https://github.com/.../LIBERO.git",
                     ref="<full-commit-sha>", dest="/opt/LIBERO",
                     pip_requirements="requirements.txt", pip_install=True)],
        project=ProjectInstall(src=".", dest="/opt/robot_env", include=["assets/**"]),
        pip=[PipInstall(["torch", "torchvision"],
                        index_url="https://download.pytorch.org/whl/cu124")],
        env={"MUJOCO_GL": "egl"}, pythonpath=["/opt/LIBERO"], gpu=True,
    )

    def prepare(self):
        ...  # construct-time work (download a checkpoint, warm a cache); runs before make()

    def make(self, task="libero_object/0", **kwargs):
        import libero.env
        return libero.env.Environment(task=task, **kwargs)


rlmesh.register(LiberoObject)
```

```{important}
Put every heavy import inside `make()`/`prepare()`. `register(LiberoObject)` reads the class
attributes and records the entrypoint string. It never instantiates the class or imports its
dependencies, so authoring works on a machine that cannot run the env. The
`from __future__ import annotations` line keeps the class importable where those deps are absent.
Validate without importing anything with `LiberoObject.check()`.
```

State set in `prepare()` is read in `make()` through instance attributes. The recipe instance is then
discarded, so it cannot own anything the running env needs. See {doc}`../api/env-recipes` for the full
`prepare()`/`make()` lifecycle and registry-introspection functions.

```{warning}
The recipe instance is discarded the moment `make()` returns. A resource opened in `prepare()` (a
file, subprocess, or socket) dies with it unless the returned env holds it. Create runtime resources
in `make()` and release them in `env.close()`.
```

`register` also works as a decorator. `@rlmesh.register` above the class registers it on import and
returns the class unchanged:

```python
@rlmesh.register
class LiberoObject(rlmesh.EnvRecipe):
    name = "acme/libero-object"
    ...
```

The class must live in an importable module, not a `__main__` script, because the container imports
the factory by reference.

## Mount a runtime asset

A large asset (a scene/USD pack, a dataset, a teacher checkpoint) is a runtime mount, never baked
into the image — the env-side twin of a {doc}`model recipe <model-recipes>`'s weights. Declare an
`ArtifactInput` on `inputs`, then resolve its local path inside `make()`/`prepare()` with
`self.input_path(name)`:

```python
from rlmesh.recipes import ArtifactInput


class Scene(rlmesh.EnvRecipe):
    name = "robot/scene"
    inputs = (ArtifactInput("scene", "/assets/scene", uri="hf://org/scene@<sha>"),)

    def make(self, **kwargs):
        import robot_env
        return robot_env.from_assets(self.input_path("scene"))
```

A `uri="hf://org/repo@sha"` resolves through the rlmesh cache (`pip install --pre "rlmesh[hf]"` on
the host); `local_dir=` bind-mounts a host directory into the sandbox at the input's target path.
Only the authored class form carries `inputs` — a gym/hf source env has no `input_path`.

## Share one build across tasks

Declare a heavy build once as a base recipe; each task references it with `from_recipe`. Every task
in the family resolves to the same image:

```python
from rlmesh.recipes import Build, PipInstall, Recipe

# A build-only base recipe (make=None) holds the shared build:
rlmesh.register(Recipe(name="droid/base", build=Build(
    base="nvidia/cuda:12.4.1-runtime-ubuntu22.04", gpu=True, pip=[PipInstall(["torch==2.4.0"])])))


class Scene1(rlmesh.EnvRecipe):
    name = "droid/scene1"
    build = Build(from_recipe="droid/base")        # reuse the base build; add your own make()

    def make(self, **kwargs):
        import robot_env
        return robot_env.scene1()
```

Per-task differences belong in `make`/`setup`. They ride the runtime payload and never invalidate the
shared image.

## A non-Debian base, or wrapping an existing Dockerfile

`system`/`system_runtime` are installed with `apt`, so a structured `Build` targets a Debian or Ubuntu
base. The defaults (`python:3.11-slim` and the `nvidia/cuda` images) qualify. For another distro, or
to wrap a Dockerfile you already have, set `build.dockerfile` to a verbatim body. The deriver appends
the rlmesh entrypoint. It is mutually exclusive with the structured build fields:

```python
from rlmesh.recipes import Build

Build(dockerfile="FROM alpine:3.20\nRUN apk add --no-cache python3 py3-pip\n...")
```

## Serving

`rlmesh.EnvServer` accepts the same `str | Recipe | EnvRecipe` source as `make`. An `EnvRecipe` or
`Recipe` is built before it is served:

```python
from rlmesh import EnvServer

EnvServer(LiberoObject).serve()
```

## Safety

The build phase runs `apt`, `git`, and `pip` only inside `docker build`, never in your process; the
`setup` phase is data-only.

```{important}
A recipe from your installed or loaded code (a `register` call, an installed package, an `EnvRecipe`)
is trusted to build anything. A recipe handed in as raw data from an untrusted source is gated: its
build must pin every git fetch and download, version-pin and allowlist its pip steps, and may not
carry raw shell commands or a verbatim Dockerfile.
```

Keep recipe sources reviewed and pinned for reproducible evaluations.
