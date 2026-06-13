# Recipes

Recipes are experimental in this beta. A recipe is an inert, JSON-serializable description of how to
_construct_ an environment, **decoupled from where it runs**. You can author a recipe for an
environment you cannot run locally -- an IsaacSim recipe written on a Mac, say -- because the recipe
only references your construction code by name; the code runs later, in a container. See
{doc}`../api/recipes` for the exact classes and methods.

**Which rung am I on?**

- Already pip-installed the env? `rlmesh.make("id")`.
- Want it installed for you in a clean container? `SandboxEnv("id", packages=[...])`.
- Naming one for reuse? `rlmesh.register("name", gym=...)` or `register("name", factory=...)`.
- Your own construction code? Subclass `rlmesh.EnvRecipe`.

Everything below those is the inert form your recipe lowers to; you rarely write it by hand.

## Run a built-in

`rlmesh.make` is a strict superset of `gymnasium.make`. If the package is already installed, this is
the whole story:

```python
import rlmesh

env = rlmesh.make("CartPole-v1")
obs, info = env.reset(seed=0)
```

## Run it isolated, and install the dependency for you

When an env needs a `pip install` you don't want in your process, the flat one-liner builds a
container and hands you a remote env. `packages=` are installed in the container; `imports=` are
module names imported so the env registers itself (not packages -- those go in `packages=`):

```python
from rlmesh.numpy import SandboxEnv  # pick your array backend (numpy / torch / jax)

with SandboxEnv("ALE/Breakout-v5", packages=["ale-py"], imports=["ale_py"]) as env:
    obs, info = env.reset(seed=0)
```

## Name it for reuse

`rlmesh.register` and `rlmesh.make` are the pair: register a recipe once, then anyone refers to it
by name -- locally or in a sandbox. Pass exactly one of `gym=` (a gym id) or `factory=` (a
`module:callable`):

```python
import rlmesh

rlmesh.register("atari/breakout", gym="ALE/Breakout-v5", packages=["ale-py"], imports=["ale_py"])

env = rlmesh.make("atari/breakout")                       # in-process
# with SandboxEnv("atari/breakout") as env: ...           # in a container
```

## Point at a factory you already have

`GymMake` only covers ids that `gymnasium.make` can build. An environment with its _own_ `make` or
one that needs a wrapper -- like `safety_gymnasium` -- is constructed by a factory you reference by
name:

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

When your `make()` _is_ your code -- a real build, GPU, your own package -- subclass
`rlmesh.EnvRecipe`. It co-locates the recipe data (the `name`/`build`/`setup` class attributes) with
the construction code (`make()`, and an optional `prepare()` hook), and projects to an inert recipe:

```python
from __future__ import annotations  # required: keeps the class importable where deps are absent

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
        ...  # construct-time CODE (download a checkpoint, warm a cache) -- runs before make()

    def make(self, task="libero_object/0", **kwargs):
        import libero.env                       # heavy import INSIDE make() (never at author time)
        return libero.env.Environment(task=task, **kwargs)


rlmesh.register(LiberoObject)
```

Put every heavy import inside `make()`/`prepare()`. `register(LiberoObject)` reads the class
attributes and records the entrypoint string -- it never instantiates the class or imports its
dependencies, so this works on a machine that cannot run the env. Validate a recipe without
importing anything with `LiberoObject.check()`.

`register` also works as a decorator -- `@rlmesh.register` above the class registers it on import
and returns the class unchanged:

```python
@rlmesh.register
class LiberoObject(rlmesh.EnvRecipe):
    name = "acme/libero-object"
    ...
```

It must live in an importable module (not a `__main__` script), because the container imports the
factory by reference. To see what is registered, `rlmesh.recipes.pprint_registry()` prints the
recipes grouped by namespace, and `rlmesh.recipes.registry()` returns a read-only `name -> recipe`
view.

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

Per-task differences belong in `make`/`setup`; they ride the runtime payload and never invalidate
the shared image.

## A non-Debian base, or wrapping an existing Dockerfile

`system`/`system_runtime` are installed with `apt`, so a structured `Build` targets a Debian/Ubuntu
base (the defaults -- `python:3.11-slim` and the `nvidia/cuda` images -- are). For another distro,
or to wrap a Dockerfile you already have, set `build.dockerfile` to a verbatim body (the deriver
appends the rlmesh entrypoint). It is mutually exclusive with the structured build fields:

```python
from rlmesh.recipes import Build

Build(dockerfile="FROM alpine:3.20\nRUN apk add --no-cache python3 py3-pip\n...")
```

## Serving

`rlmesh.EnvServer` accepts the same `str | Recipe | EnvRecipe` source as `make`; an `EnvRecipe` (or
`Recipe`) is built before it is served:

```python
from rlmesh import EnvServer

EnvServer(LiberoObject).serve()
```

## Safety

The build phase runs `apt`, `git`, and `pip` only inside `docker build`, never in your process; the
`setup` phase is data-only. A recipe from your installed/loaded code (a `register` call, an
installed package, an `EnvRecipe`) is trusted to build anything. A recipe handed in as raw data from
an untrusted source is gated: its build must pin every git fetch and download, version-pin and
allowlist its pip steps, and may not carry raw shell commands or a verbatim Dockerfile. Keep recipe
sources reviewed and pinned for reproducible evaluations.
