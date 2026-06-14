# Recipe Examples

Recipes define _how an environment is constructed_, decoupled from where it runs. These are small
and copyable.

## Fold a render-only env's frame into its observation

Some envs (MetaWorld, many MuJoCo tasks) return a flat state array and expose the camera image only
through `env.render()` -- the image is not in the observation. A recipe fixes this in one place:
`make()` returns the env already wrapped with Gymnasium's `AddRenderObservation`, and `build` pins
the render backend (`MUJOCO_GL`) so off-screen rendering works in the container.

- [`metaworld_reach.py`](metaworld_reach.py): the recipe. An `EnvRecipe` in an importable module (so
  the factory can travel by reference into the container), auto-registered with `@rlmesh.register`
  so it resolves by name.
- [`render_into_obs.py`](render_into_obs.py): validates the recipe dependency-free (`check()` runs
  without importing MetaWorld -- authoring != running) and demonstrates the render→obs mechanic on a
  tiny stub env. **Runs with no MetaWorld and no Docker:**

  ```bash
  uv run python examples/python/recipes/render_into_obs.py
  ```

  The observation goes from `Box(39,)` (state only) to `Dict(state=Box(39,), pixels=Box(84,84,3))`
  (proprioception + camera, together). Use `render_only=True` for pixels-only.

- [`serve_metaworld.py`](serve_metaworld.py): builds the recipe in a Docker-backed sandbox and rolls
  out an episode. **Needs Docker:**

  ```bash
  uv run python examples/python/recipes/serve_metaworld.py
  ```

## Author IsaacSim where IsaacSim can't be imported

[`isaacsim_franka.py`](isaacsim_franka.py) is the headline _authoring != running_ case: a GPU recipe
you can write on a Mac where `import isaaclab` fails. The recipe never imports it — the heavy
imports live inside `make()`/`prepare()`, which only run in the built container. It also shows a
`prepare()` construct-time hook (launch the headless simulator) and "vectorize inside the factory."

## A tour of weirder situations

[`weird_situations.py`](weird_situations.py) validates a spread of less-obvious recipes
dependency-free (no Docker, no GPU, no env packages) and prints the registry:

```bash
uv run python examples/python/recipes/weird_situations.py
```

- a factory that **isn't** `gymnasium.make` (safety-gymnasium's own `make` + wrapper → `factory=`);
- **one image, many tasks** via `build.from_recipe` (a LIBERO suite sharing a single build);
- the **verbatim Dockerfile** trapdoor for a pre-baked / non-Debian image;
- a **pinned source fetch** (sha256 / commit SHA) for a reproducible build.
