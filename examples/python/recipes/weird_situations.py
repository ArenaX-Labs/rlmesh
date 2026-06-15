"""A tour of less-obvious recipe situations, each validated dependency-free.

Run it: every recipe is ``check()``ed without importing any heavy sim deps, and
the registry is printed. Nothing here needs Docker, a GPU, or the env packages.

    uv run python examples/python/recipes/weird_situations.py

It also imports the class-style ``metaworld_reach`` recipe so the printed
registry shows it alongside the inline ones.
"""

import metaworld_reach  # noqa: F401  -- @register on import (heavy imports stay inside make)
from rlmesh import recipes
from rlmesh.recipes import Build, Fetch, PyMake, Recipe


def register_weird_recipes():
    # 1. A factory that ISN'T gymnasium.make --------------------------------
    # safety-gymnasium builds envs via its own safety_gymnasium.make plus a
    # Gymnasium wrapper, so it's a factory (PyMake), not a gym id. The factory
    # travels by reference; the wrapper lives in safety_serve.make, never in the
    # recipe. (This is the case the chat got wrong by reaching for GymMake.)
    recipes.register(
        "safety/point-goal",
        factory="safety_serve:make",
        packages=["safety-gymnasium==1.0.0"],
    )

    # 2. One image, many tasks (from_recipe) --------------------------------
    # A build-only base holds the heavy image; each task references it and adds
    # only its own make/setup, so the whole suite shares ONE built image instead
    # of rebuilding per task.
    recipes.register(
        Recipe(
            name="libero/base",
            build=Build(
                base="nvcr.io/nvidia/pytorch:24.01-py3",
                fetch=[
                    Fetch(
                        kind="git",
                        repo="https://github.com/Lifelong-Robot-Learning/LIBERO.git",
                        ref="a" * 40,  # pin a full commit SHA
                        dest="/opt/LIBERO",
                        pip_install=True,
                    )
                ],
                gpu=True,
            ),
        )
    )
    for task in range(3):
        recipes.register(
            Recipe(
                name=f"libero/spatial-{task}",
                make=PyMake(entrypoint="libero_serve:make", kwargs={"task": task}),
                build=Build(from_recipe="libero/base"),
            )
        )

    # 3. A pre-baked / non-Debian image (the verbatim Dockerfile trapdoor) ---
    # When the structured build vocab doesn't fit (custom distro, hand-tuned
    # image), drop to a verbatim Dockerfile. It's walled off and Installed-only.
    recipes.register(
        Recipe(
            name="vendor/sim",
            make=PyMake(entrypoint="vendor_serve:make"),
            build=Build(
                dockerfile="FROM vendor/sim-runtime:2.3\nRUN /opt/sim/install.sh\n"
            ),
        )
    )

    # 4. A pinned source fetch (reproducibility) ----------------------------
    # A tarball pinned by sha256: the recipe is a reproducible build, not
    # "whatever the URL serves today."
    recipes.register(
        Recipe(
            name="bench/fixed",
            make=PyMake(entrypoint="bench_serve:make"),
            build=Build(
                fetch=[
                    Fetch(
                        kind="url",
                        url="https://example.org/assets.tar.gz",
                        sha256="0" * 64,
                        dest="/opt/assets",
                    )
                ]
            ),
        )
    )


def main():
    register_weird_recipes()

    for name in recipes.registered_names():
        recipes.check(
            recipes.resolve(name)
        )  # round-trip + entrypoint shape, imports nothing
    print("all recipes validated dependency-free (no sim deps imported)\n")

    recipes.pprint_registry()


if __name__ == "__main__":
    main()
