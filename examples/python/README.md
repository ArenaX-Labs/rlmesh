# Python Examples

Each example is small and copyable, and shares the repository environment. Run any of them with `uv run python examples/python/<dir>/<file>.py` from the repository root. Examples that need their own lockfile are demos and live in the separate `rlmesh-examples` repository, not here.

Most server/client examples default to `127.0.0.1:5555`; start the server in one terminal and the client in another.

- [`quickstart/`](quickstart): the canonical loop — serve an env, connect an evaluator.
- [`adapters/`](adapters): tag an env, then run a model against it through a resolved IO adapter.
- [`byo_container/`](byo_container): hand-written Dockerfiles for an env and a model image (run the same image locally or on the hosted platform).
- [`sandbox/`](sandbox): start an owned Docker-backed environment process (needs Docker).
- [`vla_adapters/`](vla_adapters): tag-driven adapters across many model × env pairs.
