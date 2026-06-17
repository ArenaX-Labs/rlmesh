# Quickstart Example

The smallest copyable RLMesh server/client loop. Full walkthrough:
[docs.rlmesh.dev quickstart](https://docs.rlmesh.dev/quickstart) (or `docs/quickstart.md`).

From the repository root, serve an env:

```bash
uv run python examples/python/quickstart/serve_gymnasium.py   # Gymnasium CartPole-v1
```

In another terminal, connect a client:

```bash
uv run python examples/python/quickstart/eval.py              # sampled-action eval
```

The files:

- `serve_gymnasium.py`: serve any Gymnasium registration (`--env-id Acrobot-v1`).
- `serve.py`: serve a dependency-light custom `CounterEnv` (no Gymnasium).
- `eval.py`: connect a client and step with sampled actions.
- `eval_many.py`: one evaluator across multiple endpoints.
- `model.py`: run a tiny model worker against an endpoint.

To copy outside the repo, install the published package:

```bash
pip install "rlmesh[gymnasium,numpy]"
```
