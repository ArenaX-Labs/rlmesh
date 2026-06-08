# Quickstart Example

This is the smallest copyable RLMesh server/client loop. Start with Gymnasium `CartPole-v1`, then
swap in another registration or a custom Gymnasium-style object.

From the repository root, start the server:

```bash
uv run python examples/python/quickstart/serve_gymnasium.py
```

In another terminal, run the client:

```bash
uv run python examples/python/quickstart/eval.py
```

To copy this example into a separate project:

```bash
pip install --pre "rlmesh[gymnasium,numpy]"
python serve_gymnasium.py
python eval.py
```

Use `--env-id` for another Gymnasium registration:

```bash
python serve_gymnasium.py --env-id Acrobot-v1
```

For a dependency-light custom environment, run `serve.py`. It exposes a tiny `CounterEnv` object
that implements `reset`, `step`, `close`, `observation_space`, and `action_space`.
