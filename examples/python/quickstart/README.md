# Quickstart Example

This is the smallest copyable RLMesh environment server and sampled-action client. The server
exposes a tiny counter environment at `127.0.0.1:5555`; the client connects with
`from rlmesh.numpy import RemoteEnv`.

From the repository root, start the server:

```bash
uv run python examples/python/quickstart/serve.py
```

In another terminal, run the client:

```bash
uv run python examples/python/quickstart/eval.py
```

To copy this example into a separate project:

```bash
pip install --pre "rlmesh[numpy]"
python serve.py
python eval.py
```

Replace `CounterEnv` in `serve.py` with any Gymnasium-style environment object that implements
`reset`, `step`, `close`, `observation_space`, and `action_space`.
