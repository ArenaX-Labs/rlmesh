# Quickstart Example

Run this loop first. One process owns a Gymnasium environment. Another process connects to it as a
remote environment.

## Serve CartPole

```bash
uv run python examples/python/quickstart/serve_gymnasium.py
```

The server script is intentionally thin:

```python
env = gym.make("CartPole-v1")
EnvServer(env, "127.0.0.1:5555").serve()
```

`EnvServer` reads the Gymnasium spaces, exposes the environment endpoint, and keeps the environment
dependencies in the server process.

## Connect a Client

In another terminal:

```bash
uv run python examples/python/quickstart/eval.py
```

The client uses the same Gymnasium-style loop:

```python
env = RemoteEnv("127.0.0.1:5555")
obs, info = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

The client does not import the environment package. It only connects to the endpoint and uses the
spaces reported by the server.

## Swap Environments

Use another Gymnasium registration:

```bash
uv run python examples/python/quickstart/serve_gymnasium.py --env-id Acrobot-v1
```

Use another address when running multiple servers:

```bash
uv run python examples/python/quickstart/serve_gymnasium.py --address 127.0.0.1:5556
```
