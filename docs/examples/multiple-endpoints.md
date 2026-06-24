# Multiple Endpoints

Run more than one environment endpoint, then connect one evaluator to all of them.

The runnable evaluator is {source}`examples/python/quickstart/eval_many.py <examples/python/quickstart/eval_many.py>`.

## Start Two Servers

Each environment is served by `rlmesh.EnvServer`. The Gymnasium server wraps a registered environment:

```python
import gymnasium as gym
from rlmesh import EnvServer

env = gym.make(args.env_id)
server = EnvServer(env, args.address)
server.serve()
```

The custom server wraps a plain Python object with the same shape:

```python
import rlmesh

server = rlmesh.EnvServer(CounterEnv(), args.address)
server.serve()
```

Terminal one owns Gymnasium `CartPole-v1`, terminal two owns the small custom `CounterEnv`:

```bash
uv run python examples/python/quickstart/serve_gymnasium.py --address 127.0.0.1:5555
uv run python examples/python/quickstart/serve.py --address 127.0.0.1:5556
```

## Evaluate Both

`eval_many.py` opens a `RemoteEnv` per address and runs the same sampled-action loop against each endpoint:

```python
def evaluate(address: str, max_steps: int) -> str:
    from rlmesh.numpy import RemoteEnv

    env = RemoteEnv(address)
    try:
        lines = [f"{address}: connected"]
        obs, info = env.reset(seed=0)
        for step in range(1, max_steps + 1):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            lines.append(f"{address}: step={step} reward={reward:.3f}")
            if term or trunc:
                lines.append(f"{address}: episode complete")
                break
        else:
            lines.append(f"{address}: stopped after {max_steps} steps")
        return "\n".join(lines)
    finally:
        env.close()
```

The addresses are passed in and each one is evaluated on its own thread:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=len(args.addresses)) as executor:
    futures = [
        executor.submit(evaluate, address, args.max_steps) for address in args.addresses
    ]
    for future in futures:
        print(future.result())
```

Run it in terminal three:

```bash
uv run python examples/python/quickstart/eval_many.py \
  127.0.0.1:5555 \
  127.0.0.1:5556
```

That is one evaluator running across multiple environment runtimes, locally. The client shape is the same for every endpoint, whether the server wraps a Gymnasium environment or a custom object.
