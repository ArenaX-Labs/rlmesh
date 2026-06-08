# Multiple Endpoints

Run more than one environment endpoint, then connect one evaluator to all of them.

## Start Two Servers

Terminal one:

```bash
uv run python examples/python/quickstart/serve_gymnasium.py --address 127.0.0.1:5555
```

Terminal two:

```bash
uv run python examples/python/quickstart/serve.py --address 127.0.0.1:5556
```

The first server owns Gymnasium `CartPole-v1`. The second owns the small custom `CounterEnv`.

## Evaluate Both

Terminal three:

```bash
uv run python examples/python/quickstart/eval_many.py \
  127.0.0.1:5555 \
  127.0.0.1:5556
```

`eval_many.py` opens a `RemoteEnv` for each address and runs the same sampled-action loop against
each endpoint.

This is the simplest local version of running one evaluator across multiple environment runtimes.
