# Examples

The examples are small on purpose. Each one shows one part of the model-environment loop.

- {doc}`examples/quickstart`: serve Gymnasium `CartPole-v1`, connect one evaluator.
- {doc}`examples/custom-work`: drop in a custom environment or tiny model worker.
- {doc}`examples/sandboxes`: start an owned Docker-backed environment process.
- {doc}`examples/isolated-dependencies`: keep heavier environment dependencies isolated.
- {doc}`examples/multiple-endpoints`: run one evaluator across multiple environment endpoints.
- {doc}`examples/adapters`: pair any model with any environment through declarative IO adapters.

```{toctree}
:hidden:
:maxdepth: 1

examples/quickstart
examples/custom-work
examples/sandboxes
examples/isolated-dependencies
examples/multiple-endpoints
examples/adapters
```
