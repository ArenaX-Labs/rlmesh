# Serving Helpers

```{note}
`rlmesh.serving` is **experimental** in this beta. Use it with version pinning; signatures may still
change before the stable release.
```

`rlmesh.serving` exposes a small surface for constructing an environment to serve through
{py:class}`~rlmesh.EnvServer`. It promotes the loaders previously hidden in `rlmesh._cli.serve_env`
so that scripts and downstream runners can build an environment by Gymnasium id or by
`module:callable` entrypoint without depending on private modules.

```python
import rlmesh

env = rlmesh.serving.load_env("CartPole-v1")
rlmesh.EnvServer(env).serve()
```

## Loaders

```{eval-rst}
.. autofunction:: rlmesh.serving.load_env
```

```{eval-rst}
.. autofunction:: rlmesh.serving.load_env_entrypoint
```

```{eval-rst}
.. autofunction:: rlmesh.serving.import_packages
```
