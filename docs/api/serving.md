# Serving Helpers

```{note}
`rlmesh._serving` is **experimental** in this beta and not yet part of the public surface. Use it
with version pinning; signatures may still change before the stable release.
```

`rlmesh._serving` exposes a small surface for constructing an environment to serve through
{py:class}`~rlmesh.EnvServer`. It promotes the loaders previously hidden in `rlmesh._cli.serve_env`
so that scripts and downstream runners can build an environment by Gymnasium id or by
`module:callable` entrypoint.

```python
import rlmesh
from rlmesh import _serving

env = _serving.load_env("CartPole-v1")
rlmesh.EnvServer(env).serve()
```

## Loaders

```{eval-rst}
.. autofunction:: rlmesh._serving.load_env
```

```{eval-rst}
.. autofunction:: rlmesh._serving.load_env_entrypoint
```

```{eval-rst}
.. autofunction:: rlmesh._serving.import_packages
```
