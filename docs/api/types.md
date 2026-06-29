# Types

The `rlmesh.types` module defines the structural protocols that {py:class}`~rlmesh.EnvServer` accepts and the shared value aliases used by dependency-free clients. The protocols are structural, so any object with the right methods satisfies them; you do not subclass anything. Reach for them to type-annotate an environment or a value, or to check what `EnvServer` expects. For authoring an environment against these protocols see {doc}`../user-guide/environments`.

```{eval-rst}
.. automodule:: rlmesh.types
   :members:
   :show-inheritance:
```
