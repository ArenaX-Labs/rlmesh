# Spaces

RLMesh spaces are Python wrappers around native `SpaceSpec` values. They provide the familiar
`sample`, `contains`, and `seed` methods while keeping the spec available for transport and
conversion.

## Base Types

```{eval-rst}
.. autoclass:: rlmesh.spaces.Space
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.spaces.SpaceAdapter
   :members:
   :show-inheritance:
```

## Conversion Helpers

```{eval-rst}
.. autofunction:: rlmesh.spaces.from_gymnasium_space
```

```{eval-rst}
.. autofunction:: rlmesh.spaces.to_gymnasium_space
```

```{eval-rst}
.. autofunction:: rlmesh.spaces.space_from_spec
```

## Fundamental Spaces

```{eval-rst}
.. autoclass:: rlmesh.spaces.Box
   :members:
   :inherited-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.spaces.Discrete
   :members:
   :inherited-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.spaces.MultiBinary
   :members:
   :inherited-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.spaces.MultiDiscrete
   :members:
   :inherited-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.spaces.Text
   :members:
   :inherited-members:
   :show-inheritance:
```

## Composite Spaces

```{eval-rst}
.. autoclass:: rlmesh.spaces.Tuple
   :members:
   :inherited-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.spaces.Dict
   :members:
   :inherited-members:
   :show-inheritance:
```
