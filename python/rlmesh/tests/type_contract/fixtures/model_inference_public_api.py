"""Pins the inferred model generics doctrine.

An annotated predict callable flows its observation/action types onto the
model's generic parameters (and from there onto its ``Session``); an
unannotated lambda pays no ceremony (framework-default observation side);
duck-typed policy objects and class sources keep the permissive fallback; a
``Model`` subclass binds the framework default value types, with the chunk
corners admitting the optional ``execution_horizon`` in both spellings
(second positional parameter, and keyword-only).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import rlmesh
from rlmesh import numpy as rlmesh_numpy
from typing_extensions import assert_type

Obs = dict[str, np.ndarray[Any, Any]]
Act = np.ndarray[Any, Any]


def typed_predict(observation: Obs) -> Act:
    return observation["state"]


typed_model = rlmesh_numpy.Model(typed_predict)
assert_type(typed_model, rlmesh_numpy.Model[Obs, Act])

untyped_model = rlmesh_numpy.Model(lambda observation: 0)
assert_type(untyped_model, rlmesh_numpy.Model[rlmesh_numpy.NumpyValue, int])


class DuckPolicy:
    def predict(self, observation: Obs) -> Act:
        return observation["state"]


duck_model = rlmesh_numpy.Model(DuckPolicy())
assert_type(
    duck_model, rlmesh_numpy.Model[rlmesh_numpy.NumpyValue, rlmesh_numpy.NumpyValue]
)
class_model = rlmesh_numpy.Model(DuckPolicy)
assert_type(
    class_model, rlmesh_numpy.Model[rlmesh_numpy.NumpyValue, rlmesh_numpy.NumpyValue]
)


class ChunkedPolicy(rlmesh_numpy.Model):
    def predict(self, observation: rlmesh_numpy.NumpyValue) -> rlmesh_numpy.NumpyValue:
        return observation

    def predict_chunk(
        self, observation: rlmesh_numpy.NumpyValue, execution_horizon: int = 1
    ) -> rlmesh_numpy.NumpyValue:
        return observation

    def predict_batch(
        self, observations: rlmesh_numpy.NumpyValue
    ) -> rlmesh_numpy.NumpyValue:
        return observations

    def predict_chunk_batch(
        self,
        observations: rlmesh_numpy.NumpyValue,
        *,
        execution_horizon: int = 1,
    ) -> rlmesh_numpy.NumpyValue:
        return observations


def _session_types(model: rlmesh_numpy.Model[Obs, Act], policy: ChunkedPolicy) -> None:
    assert_type(model.session("127.0.0.1:5555"), rlmesh.Session[Obs, Act])
    assert_type(rlmesh.session(model, "127.0.0.1:5555"), rlmesh.Session[Obs, Act])
    assert_type(
        rlmesh.session(typed_predict, "127.0.0.1:5555"), rlmesh.Session[Obs, Act]
    )
    assert_type(
        rlmesh.session(rlmesh.RANDOM_SAMPLE, "127.0.0.1:5555"),
        rlmesh.Session[Any, Any],
    )
    assert_type(policy.predict(0), rlmesh_numpy.NumpyValue)
    assert_type(
        policy.session("127.0.0.1:5555"),
        rlmesh.Session[rlmesh_numpy.NumpyValue, rlmesh_numpy.NumpyValue],
    )
