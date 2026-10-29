"""A session checks no model endpoint token yet: a set one is refused."""

from __future__ import annotations

import pytest
import rlmesh


def test_a_set_model_endpoint_token_names_safety_step_6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RLMESH_MODEL_ENDPOINT_TOKEN", "deployment-token")
    with pytest.raises(ValueError, match="safety step 6"):
        rlmesh.Model(lambda observation: 0).serve(":0", name="model")
