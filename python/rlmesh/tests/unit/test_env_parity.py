"""Env authoring changes that align Python with the native languages (ENV-P*)."""

from __future__ import annotations

import pytest
import rlmesh
from rlmesh._client._endpoint import normalize_bind_address


def test_path_with_tcp_transport_is_refused():
    """ENV-P3: a Unix path cannot be served over tcp."""
    with pytest.raises(ValueError, match="transport='tcp'"):
        normalize_bind_address(path="/tmp/env.sock", transport="tcp")
    assert normalize_bind_address(path="/tmp/env.sock") == "unix:///tmp/env.sock"


def test_trial_index_reads_only_floats_that_fit_int64():
    """ENV-P9: an integral float outside int64 is no ordinal."""
    assert rlmesh.trial_index({"trial_index": 4.0}) == 4
    assert rlmesh.trial_index({"trial_index": -(2.0**63)}) == -(2**63)
    assert rlmesh.trial_index({"trial_index": 1e19}) is None
    assert rlmesh.trial_index({"trial_index": 2.0**63}) is None


class _Spaceless:
    def reset(self, *, seed=None, options=None):
        return 0, {}

    def step(self, action):
        return 0, 0.0, False, False, {}

    def close(self):
        pass


def test_an_env_without_spaces_is_a_type_error():
    """ENV-P4: missing spaces are TypeError, checked before tags."""
    with pytest.raises(TypeError, match="env must have 'observation_space'"):
        rlmesh.EnvServer(_Spaceless(), "127.0.0.1:0")
