"""Keys used when specs travel inside contract metadata mappings.

Keys are versioned like protobuf packages: within ``v1`` the JSON spec
format evolves additively only (new optional fields with defaults), and a
breaking format change ships under a new ``v2`` key. Publishers may carry
multiple versions in one metadata mapping during a migration; readers
dispatch on the key alone, without parsing payloads.

Values are defined once, in the ``rlmesh-adapters`` crate (``v1/keys.rs``);
this module re-exports them through the native bindings.
"""

from ..._rlmesh import ENV_METADATA_KEY, MODEL_METADATA_KEY

__all__ = ["ENV_METADATA_KEY", "MODEL_METADATA_KEY"]
