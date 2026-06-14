"""Generalized env-to-model adapters (experimental).

Instead of writing one bespoke adapter per environment/model pair,
environments *tag* their observation and action spaces once with
:class:`EnvTags`, models describe their expected inputs and outputs
once with :class:`ModelSpec`, and :func:`resolve` derives the concrete
preprocessing/postprocessing for any pair by matching semantic roles::

    env tags ──┐                              ┌── model spec
    (roles +   │   resolve() matches by role  │   (full payload +
     a few     ├───────────────► Adapter ◄────┤    action layout)
     facts)    │   widths/dtypes from spaces  │
    obs/action │                              │
      spaces ──┘                              └── transform_obs / transform_action

::

    import rlmesh.adapters as adapt

    adapter = adapt.resolve(tags, obs_space, action_space, model_spec)
    payload = adapter.transform_obs(raw_obs)  # env obs -> model input
    action = adapter.transform_action(output)  # model output -> env action

    model = rlmesh.numpy.Model(adapter.wrap_predict(predict_fn))

The asymmetry is deliberate: an environment only *tags* (roles plus the
few facts spaces cannot carry -- image layout, rotation encoding, an
explicit range); keys, widths, dtypes and bounds are read from the
gymnasium spaces by the native ``join`` step. A model *fully specifies* its
payload.

Tags travel through contract metadata so an adapter can be resolved
from a handshake alone: :func:`tag` publishes them under
:data:`ENV_METADATA_KEY`, and :func:`resolve_from_contract` recovers them
from a remote env's contract. A remotely published model spec must be fully
declarative; custom inputs holding in-process callables are local-only.

Transformations are interpreted from declarative spec data; no code is ever
evaluated from a spec. Bespoke feature engineering plugs in through
:class:`InlineCustomInput` (an in-process callable) or
:class:`EntrypointCustomInput` (an explicitly trusted ``module:callable``
entrypoint). When a pairing needs logic specs cannot
express (e.g. control-space conversion requiring a kinematic model),
subclass :class:`AdapterBase` to provide a fully custom adapter that is
interchangeable with resolved ones.

Resolution and plan application run in the native ``rlmesh-adapters`` core
-- the same implementation behind every language binding, pinned by the
conformance vectors shipped with that crate. This package keeps the
host-language half: spec construction and serialization, entrypoint trust
gating, custom callables, and the custom-adapter base class.

This package requires NumPy (install ``rlmesh[numpy]``). Encoded image
bytes (PNG/JPEG) in observations are decoded natively -- no Pillow.
"""

from .adapter import Adapter, AdapterBase
from .constants import (
    ACTION_DELTA_POS,
    ACTION_DELTA_POS_2,
    ACTION_DELTA_ROT,
    ACTION_DELTA_ROT_2,
    ACTION_GRIPPER,
    ACTION_GRIPPER_2,
    EEF_POS,
    EEF_POS_2,
    EEF_ROT,
    EEF_ROT_2,
    ENV_METADATA_KEY,
    GRIPPER_POS,
    GRIPPER_POS_2,
    IMAGE_PRIMARY,
    IMAGE_SECONDARY,
    IMAGE_WRIST,
    INSTRUCTION,
    JOINT_POS,
    JOINT_VEL,
    MODEL_METADATA_KEY,
)
from .errors import AdapterResolutionError
from .resolver import resolve, resolve_from_contract
from .specs import (
    ROTATION_DIMS,
    ActionComponent,
    ActionLayout,
    CustomEncoding,
    EntrypointCustomInput,
    EnvTags,
    ImageInput,
    ImageLayout,
    ImageTag,
    InlineCustomInput,
    ModelInput,
    ModelSpec,
    ObsTag,
    ObsTags,
    ObsTransform,
    RotationEncoding,
    RotationTransform,
    StateComponent,
    StateField,
    StateInput,
    StateLayout,
    StateTag,
    TextInput,
    TextTag,
)
from .tag import tag

__all__ = [
    "ACTION_DELTA_POS",
    "ACTION_DELTA_POS_2",
    "ACTION_DELTA_ROT",
    "ACTION_DELTA_ROT_2",
    "ACTION_GRIPPER",
    "ACTION_GRIPPER_2",
    "EEF_POS",
    "EEF_POS_2",
    "EEF_ROT",
    "EEF_ROT_2",
    "ENV_METADATA_KEY",
    "GRIPPER_POS",
    "GRIPPER_POS_2",
    "IMAGE_PRIMARY",
    "IMAGE_SECONDARY",
    "IMAGE_WRIST",
    "INSTRUCTION",
    "JOINT_POS",
    "JOINT_VEL",
    "MODEL_METADATA_KEY",
    "ROTATION_DIMS",
    "ActionComponent",
    "ActionLayout",
    "Adapter",
    "AdapterBase",
    "AdapterResolutionError",
    "CustomEncoding",
    "EntrypointCustomInput",
    "EnvTags",
    "IOAdapter",
    "ImageInput",
    "ImageLayout",
    "ImageTag",
    "InlineCustomInput",
    "ModelInput",
    "ModelSpec",
    "ObsTag",
    "ObsTags",
    "ObsTransform",
    "RotationEncoding",
    "RotationTransform",
    "StateComponent",
    "StateField",
    "StateInput",
    "StateLayout",
    "StateTag",
    "TextInput",
    "TextTag",
    "resolve",
    "resolve_from_contract",
    "tag",
]


def __getattr__(name: str) -> object:
    """Deprecation shim: ``IOAdapter`` was renamed to ``Adapter``."""
    if name == "IOAdapter":
        import warnings

        warnings.warn(
            "rlmesh.adapters.IOAdapter is deprecated; use rlmesh.adapters.Adapter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Adapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
