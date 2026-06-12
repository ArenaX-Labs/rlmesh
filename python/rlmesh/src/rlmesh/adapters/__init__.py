"""Generalized env-to-model IO adapters (experimental).

Instead of writing one bespoke adapter per environment/model pair,
environments describe their observation and action formats once with
:class:`EnvIOSpec`, models describe their expected inputs and outputs once
with :class:`ModelIOSpec`, and :func:`resolve` derives the concrete
preprocessing/postprocessing for any pair by matching semantic roles::

    import rlmesh.adapters as adapt

    adapter = adapt.resolve(env_spec, model_spec)
    payload = adapter.transform_obs(raw_obs)  # env obs -> model input
    action = adapter.transform_action(output)  # model output -> env action

    model = rlmesh.numpy.Model(adapter.wrap_predict(predict_fn))

Specs travel through contract metadata so an adapter can be resolved from
handshakes alone: environments publish theirs under ``ENV_METADATA_KEY``
(:meth:`EnvIOSpec.to_metadata` / :meth:`EnvIOSpec.from_metadata`), and
served models can publish theirs under ``MODEL_METADATA_KEY``
(:meth:`ModelIOSpec.to_metadata` / :meth:`ModelIOSpec.from_metadata`).
A remotely published model spec must be fully declarative; custom inputs
holding in-process callables are local-only.

Transformations are interpreted from declarative spec data; no code is ever
evaluated from a spec. Bespoke feature engineering plugs in through
:class:`CustomInput` as an in-process callable or an explicitly trusted
``module:callable`` entrypoint. When a pairing needs logic specs cannot
express (e.g. control-space conversion requiring a kinematic model),
subclass :class:`AdapterBase` to provide a fully custom adapter that is
interchangeable with resolved ones.

Resolution and plan application run in the native ``rlmesh-adapters``
core -- the same implementation behind every language binding, pinned by
the conformance vectors shipped with that crate. This package keeps
the host-language half: spec construction and serialization, entrypoint
trust gating, custom callables, and the custom-adapter base class.

Package layout:

- :mod:`rlmesh.adapters.constants` -- semantic roles and metadata keys.
- :mod:`rlmesh.adapters.specs` -- the declarative spec dataclasses
  (``env``, ``model``, ``action``), their vocabulary types (rotations,
  layouts), and their dict round-trips.
- :mod:`rlmesh.adapters.resolver` -- serializes specs into the native
  core and wraps the resolved plan.
- :mod:`rlmesh.adapters.adapter` -- the runtime :class:`IOAdapter` and
  the :class:`AdapterBase` escape hatch.
- :mod:`rlmesh.adapters.errors` -- error types.
- :mod:`rlmesh.adapters.helpers` -- array and native-bridge internals.

This package requires NumPy (install ``rlmesh[numpy]``). Encoded image
bytes (PNG/JPEG) in observations are decoded natively -- no Pillow.
"""

from .adapter import AdapterBase, IOAdapter
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
from .resolver import resolve
from .specs import (
    ROTATION_DIMS,
    ActionComponent,
    ActionLayout,
    CustomInput,
    EnvFeature,
    EnvImage,
    EnvIOSpec,
    EnvState,
    EnvText,
    ImageInput,
    ImageLayout,
    ModelInput,
    ModelIOSpec,
    ObsTransform,
    RotationEncoding,
    StateComponent,
    StateInput,
    TextInput,
)

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
    "AdapterBase",
    "AdapterResolutionError",
    "CustomInput",
    "EnvFeature",
    "EnvIOSpec",
    "EnvImage",
    "EnvState",
    "EnvText",
    "IOAdapter",
    "ImageInput",
    "ImageLayout",
    "ModelIOSpec",
    "ModelInput",
    "ObsTransform",
    "RotationEncoding",
    "StateComponent",
    "StateInput",
    "TextInput",
    "resolve",
]
