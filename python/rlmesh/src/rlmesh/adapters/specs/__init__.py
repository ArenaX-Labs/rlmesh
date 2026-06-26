"""Declarative spec dataclasses and their vocabulary types, by side.

The env side tags (:class:`EnvTags` + the ``*Tag`` types);
the model side fully specifies (:class:`ModelSpec` + the ``*Input`` types).
"""

from .action import ActionComponent, ActionLayout
from .custom_encoding import CustomEncoding, RotationTransform
from .env_tags import (
    EnvTags,
    ImageTag,
    ObsTag,
    ObsTags,
    StateField,
    StateLayout,
    StateTag,
    TextTag,
)
from .model import ModelSpec
from .model_inputs import (
    EntrypointCustomInput,
    ImageInput,
    InlineCustomInput,
    ModelInput,
    ObsTransform,
    StateComponent,
    StateInput,
    TextInput,
)
from .vocabularies import ROTATION_DIMS, FitMode, ImageLayout, RotationEncoding

__all__ = [
    "ROTATION_DIMS",
    "ActionComponent",
    "ActionLayout",
    "CustomEncoding",
    "EntrypointCustomInput",
    "EnvTags",
    "FitMode",
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
]
