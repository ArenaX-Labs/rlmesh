"""Declarative spec dataclasses and their vocabulary types, by side.

The env side tags (:class:`EnvTags` + the ``*Tag`` types);
the model side fully specifies (:class:`ModelSpec` + the ``*Input`` types).
"""

from .action import ActionComponent, ActionLayout
from .env_tags import (
    EnvTags,
    ImageTag,
    ObsTag,
    StateTag,
    TextTag,
)
from .layouts import ImageLayout
from .model import ModelSpec
from .model_inputs import (
    CustomInput,
    ImageInput,
    ModelInput,
    ObsTransform,
    StateComponent,
    StateInput,
    TextInput,
)
from .rotations import ROTATION_DIMS, RotationEncoding

__all__ = [
    "ROTATION_DIMS",
    "ActionComponent",
    "ActionLayout",
    "CustomInput",
    "EnvTags",
    "ImageInput",
    "ImageLayout",
    "ImageTag",
    "ModelInput",
    "ModelSpec",
    "ObsTag",
    "ObsTransform",
    "RotationEncoding",
    "StateComponent",
    "StateInput",
    "StateTag",
    "TextInput",
    "TextTag",
]
