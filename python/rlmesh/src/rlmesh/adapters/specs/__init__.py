"""Declarative spec dataclasses and their vocabulary types, by side.

The env side tags (:class:`EnvTags` + the ``*Tag`` leaves plus :class:`Split`);
the model side fully specifies (:class:`ModelSpec` + the bare leaves
:class:`Image`/:class:`State`/:class:`Concat`/:class:`Text`/:class:`Custom`).
"""

from .action import Action, Actuator
from .custom_encoding import CustomEncoding, RotationTransform
from .env_tags import (
    EnvTags,
    Field,
    ImageTag,
    ObservationRoles,
    ObsLeaf,
    ObsNode,
    Split,
    StateTag,
    TextTag,
)
from .model import ModelSpec
from .model_inputs import (
    Concat,
    ConcatPart,
    Constant,
    Custom,
    Image,
    InputNode,
    ModelLeaf,
    ObsTransform,
    Rotation,
    State,
    Text,
)
from .vocabularies import (
    ROTATION_DIMS,
    FitMode,
    Frame,
    ImageLayout,
    Reference,
    Resample,
    RotationEncoding,
    StackPad,
    StackSpec,
)

__all__ = [
    "ROTATION_DIMS",
    "Action",
    "Actuator",
    "Concat",
    "ConcatPart",
    "Constant",
    "Custom",
    "CustomEncoding",
    "EnvTags",
    "Field",
    "FitMode",
    "Frame",
    "Image",
    "ImageLayout",
    "ImageTag",
    "InputNode",
    "ModelLeaf",
    "ModelSpec",
    "ObsLeaf",
    "ObsNode",
    "ObsTransform",
    "ObservationRoles",
    "Reference",
    "Resample",
    "Rotation",
    "RotationEncoding",
    "RotationTransform",
    "Split",
    "StackPad",
    "StackSpec",
    "State",
    "StateTag",
    "Text",
    "TextTag",
]
