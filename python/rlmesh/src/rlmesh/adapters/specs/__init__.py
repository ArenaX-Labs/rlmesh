"""Declarative spec dataclasses and their vocabulary types, by side.

The env side annotates (:class:`EnvAnnotations` + the ``*Annotation`` types);
the model side fully specifies (:class:`ModelSpec` + the ``*Input`` types).
"""

from .action import ActionComponent, ActionLayout
from .env_annotations import (
    EnvAnnotations,
    ImageAnnotation,
    ObsAnnotation,
    StateAnnotation,
    TextAnnotation,
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
    "EnvAnnotations",
    "ImageAnnotation",
    "ImageInput",
    "ImageLayout",
    "ModelInput",
    "ModelSpec",
    "ObsAnnotation",
    "ObsTransform",
    "RotationEncoding",
    "StateAnnotation",
    "StateComponent",
    "StateInput",
    "TextAnnotation",
    "TextInput",
]
