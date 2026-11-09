"""Record eval runs and export a portable bundle.

The :class:`Recorder` captures per-episode outcomes and optional video from a
run and writes an ``rlmesh.result.v1`` bundle. Video is recorded from image
observations as AV1 mp4 or copied from an env-produced file.
"""

from __future__ import annotations

from .constants import SCHEMA
from .recorder import Recorder
from .schema import EpisodeRecord, MediaRef, ResultSet, WorkloadRecord

__all__ = [
    "SCHEMA",
    "EpisodeRecord",
    "MediaRef",
    "Recorder",
    "ResultSet",
    "WorkloadRecord",
]
