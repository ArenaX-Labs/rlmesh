"""Record eval runs and export a portable bundle the managed platform can ingest.

The :class:`Recorder` captures per-episode outcomes (and optional env media) from
any OSS run and writes an ``rlmesh.result.v1`` bundle -- the SDK's own vocabulary, a
decoupled superset of what the platform dashboard displays. The closed-side upload
path maps result.v1 -> the platform's private ``episode.v1`` and stitches frame
stacks into mp4.
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
