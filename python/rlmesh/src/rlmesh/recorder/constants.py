"""Recorder schema tags and layout constants.

The recorder uses the ``rlmesh.result.v1`` document format for portable bundles.
"""

from __future__ import annotations

from .._models._view import RENDER_SOURCE

#: Schema tag on every exported bundle; additive fields bump the minor, a breaking change gets a new tag.
SCHEMA = "rlmesh.result.v1"

#: Bundle filename holding the result document.
RESULT_FILENAME = "result.json"

#: Bundle-relative directory holding per-episode media.
MEDIA_DIR = "media"

#: Default env ``info`` keys checked for an env-produced video path; the first hit wins.
DEFAULT_VIDEO_INFO_KEYS = ("video_artifact_path", "video_url")

#: Camera label for an env-produced video with no camera name.
DEFAULT_CAMERA = "default"

#: Camera label for the env's ``render()`` frame; matches the viewer's label.
RENDER_CAMERA = RENDER_SOURCE

#: Default playback rate of recorded mp4s, in frames per second.
DEFAULT_FPS = 30

#: Default video codec: AV1 in process; ``"h264"`` runs the user's ffmpeg.
DEFAULT_CODEC = "av1"

#: Default AV1 quality, 1..=100 (higher is better and larger).
DEFAULT_QUALITY = 60
