"""Recorder schema tags and layout constants.

The recorder uses the ``rlmesh.result.v1`` document format for portable bundles.
"""

from __future__ import annotations

from .._models._view import RENDER_SOURCE

#: Schema tag stamped onto every exported bundle. Bump the minor only for additive
#: fields; a breaking change gets a new ``.v2`` tag so an older reader can reject it.
SCHEMA = "rlmesh.result.v1"

#: Bundle filename holding the result.v1 document (run metadata + episodes).
RESULT_FILENAME = "result.json"

#: Bundle-relative top directory under which per-episode media (video/frames) live.
MEDIA_DIR = "media"

#: Default env ``info`` keys the recorder checks for an env-produced video file path
#: (case 1: the env renders its own video and just hands us the path). First hit
#: wins.
DEFAULT_VIDEO_INFO_KEYS = ("video_artifact_path", "video_url")

#: Camera label used for an env-produced video that carries no camera name.
DEFAULT_CAMERA = "default"

#: Camera label for the env's ``render()`` frame -- the viewer's default source,
#: recorded alongside the declared image roles. Aliases the viewer's own label so
#: the two surfaces share one camera vocabulary by construction.
RENDER_CAMERA = RENDER_SOURCE

#: Default playback rate for the recorded mp4, in frames per second.
DEFAULT_FPS = 30

#: Default video codec: AV1, encoded in process with nothing installed. ``"h264"``
#: runs the user's own ffmpeg.
DEFAULT_CODEC = "av1"

#: Default AV1 record quality, 1..=100 (higher is better/larger). 60 is a size/quality
#: balance for eval footage; raise toward 100 for near-lossless inspection.
DEFAULT_QUALITY = 60
