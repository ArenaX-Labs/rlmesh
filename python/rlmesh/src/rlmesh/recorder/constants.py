"""Recorder schema tags and layout constants.

The recorder speaks its own OSS-native vocabulary (``rlmesh.result.v1``), not the
managed platform's private ``rlmesh.episode.v1`` wire format. The one small mapper
that translates result.v1 -> episode.v1 (and stitches frame stacks into mp4) lives
on the managed side, so the open-source SDK never imports the closed schema.
"""

from __future__ import annotations

#: Schema tag stamped onto every exported bundle. Bump the minor only for additive
#: fields; a breaking change gets a new ``.v2`` tag so an older mapper can reject it.
SCHEMA = "rlmesh.result.v1"

#: Bundle filename holding the result.v1 document (run metadata + episodes).
RESULT_FILENAME = "result.json"

#: Bundle-relative top directory under which per-episode media (video/frames) live.
MEDIA_DIR = "media"

#: Default env ``info`` keys the recorder checks for an env-produced video file path
#: (case 1: the env renders its own video and just hands us the path). First hit
#: wins. Mirrors the managed runner, which reads ``video_artifact_path`` / ``video_url``.
DEFAULT_VIDEO_INFO_KEYS = ("video_artifact_path", "video_url")

#: Camera label used for an env-produced video that carries no camera name.
DEFAULT_CAMERA = "default"

#: Camera label for the env's ``render()`` frame -- the viewer's default source,
#: recorded alongside the declared image roles.
RENDER_CAMERA = "render"

#: Default playback rate for the recorded mp4. Matches the managed runner's default
#: so uploaded and locally-recorded videos play at the same speed.
DEFAULT_FPS = 30

#: Default AV1 record quality, 1..=100 (higher is better/larger). 60 is a size/quality
#: balance for eval footage; raise toward 100 for near-lossless inspection.
DEFAULT_QUALITY = 60
