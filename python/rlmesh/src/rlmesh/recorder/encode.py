"""Optional mp4 encoding for captured frame stacks.

Off by default: the recorder ships raw ``.npz`` frame stacks and stays
dependency-free. Opt in with ``Recorder(video=True)`` to encode each captured stack
to a browser-playable H.264 mp4 instead. Encoding shells out to the ``ffmpeg``
*binary* (so it works across a wide range of ffmpeg versions), piping raw RGB24
frames to stdin with the same flags the managed runner uses
(``libx264`` / ``yuv420p`` / ``+faststart``). The binary is sourced from the
``imageio-ffmpeg`` wheel when present (``pip install 'rlmesh[recorder]'``), else a
system ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def _imageio_ffmpeg_exe() -> str | None:
    """The ffmpeg binary bundled by ``imageio-ffmpeg``, or ``None`` if unavailable.

    ``imageio_ffmpeg`` is untyped and optional, so reach its entry point through
    ``getattr`` (keeping types out of ``Unknown``) inside a broad ``except``.
    """
    try:
        import imageio_ffmpeg  # pyright: ignore[reportMissingTypeStubs]
    except Exception:
        return None
    getter = getattr(imageio_ffmpeg, "get_ffmpeg_exe", None)
    if not callable(getter):
        return None
    try:
        exe = getter()
    except Exception:
        return None
    return str(exe) if exe else None


def resolve_ffmpeg() -> str:
    """Path to an ffmpeg binary, or raise a directive error explaining how to get one.

    Prefers the ``imageio-ffmpeg`` bundled binary (the ``rlmesh[recorder]`` extra),
    then a system ``ffmpeg`` on PATH.
    """
    exe = _imageio_ffmpeg_exe() or shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError(
        "mp4 encoding requires ffmpeg. Install the optional extra "
        "`pip install 'rlmesh[recorder]'` (bundles ffmpeg via imageio-ffmpeg), or put "
        "an ffmpeg binary on PATH. Or keep the default frame-stack export "
        "(Recorder(video=False))."
    )


def ffmpeg_available() -> bool:
    """Whether an ffmpeg binary can be resolved (bundled or on PATH)."""
    try:
        resolve_ffmpeg()
        return True
    except RuntimeError:
        return False


def _to_rgb24(frame: Any, height: int, width: int) -> np.ndarray:
    """One HWC uint8 frame coerced to contiguous 3-channel RGB of the given size."""
    import numpy as np

    array = np.ascontiguousarray(frame)
    if array.ndim == 2:
        array = array[:, :, None]
    channels = array.shape[2]
    if channels == 1:
        array = np.repeat(array, 3, axis=2)
    elif channels == 4:
        array = array[:, :, :3]
    elif channels != 3:
        raise ValueError(f"frame has {channels} channels; expected 1, 3, or 4")
    if array.shape[0] != height or array.shape[1] != width:
        raise ValueError(
            f"frame size {tuple(array.shape[:2])} does not match the episode's "
            f"first frame {(height, width)}; all frames of a camera must be one size"
        )
    return np.ascontiguousarray(array, dtype=np.uint8)


def encode_frames_to_mp4(
    path: str, frames: list[np.ndarray], fps: int
) -> dict[str, Any]:
    """Encode a list of HWC uint8 frames to an H.264 mp4 at ``path`` via ffmpeg.

    Streams frames to ffmpeg's stdin (bounded memory), pads to even dimensions
    (``yuv420p`` requires it), and writes faststart so a browser ``<video>`` plays it.
    Returns ``{frame_count, height, width, fps}`` for the media manifest row.
    """
    import numpy as np

    if not frames:
        raise ValueError("no frames to encode")
    first = np.ascontiguousarray(frames[0])
    height, width = int(first.shape[0]), int(first.shape[1])

    command = [
        resolve_ffmpeg(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        path,
    ]
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(_to_rgb24(frame, height, width).tobytes())
    except BrokenPipeError:
        pass  # ffmpeg exited early; the returncode check below surfaces why
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
    # ffmpeg runs with `-loglevel error`, so stderr stays small; read it, then wait.
    stderr = proc.stderr.read() if proc.stderr is not None else b""
    proc.wait()
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", "replace")[:500]
        raise RuntimeError(f"ffmpeg encode failed (exit {proc.returncode}): {detail}")
    return {
        "frame_count": len(frames),
        "height": height,
        "width": width,
        "fps": float(fps),
    }
