"""Write a :class:`~rlmesh.recorder.schema.ResultSet` to a bundle (folder or zip).

A bundle is ``result.json`` (the ``rlmesh.result.v1`` document) plus a ``media/``
tree of the referenced assets. A folder is the working form (inspect/iterate
locally); a zip is the shipping form (the single file a human uploads). Neither
computes a manifest or per-file hashes -- that is the closed-side mapper's job, so
the SDK stays decoupled from the platform's storage layout.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from .constants import MEDIA_DIR, RESULT_FILENAME

if TYPE_CHECKING:
    from .schema import ResultSet


def _use_zip(archive: bool | str | None, path: Path) -> bool:
    if isinstance(archive, str):
        value = archive.strip().lower()
        if value == "zip":
            return True
        if value in ("folder", "dir"):
            return False
        raise ValueError(f"archive must be 'zip' or 'folder'; got {archive!r}")
    if archive is None:
        return path.suffix.lower() == ".zip"
    return archive


def write_bundle(
    result_set: ResultSet,
    path: Path,
    *,
    assets: list[tuple[str, str]],
    recorded_at: str,
    archive: bool | str | None = None,
) -> Path:
    """Serialize ``result_set`` and its ``assets`` to ``path``; return the path.

    ``assets`` are ``(source, bundle-relative destination)`` pairs; a missing source
    is skipped (the media manifest row still points at where it should have been).
    """
    payload = json.dumps(result_set.to_dict(recorded_at=recorded_at), indent=2)

    if _use_zip(archive, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive_file:
            archive_file.writestr(RESULT_FILENAME, payload)
            for source, rel in assets:
                if os.path.isfile(source):
                    archive_file.write(source, rel, compress_type=zipfile.ZIP_STORED)
        return path

    if path.exists() and not path.is_dir():
        raise FileExistsError(
            f"{path} exists and is not a directory; pass archive='zip' to "
            "overwrite a zip bundle, or export to a different path"
        )
    path.mkdir(parents=True, exist_ok=True)
    old_media = path / MEDIA_DIR
    if old_media.is_dir():
        if not (path / RESULT_FILENAME).is_file():
            raise FileExistsError(
                f"refusing to overwrite {old_media}: {path} is not an rlmesh "
                f"bundle (no {RESULT_FILENAME}); export to a different path"
            )
        shutil.rmtree(old_media, ignore_errors=True)
    (path / RESULT_FILENAME).write_text(payload, encoding="utf-8")
    for source, rel in assets:
        if not os.path.isfile(source):
            continue
        dest = path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    return path
