#!/usr/bin/env python3
"""Render the Homebrew cask for a release's macOS CLI archive, or publish it to a tap.

usage: homebrew_cask.py VERSION ARCHIVE                 print the cask
       homebrew_cask.py VERSION ARCHIVE --publish TAP   write Casks/rlmesh.rb in TAP (owner/repo) with gh
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

CASK = """\
cask "rlmesh" do
  version "{version}"
  sha256 "{sha256}"

  url "https://github.com/ArenaX-Labs/rlmesh/releases/download/v#{{version}}/rlmesh-cli-#{{version}}-universal-apple-darwin.tar.gz"
  name "RLMesh"
  desc "Connect models to environments for evaluation"
  homepage "https://rlmesh.dev/"

  livecheck do
    url :url
    strategy :github_latest
  end

  depends_on macos: ">= :big_sur"

  binary "rlmesh-cli-#{{version}}-universal-apple-darwin/rlmesh"
  binary "rlmesh-cli-#{{version}}-universal-apple-darwin/rlmesh", target: "docker-credential-rlmesh"
  binary "rlmesh-cli-#{{version}}-universal-apple-darwin/rlmesh", target: "rlmeshd"

  zap trash: "~/Library/Application Support/rlmesh"
end
"""

CASK_PATH = "Casks/rlmesh.rb"


def archive_name(version: str) -> str:
    """The release asset the cask installs."""
    return f"rlmesh-cli-{version}-universal-apple-darwin.tar.gz"


def render(version: str, archive: Path) -> str:
    """The cask for ``archive``, which must be this version's release asset."""
    if archive.name != archive_name(version):
        sys.exit(f"expected {archive_name(version)}, got {archive.name}")
    sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    return CASK.format(version=version, sha256=sha256)


def publish(version: str, cask: str, tap: str) -> None:
    """Create or replace the tap's cask through the GitHub contents API."""
    if "-" in version:
        print(f"::notice::{version} is a pre-release; the Homebrew tap keeps the last release.")
        return
    existing = subprocess.run(
        ["gh", "api", f"repos/{tap}/contents/{CASK_PATH}"],
        capture_output=True,
        text=True,
        check=False,
    )
    body = {
        "message": f"chore: rlmesh {version}",
        "content": base64.b64encode(cask.encode()).decode(),
    }
    if existing.returncode == 0:
        body["sha"] = json.loads(existing.stdout)["sha"]
    subprocess.run(
        ["gh", "api", "-X", "PUT", f"repos/{tap}/contents/{CASK_PATH}", "--input", "-"],
        input=json.dumps(body),
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(f"updated {tap} {CASK_PATH} to {version}")


def main(argv: list[str]) -> None:
    """Run from the command line."""
    if len(argv) == 2:
        sys.stdout.write(render(argv[0], Path(argv[1])))
    elif len(argv) == 4 and argv[2] == "--publish":
        publish(argv[0], render(argv[0], Path(argv[1])), argv[3])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
