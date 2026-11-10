"""Shared paths for Unreal Reach Target identity / .env files."""

from __future__ import annotations

import os
from pathlib import Path

UNREAL_DIR = Path(__file__).resolve().parent
DOTENV_PATH = UNREAL_DIR / ".env"
DOTENV_EXAMPLE_PATH = UNREAL_DIR / ".env.example"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "7447"
DEFAULT_PARTICIPANT = "env"
HOST_IDENTITY_VAR = "RLMESH_HOST_IDENTITY"
UNREAL_ADDR_VAR = "RLMESH_UNREAL_ADDR"
EPISODES_VAR = "RLMESH_EPISODES"


def rlmesh_data_dir() -> Path:
    """Platform data dir for the static Unreal host-identity file."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "rlmesh"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "rlmesh"
    return Path.home() / ".local" / "share" / "rlmesh"


def host_identity_path() -> Path:
    return rlmesh_data_dir() / "unreal_host_identity"


def make_address(
    address_key: str,
    *,
    host: str = DEFAULT_HOST,
    port: str = DEFAULT_PORT,
    participant: str = DEFAULT_PARTICIPANT,
) -> str:
    key = address_key.strip().lstrip("#")
    return f"{host}:{port}#{key}/{participant}"
