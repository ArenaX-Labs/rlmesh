from __future__ import annotations

from typing import cast


def redact_address(address: str) -> str:
    """``address`` with its key (``#…``) left out.

    What reprs, logs and errors show, since the key is a credential (its
    holder attaches as ``local``).
    """
    before, sep, rest = address.partition("#")
    if not sep:
        return address
    slash = rest.find("/")
    return f"{before}#…{rest[slash:] if slash >= 0 else ''}"


def env_address(address: str) -> str:
    """An env's participant address: a lone ``host:port`` is short for ``host:port/env``."""
    from .._load_native import load_native

    return cast("str", load_native("env_address")(address))


def normalize_bind_address(address: str | None = None) -> str:
    """Normalize a session bind address (default ``:0/env``).

    A lone ``host:port`` is short for ``host:port/env``.
    """
    from .._upgrade import refuse_v1_address

    address = ":0/env" if address is None else address
    if not address:
        raise ValueError("address cannot be empty")
    address = env_address(address)
    refuse_v1_address("EnvServer", address)
    return address


def normalize_connect_address(address: str | None = None) -> str:
    """Validate and normalize a session participant address."""
    from .._load_native import load_native

    return cast("str", load_native("normalize_connect_address")(address))


__all__ = ["env_address", "normalize_bind_address", "normalize_connect_address"]
