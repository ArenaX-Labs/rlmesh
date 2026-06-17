"""Helpers for normalizing public Python endpoint arguments."""

from __future__ import annotations

import sys
from typing import Literal

Transport = Literal["tcp", "unix"]


def _supports_unix_transport() -> bool:
    return not sys.platform.startswith("win")


def normalize_bind_address(
    address: str | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    path: str | None = None,
    transport: Transport | None = None,
) -> str | None:
    """Normalize bind-style endpoint arguments to a single address string.

    Args:
        address: Explicit endpoint address. Cannot be combined with helper
            arguments.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.

    Returns:
        Normalized endpoint address, or ``None`` to use the server default.
    """
    _validate_address_inputs(
        address, host=host, port=port, path=path, transport=transport
    )
    if address is not None:
        return address
    if path is not None or transport == "unix":
        if path is None:
            raise ValueError("path is required when transport='unix'")
        if not _supports_unix_transport():
            raise ValueError(
                "unix sockets are not supported on Windows; use tcp://host:port instead"
            )
        return f"unix://{path}"
    if host is None and port is None and transport is None:
        return None
    resolved_host = host or "127.0.0.1"
    resolved_port = 0 if port is None else _normalize_port(port)
    return f"tcp://{resolved_host}:{resolved_port}"


def normalize_connect_address(
    address: str | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    path: str | None = None,
    transport: Transport | None = None,
) -> str:
    """Normalize connect-style endpoint arguments to a single address string.

    Args:
        address: Explicit endpoint address. Cannot be combined with helper
            arguments.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.

    Returns:
        Normalized endpoint address.
    """
    _validate_address_inputs(
        address, host=host, port=port, path=path, transport=transport
    )
    if address is not None:
        return address
    if path is not None or transport == "unix":
        if path is None:
            raise ValueError("path is required when transport='unix'")
        if not _supports_unix_transport():
            raise ValueError(
                "unix sockets are not supported on Windows; use tcp://host:port instead"
            )
        return f"unix://{path}"
    if host is None and port is None:
        raise ValueError("address or host/port is required")
    resolved_host = host or "127.0.0.1"
    if port is None:
        raise ValueError("port is required when connecting with host/path helpers")
    return f"tcp://{resolved_host}:{_normalize_port(port)}"


def _validate_address_inputs(
    address: str | None,
    *,
    host: str | None,
    port: object,
    path: str | None,
    transport: str | None,
) -> None:
    if transport not in (None, "tcp", "unix"):
        raise ValueError("transport must be 'tcp', 'unix', or None")
    if address is not None:
        if not address:
            raise ValueError("address cannot be empty")
        if any(value is not None for value in (host, port, path, transport)):
            raise ValueError(
                "address cannot be combined with host, port, path, or transport"
            )
        return
    if path is not None and any(value is not None for value in (host, port)):
        raise ValueError("path cannot be combined with host or port")
    if transport == "unix" and any(value is not None for value in (host, port)):
        raise ValueError("unix transport cannot be combined with host or port")


def _normalize_port(port: object) -> int:
    if not isinstance(port, int):
        raise TypeError("port must be an int")
    if port < 0 or port > 65535:
        raise ValueError("port must be in the range 0..65535")
    return port


__all__ = ["Transport", "normalize_bind_address", "normalize_connect_address"]
