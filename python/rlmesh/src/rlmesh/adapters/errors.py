"""Errors raised by the adapters package."""


class AdapterResolutionError(ValueError):
    """Raised when an env spec and a model spec cannot be reconciled."""


__all__ = ["AdapterResolutionError"]
