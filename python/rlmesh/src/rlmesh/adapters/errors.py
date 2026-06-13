"""Errors raised by the adapters package."""


class AdapterResolutionError(ValueError):
    """Raised when env tags and a model spec cannot be reconciled."""


__all__ = ["AdapterResolutionError"]
