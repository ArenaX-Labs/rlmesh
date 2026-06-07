from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

VOLATILE_INFO_KEYS = {"episode_ids"}


def fingerprint(value: object) -> object:
    numpy_result = fingerprint_numpy(value)
    if numpy_result is not None:
        return numpy_result

    torch_result = fingerprint_torch(value)
    if torch_result is not None:
        return torch_result

    if isinstance(value, Mapping):
        return {
            str(key): fingerprint(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [fingerprint(item) for item in value]}
    if isinstance(value, list):
        return [fingerprint(item) for item in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return repr(value)


def fingerprint_numpy(value: object) -> dict[str, object] | None:
    try:
        import numpy as np
    except ImportError:
        return None
    if not isinstance(value, np.ndarray):
        if isinstance(value, np.generic):
            return {
                "type": "numpy-scalar",
                "dtype": str(value.dtype),
                "value": value.item(),
            }
        return None
    contiguous = np.ascontiguousarray(value)
    return {
        "type": "ndarray",
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def fingerprint_torch(value: object) -> dict[str, object] | None:
    try:
        import torch
    except ImportError:
        return None
    if not isinstance(value, torch.Tensor):
        return None
    array = value.detach().cpu().contiguous().numpy()
    return {
        "type": "torch-tensor",
        "shape": list(array.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def canonical_info(info: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key): fingerprint(value)
        for key, value in sorted(info.items())
        if str(key) not in VOLATILE_INFO_KEYS
    }


def normalize_trace(trace: Mapping[str, object]) -> dict[str, object]:
    return {str(key): fingerprint(value) for key, value in sorted(trace.items())}


def sequence_preview(values: Sequence[object], limit: int = 5) -> list[object]:
    return [fingerprint(value) for value in values[:limit]]
