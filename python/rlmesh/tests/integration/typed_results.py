"""Checks a live result against its ``TypedDict``: every key, and each value's type, all the way down."""

from __future__ import annotations

import types
from typing import (
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)


def check(value: Any, hint: Any, where: str = "result") -> None:
    """Fail when ``value`` has a key ``hint`` lacks, lacks one it has, or holds another type."""
    origin = get_origin(hint)
    if is_typeddict(hint):
        assert isinstance(value, dict), f"{where}: {value!r} is not a dict"
        fields = get_type_hints(hint)
        extra, missing = set(value) - set(fields), set(fields) - set(value)
        assert not extra and not missing, (
            f"{where} as {hint.__name__}: keys not typed {sorted(extra)}, "
            f"typed keys missing {sorted(missing)}"
        )
        for key, field in fields.items():
            check(value[key], field, f"{where}[{key!r}]")
    elif origin in (Union, types.UnionType):
        options = get_args(hint)
        if value is None:
            assert type(None) in options, f"{where}: None is not {hint}"
            return
        typed = [o for o in options if is_typeddict(o)]
        if len(typed) > 1:  # an event: told apart by its kind
            [match] = [
                o
                for o in typed
                if value.get("kind") in get_args(get_type_hints(o)["kind"])
            ]
            check(value, match, where)
            return
        [option] = [o for o in options if o is not type(None)]
        check(value, option, where)
    elif origin is Literal:
        assert value in get_args(hint), f"{where}: {value!r} is not {hint}"
    elif origin is dict:
        assert isinstance(value, dict), f"{where}: {value!r} is not a dict"
        key_type, value_type = get_args(hint)
        for key, item in value.items():
            check(key, key_type, f"{where} key")
            check(item, value_type, f"{where}[{key!r}]")
    elif origin is list:
        assert isinstance(value, list), f"{where}: {value!r} is not a list"
        [item_type] = get_args(hint)
        for i, item in enumerate(value):
            check(item, item_type, f"{where}[{i}]")
    elif origin is tuple:
        item_types = get_args(hint)
        assert isinstance(value, tuple) and len(value) == len(item_types), (
            f"{where}: {value!r} is not {hint}"
        )
        for i, (item, item_type) in enumerate(zip(value, item_types, strict=True)):
            check(item, item_type, f"{where}[{i}]")
    elif hint is float:
        assert isinstance(value, int | float) and not isinstance(value, bool), (
            f"{where}: {value!r} is not a float"
        )
    elif hint is int:
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"{where}: {value!r} is not an int"
        )
    else:
        assert isinstance(value, hint), f"{where}: {value!r} is not {hint.__name__}"
