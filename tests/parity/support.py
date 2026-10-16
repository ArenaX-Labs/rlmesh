"""Shared fixture loader and lossless projections for the native runners."""

from __future__ import annotations

import json
import pathlib
import sys

import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
LANGUAGES = {"python", "rust", "c", "cpp"}


def scenarios():
    """Load scenarios, rejecting duplicate IDs and unrecognized languages."""
    found = []
    ids = set()
    for path in sorted((ROOT / "tests/parity/fixtures").glob("*.toml")):
        for case in tomllib.loads(path.read_text())["scenario"]:
            assert case["id"] not in ids, case["id"]
            ids.add(case["id"])
            assert case["family"] and case["rules"] and case["expect"], case
            assert set(case.get("languages", LANGUAGES)) <= LANGUAGES, case
            if set(case.get("languages", LANGUAGES)) != LANGUAGES:
                assert case.get("why", "").strip(), case
            found.append(case)
    return found


def native_args(case):
    """Project inputs only; expected outputs never enter a native executable."""
    return [
        case["operation"],
        case.get("variant", ""),
        str(case.get("episodes", -1)),
        ",".join(map(str, case["seeds"])) if "seeds" in case else "none",
        str(int(case.get("close_env", False))),
    ]


def compare(case, observed):
    """Compare every specified expectation, including exact error text."""
    for key, value in case["expect"].items():
        assert key in observed, (case["id"], key, observed)
        assert observed[key] == value, (case["id"], key, value, observed)
    if "error" not in case["expect"]:
        assert "error" not in observed, (case["id"], observed)


if __name__ == "__main__":
    json.dump(scenarios(), sys.stdout)
