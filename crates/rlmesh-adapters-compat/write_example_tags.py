"""Record the env tags the repo's example envs publish, for the 0.1 reader check.

``python crates/rlmesh-adapters-compat/write_example_tags.py`` rewrites
``tests/fixtures/example_env_tags.json`` (then ``mise run fmt:config`` formats
it); ``--check`` exits 1 when what it holds is stale (the Python suite runs
that).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    Path(__file__).resolve().parent / "tests" / "fixtures" / "example_env_tags.json"
)
#: Example modules and the module-level EnvTags each publishes.
EXAMPLES = (
    ("examples/python/adapters/serve_and_run.py", "ENV_TAGS"),
    ("examples/python/vla_adapters/envs/libero.py", "TAGS"),
    ("examples/python/vla_adapters/envs/metaworld.py", "TAGS"),
    ("examples/python/vla_adapters/envs/simpler_bridge.py", "TAGS"),
)


def recorded() -> dict[str, object]:
    """The example envs' tags as they publish them now, by ``path:NAME``."""
    tags: dict[str, object] = {}
    for path, name in EXAMPLES:
        spec = importlib.util.spec_from_file_location(
            f"_example_{len(tags)}", ROOT / path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tags[f"{path}:{name}"] = json.loads(getattr(module, name).to_json())
    return tags


def main(argv: list[str]) -> int:
    """Rewrite the fixture, or with ``--check`` say whether it is stale."""
    tags = recorded()
    if "--check" in argv:
        if json.loads(FIXTURE.read_text(encoding="utf-8")) != tags:
            print(f"{FIXTURE} is stale: run python {Path(__file__).relative_to(ROOT)}")
            return 1
        return 0
    FIXTURE.write_text(
        json.dumps(tags, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
