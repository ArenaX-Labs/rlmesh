#!/usr/bin/env python3
"""Run the Unreal Reach Target demo without pasting Serving #keys each Play.

One-time setup (stable host identity)::

  python integrations/unreal/reach_target_app.py --init-identity

Rebuild the RLMesh plugin once so it loads the identity file, press Play, then::

  python integrations/unreal/reach_target_app.py

The plugin loads ``%LOCALAPPDATA%/rlmesh/unreal_host_identity`` into
``RLMESH_HOST_IDENTITY`` at module startup (and again at BeginPlay) so Serving
``#keys`` stay fixed. This app reads ``RLMESH_UNREAL_ADDR`` from
``integrations/unreal/.env``.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

_UNREAL_DIR = Path(__file__).resolve().parent
if str(_UNREAL_DIR) not in sys.path:
    sys.path.insert(0, str(_UNREAL_DIR))

from reach_target_dotenv import apply_dotenv, upsert_dotenv
from reach_target_paths import (
    DOTENV_EXAMPLE_PATH,
    DOTENV_PATH,
    EPISODES_VAR,
    UNREAL_ADDR_VAR,
    host_identity_path,
    make_address,
)
from reach_target_policy import gap_xyz, toward_target


def _write_secret_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents.strip() + "\n", encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def init_identity(*, force: bool) -> int:
    identity_path = host_identity_path()
    if identity_path.is_file() and not force:
        print(
            f"ERROR: identity already exists at {identity_path}\n"
            "  Re-run with --force to overwrite (also rewrites .env address).",
            file=sys.stderr,
        )
        return 2

    try:
        from rlmesh._rlmesh import sandbox_host_identity
    except ImportError:
        print(
            "ERROR: rlmesh is not installed in this Python.\n"
            "From the rlmesh checkout: mise run setup:python\n"
            f"Current interpreter: {sys.executable}",
            file=sys.stderr,
        )
        return 1

    identity_text, address_key = sandbox_host_identity()
    address = make_address(address_key)

    _write_secret_file(identity_path, identity_text)
    upsert_dotenv(
        DOTENV_PATH,
        {
            UNREAL_ADDR_VAR: address,
            EPISODES_VAR: "1",
        },
    )
    if not DOTENV_EXAMPLE_PATH.is_file():
        DOTENV_EXAMPLE_PATH.write_text(
            "# Copy to .env or run: python integrations/unreal/reach_target_app.py --init-identity\n"
            f"{UNREAL_ADDR_VAR}=127.0.0.1:7447#<address_key>/env\n"
            f"{EPISODES_VAR}=1\n",
            encoding="utf-8",
        )

    print("Initialized static Unreal host identity.")
    print(f"  identity file: {identity_path}")
    print(f"  dotenv:        {DOTENV_PATH}")
    print(f"  address:       {address}")
    print("Rebuild the RLMesh plugin, press Play, then re-run this app.")
    return 0


def resolve_address(cli_address: str) -> str:
    if cli_address.strip():
        return cli_address.strip()
    return os.environ.get(UNREAL_ADDR_VAR, "").strip()


def run_reach_target(
    *,
    address: str,
    episodes: int,
    seeds: str,
    verbose: bool,
) -> int:
    try:
        import rlmesh
        import rlmesh.numpy
    except ImportError:
        print(
            "ERROR: rlmesh is not installed in this Python.\n"
            "From the rlmesh checkout: mise run setup:python\n"
            f"Current interpreter: {sys.executable}",
            file=sys.stderr,
        )
        return 1

    if not address:
        print(
            "ERROR: no Serving address.\n"
            "  1) python integrations/unreal/reach_target_app.py --init-identity\n"
            "  2) Rebuild the RLMesh plugin and press Play\n"
            "  3) Re-run this app\n"
            "Or pass --address / set RLMESH_UNREAL_ADDR.",
            file=sys.stderr,
        )
        return 2

    if "#" not in address:
        print(
            "WARNING: address has no #keys; loopback hosts usually require the full invite.",
            file=sys.stderr,
        )

    run_kwargs: dict = {"episodes": episodes}
    if seeds.strip():
        parsed = [int(part.strip()) for part in seeds.split(",") if part.strip()]
        if not parsed:
            print("ERROR: --seeds was set but no integers were parsed.", file=sys.stderr)
            return 1
        run_kwargs = {"seeds": parsed}

    if verbose:

        def policy(obs: object):
            action = toward_target(obs)
            gap = gap_xyz(obs)
            print(
                f"  gap=({gap[0]:+.1f}, {gap[1]:+.1f}, {gap[2]:+.1f}) "
                f"-> dir=({action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f})"
            )
            return action

        predict = policy
    else:
        predict = toward_target

    model = rlmesh.numpy.Model(predict)
    print(f"Connecting to {address} …")
    print(f"Model: toward_target (scripted, XYZ)  options: {run_kwargs}")
    try:
        result = rlmesh.run(model, address, **run_kwargs)
    except Exception as exc:  # noqa: BLE001 - CLI surface
        print(f"ERROR: run failed: {exc}", file=sys.stderr)
        print(
            "Checks:\n"
            "  - Unreal is in Play and LogRLMesh shows Serving\n"
            "  - LogRLMesh shows Loaded RLMESH_HOST_IDENTITY (rebuild plugin if missing)\n"
            "  - RLMESH_UNREAL_ADDR in integrations/unreal/.env matches the Serving #keys\n"
            "  - plugin rlmesh_capi matches this Python's workflow edition "
            "(RLMESH_RELEASE_BUILD=1 on Windows)",
            file=sys.stderr,
        )
        return 1

    print(result)
    print("reach_target_app.py finished successfully.")
    return 0


def main(argv: list[str] | None = None) -> int:
    apply_dotenv(DOTENV_PATH)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--init-identity",
        action="store_true",
        help="Generate a static HostIdentity and write .env (once per machine)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --init-identity, overwrite an existing identity file",
    )
    parser.add_argument(
        "--address",
        default="",
        help="Full Serving address override (else RLMESH_UNREAL_ADDR from .env)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.environ.get(EPISODES_VAR, "1") or "1"),
        help="Episodes when --seeds is not set (default from .env or 1)",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated episode seeds (overrides --episodes)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print gap and direction each step",
    )
    args = parser.parse_args(argv)

    if args.init_identity:
        return init_identity(force=args.force)

    return run_reach_target(
        address=resolve_address(args.address),
        episodes=args.episodes,
        seeds=args.seeds,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
