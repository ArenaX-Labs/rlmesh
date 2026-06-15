"""Evaluate any registered model on any registered env via a resolved adapter.

Run as a module from ``examples/python`` (no server needed):

    uv run python -m vla_adapters.eval                       # every model x env pair
    uv run python -m vla_adapters.eval --model xvla --env simpler-bridge

Against a live env endpoint:

    uv run python -m vla_adapters.eval --model smolvla --env libero --address 127.0.0.1:5555
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from typing import Any

import rlmesh.adapters as adapt

from .envs import ENVS, EnvEntry
from .models import MODELS
from .overrides import OVERRIDES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate registered models on registered envs. With no "
        "arguments, dry-runs every model x env combination."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        help="limit to one model (default: all registered models)",
    )
    parser.add_argument(
        "--env",
        choices=sorted(ENVS),
        help="limit to one env (default: all registered envs)",
    )
    parser.add_argument(
        "--address",
        help="env endpoint address (requires --model and --env); "
        "omit to do offline dry runs instead",
    )
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    if args.address and not (args.model and args.env):
        parser.error("--address requires both --model and --env")
    return args


def summarize(value: object) -> str:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"array shape={tuple(shape)} dtype={getattr(value, 'dtype', '?')}"
    if isinstance(value, list):
        return f"list len={len(value)}"
    return repr(value)


def build_adapter(
    model_name: str, env_name: str, env: EnvEntry
) -> adapt.AdapterBase[Any]:
    """Build the pairing adapter, most specific mechanism first:

    1. a pair-level override (complete overwrite for one special pairing),
    2. the model's custom adapter factory (model-wide, all envs),
    3. plain spec resolution from the env's tags and spaces.
    """
    override = OVERRIDES.get((model_name, env_name))
    if override is not None:
        return override()
    model_entry = MODELS[model_name]
    if model_entry.make_adapter is not None:
        return model_entry.make_adapter(
            env.tags, env.observation_space, env.action_space
        )
    return adapt.resolve(
        env.tags, env.observation_space, env.action_space, model_entry.spec
    )


def dry_run(
    adapter: adapt.AdapterBase[Any],
    env_entry: EnvEntry,
    predict_fn: Callable[[Mapping[str, Any]], Any],
) -> None:
    """Push one synthetic observation through obs -> model -> action."""
    obs = env_entry.sample_obs()
    payload = adapter.transform_obs(obs)
    print("model payload:")
    for key, value in payload.items():
        print(f"  {key}: {summarize(value)}")
    action = adapter.transform_action(predict_fn(payload))
    print(f"env action: {summarize(action)}")


def run_remote(address: str, model_name: str, env_name: str, episodes: int) -> None:
    """Run episodes against a live env endpoint.

    For a plain spec pairing this uses the zero-config path: the env
    publishes its tags in the contract, so ``Model(spec=...)``
    resolves the adapter from the handshake alone. Escape-hatch pairings
    (a stateful model adapter or a pair override) build the adapter
    explicitly and wrap the predict function.
    """
    from rlmesh.numpy import Model, RemoteEnv

    env = RemoteEnv(address)
    try:
        model_entry = MODELS[model_name]
        is_plain = (
            OVERRIDES.get((model_name, env_name)) is None
            and model_entry.make_adapter is None
        )
        if is_plain:
            print(
                adapt.resolve_from_contract(
                    env.env_contract, model_entry.spec
                ).describe()
            )
            model = Model(model_entry.load_predict_fn(), spec=model_entry.spec)
            model.run(env, max_episodes=episodes)
        else:
            adapter = build_adapter(model_name, env_name, ENVS[env_name])
            print(adapter.describe())
            model = Model(
                adapter.wrap_predict(model_entry.load_predict_fn()),
                on_reset=adapter.reset,
            )
            model.run(env, max_episodes=episodes)
    finally:
        env.close()


def main() -> None:
    args = parse_args()

    if args.address:
        run_remote(args.address, args.model, args.env, args.episodes)
        return

    model_names = [args.model] if args.model else sorted(MODELS)
    env_names = [args.env] if args.env else sorted(ENVS)
    for model_name in model_names:
        for env_name in env_names:
            print(f"=== {model_name} x {env_name} ===")
            model_entry = MODELS[model_name]
            env_entry = ENVS[env_name]
            adapter = build_adapter(model_name, env_name, env_entry)
            print(adapter.describe())
            print()
            dry_run(adapter, env_entry, model_entry.load_predict_fn())
            print()


if __name__ == "__main__":
    main()
