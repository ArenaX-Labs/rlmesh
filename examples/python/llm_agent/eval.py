"""Run the LLM harness against a served task for a few episodes."""

import argparse

from harness import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, LLMHarness


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an LLM on a served task.")
    parser.add_argument("address", help="environment endpoint address")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="chat model name")
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="per-call budget"
    )
    parser.add_argument("-n", type=int, default=5, help="episodes (seeds 0..n-1)")
    args = parser.parse_args()

    harness = LLMHarness(model=args.model, max_tokens=args.max_tokens)
    result = harness.run(args.address, seeds=list(range(args.n)))
    for episode in result.episodes:
        print(
            f"seed={episode.seed} steps={episode.steps} reward={episode.reward:.3f} "
            f"success={episode.success} truncated={episode.truncated}"
        )
    print(f"mean_reward={result.mean_reward:.3f} transcripts={harness.transcripts}")


if __name__ == "__main__":
    main()
