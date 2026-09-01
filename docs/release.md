# Release Process

RLMesh is versioned, gated, and tagged from a maintainer's machine; pushing the tag hands off to CI (`.github/workflows/release.yml`), which builds the wheel matrix on Linux and macOS runners and publishes everything. SemVer is the source of truth (`Cargo.toml [workspace.package].version`); the PEP 440 spelling for PyPI is derived from it. The version contract is in {doc}`versioning`.

The mechanical steps are scripted. You still own the changelog prose and the version number. The one irreversible action, pushing the tag, never runs without you asking.

## One-command flow

```bash
# Curate the changelog first (see below), then:
python scripts/release.py X.Y.Z --dry-run   # bump + full gate; no commit, tag, or publish
python scripts/release.py X.Y.Z             # also commit and tag vX.Y.Z (does NOT push)
git push origin HEAD --tags                 # you push; CI does the rest
```

`release.py` refuses to proceed if the working tree is dirty, the `vX.Y.Z` tag already exists, the changelog still has `<!-- DRAFT -->` markers, or there is no `## [X.Y.Z]` changelog section.

The tag push triggers the release workflow: Linux/Windows wheels build on a Linux runner and macOS wheels on a macOS runner (each system-tested against the built artifacts), then a publish job validates the assembled 14-wheel matrix and ships crates.io, PyPI, and the GitHub Release. The publish jobs are idempotent — re-running the workflow after a partial failure skips already-published crates and an existing Release, and PyPI skips duplicate files.

## Prerequisites

- Nothing local: no publish tokens. crates.io and PyPI both authenticate via OIDC trusted publishing scoped to `release.yml` and the `release` GitHub environment; the GitHub Release uses the workflow's own token.
- One-time registry setup: a trusted publisher configured on the PyPI `rlmesh` project and on each of the nine `rlmesh*` crates (repository `ArenaX-Labs/rlmesh`, workflow `release.yml`, environment `release`), plus the `release` environment created in the repository settings (add required reviewers there if you want a manual approval gate before publish).
- Break-glass local publish, from the tagged commit: no single host builds the full wheel matrix anymore, so first assemble `python/rlmesh/dist` from the release run's artifacts (`gh run download <run-id> -p 'wheels-*'`, then move the `.whl` files in) or build each half on its own host without re-running `release:python:prepare` in between (it wipes dist). Then `python scripts/release.py X.Y.Z --publish-crates` (needs `CARGO_REGISTRY_TOKEN`) and `--github-release` (needs `gh` auth); wheels would need a manual `twine upload`.

## Curate the changelog

The changelog is hand-written. `git-cliff` is gone.

1. `mise run changelog:draft` appends draft bullets under `## [Unreleased]` in `CHANGELOG.md`, one per user-facing commit since the last `v*` tag, each marked `<!-- DRAFT -->`.
2. Rewrite each bullet in your own words, drop internal-only changes, and group them under the Keep a Changelog sections. **Delete every `<!-- DRAFT -->` marker** — the release driver refuses to ship while any remain.
3. Rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a fresh empty `## [Unreleased]` above it, and update the compare links at the bottom.

A breaking change to a stable symbol gets a `### Breaking` entry with a before/after migration note (see {doc}`versioning`).

## Bump the version

`release.py` runs this for you, or run it alone:

```bash
mise run bump X.Y.Z
```

It rewrites every manifest and install snippet, updates the workflow cohort, runs `cargo update` and `uv lock`, then `policy:check` — the backstop that fails loudly if any version-bearing spot was missed. Prereleases use an exact provisional cohort (`YYYY.MM-X.Y.Z-beta.N`); a stable release seals the bare edition in `rlmesh.toml` with `sealed_in` and `spec_sha256`. See {doc}`editions/index`.

## Tag scheme

One unscoped annotated tag per release: `vX.Y.Z`. The legacy `rust/v*` and `python/v*` tags are history — do not add more.

## Wheels

RLMesh publishes wheels only; do not build or upload an sdist. Wheel builds are host-specific, and CI runs both halves in parallel:

- macOS (`mise run release:python:wheels:macos`) builds the macOS arm64 and x86_64 wheels.
- Linux (`mise run release:python:wheels:linux`) builds the Linux and Windows subset.

`python scripts/check_python_wheels.py python/rlmesh/dist` validates ABI/platform tags and payload contents. Release validation rejects plain `linux_*` tags; uploadable Linux wheels use `manylinux` or `musllinux`. The publish job validates license payloads (`release:artifacts:licenses`) over the merged wheel matrix and freshly packaged crates before anything uploads.

## Publish order

The publish job runs `release.py --publish-crates`, which publishes the crates in dependency order (`rlmesh-proto`, `rlmesh-spaces`, `rlmesh-viewer`, `rlmesh-adapters`, `rlmesh-cli`, `rlmesh-runtime`, `rlmesh-grpc`, `rlmesh-sandbox`, `rlmesh`), then uploads the wheels to PyPI, then cuts the GitHub Release via `release.py --github-release`. `cargo publish` waits for each crate to appear in the index before the next one builds, so the ordered run is safe to leave unattended.

## GitHub Releases

Every release gets a GitHub Release built from its tag. Pre-releases (`-beta.N`, `-rc.N`) are marked `--prerelease` so they stay out of "Latest"; a final `X.Y.Z` release becomes Latest. `release.py` sets this automatically from the version string.

## Post-publish smoke

```bash
python -m venv /tmp/rlmesh-smoke
/tmp/rlmesh-smoke/bin/python -m pip install rlmesh
/tmp/rlmesh-smoke/bin/python -c "import rlmesh; print(rlmesh.__version__)"
```

## Recovery

- A bad crates.io or PyPI publish cannot be deleted, only yanked. Yank it and ship a fixed patch release.
- If the seal gate blocks the release, the edition metadata does not match the version: rerun `mise run bump X.Y.Z` and review `rlmesh.toml`, or keep the release a pre-release.
