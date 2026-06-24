# Release Process

RLMesh is published manually from a maintainer's machine. SemVer is the source of truth (`Cargo.toml [workspace.package].version`); the PEP 440 spelling for PyPI is derived from it. The version contract is in {doc}`versioning`.

The mechanical steps are scripted. You still own the changelog prose and the version number. The two irreversible actions, pushing the tag and publishing to the registries, never run without you asking.

## One-command flow

```bash
# Curate the changelog first (see below), then:
python scripts/release.py X.Y.Z --dry-run   # bump + full gate; no commit, tag, or publish
python scripts/release.py X.Y.Z             # also commit and tag vX.Y.Z (does NOT push)
git push origin HEAD --tags                 # you push, after reviewing
python scripts/release.py X.Y.Z --publish   # crates.io + PyPI + GitHub Release
```

`release.py` refuses to proceed if the working tree is dirty, the `vX.Y.Z` tag already exists, the changelog still has `<!-- DRAFT -->` markers, or there is no `## [X.Y.Z]` changelog section.

## Prerequisites

- crates.io publish access to the eight `rlmesh*` crates and PyPI access to `rlmesh`.
- `gh` authenticated for the GitHub Release.
- Publish tokens available (for example via `fnox`): `CARGO_REGISTRY_TOKEN`, `PYPI_TOKEN`.
- A build host that can produce uploadable wheels (see Wheels).

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

It rewrites every manifest and install snippet, updates the workflow cohort, runs `cargo update` and `uv lock`, then `policy:check` — the backstop that fails loudly if any version-bearing spot was missed. Prereleases use an exact provisional cohort such as `2026.06-0.1.0-rc.1`; a stable release seals the bare edition in `rlmesh.toml` with `sealed_in` and `spec_sha256`. See {doc}`editions/index`.

## Tag scheme

One unscoped annotated tag per release: `vX.Y.Z`. The legacy `rust/v*` and `python/v*` tags are history — do not add more.

## Wheels

RLMesh publishes wheels only; do not build or upload an sdist. Wheel builds are host-specific:

- macOS (`mise run release:python:wheels:macos`) builds the full macOS, Linux, and Windows matrix.
- Linux (`mise run release:python:wheels:linux`) builds the Linux and Windows subset.

`python scripts/check_python_wheels.py python/rlmesh/dist` validates ABI/platform tags and payload contents. Release validation rejects plain `linux_*` tags; uploadable Linux wheels use `manylinux` or `musllinux`. Confirm license payloads with `mise run release:artifacts:licenses` before uploading.

## Publish order

`release.py --publish` publishes the crates in dependency order (`rlmesh-proto`, `rlmesh-spaces`, `rlmesh-adapters`, `rlmesh-cli`, `rlmesh-runtime`, `rlmesh-grpc`, `rlmesh-sandbox`, `rlmesh`), then uploads the wheels with `maturin`, then cuts the GitHub Release. `cargo publish` waits for each crate to appear in the index before the next one builds, so the ordered run is safe to leave unattended.

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
