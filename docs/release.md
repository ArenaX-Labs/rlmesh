# Release Process

RLMesh is versioned, gated, and tagged from a maintainer's machine; pushing the tag hands off to CI (`.github/workflows/release.yml`), which builds the wheel matrix on Linux and macOS runners and publishes everything. SemVer is the source of truth (`Cargo.toml [workspace.package].version`); the PEP 440 spelling for PyPI is derived from it. The version contract is in [versioning](versioning.md).

The mechanical steps are scripted. You still own the changelog prose and the version number. The one irreversible action, pushing the tag, never runs without you asking.

## One-command flow

```bash
# Curate the changelog first (see below), then:
python scripts/release.py X.Y.Z --dry-run   # bump + full gate; no commit, tag, or publish
python scripts/release.py X.Y.Z             # also commit and tag vX.Y.Z (does NOT push)
git push origin HEAD --tags                 # you push; CI does the rest
```

`release.py` refuses to proceed if the working tree is dirty, the `vX.Y.Z` tag already exists, the changelog still has `<!-- DRAFT -->` markers, or there is no `## [X.Y.Z]` changelog section.

The tag push triggers the release workflow: Linux/Windows wheels build on a Linux runner and macOS wheels on a macOS runner (each system-tested against the built artifacts), then a publish job validates the assembled 14-wheel matrix and ships crates.io, PyPI, npm, the GitHub Release, and (once configured) nuget.org through trusted publishing. Retry failed jobs only, preserving successful builder artifacts (see GitHub Releases below). Registry pushes skip existing package versions, and the GitHub Release upload preserves already-published assets.

## Prerequisites

- **v0.2 Unity release gate:** complete the [Unity release checklist](unity-release-checklist.md): Linux IL2CPP player smoke, Mono Editor reload, and failed-barrier reload. Unity support must not be claimed until all three pass; .NET tests and NativeAOT do not substitute for Unity verification.

- Nothing local: no publish tokens. crates.io, PyPI and npm all authenticate via OIDC trusted publishing scoped to `release.yml` and the `release` GitHub environment; the GitHub Release uses the workflow's own token.
- One-time registry setup: a trusted publisher configured on the PyPI `rlmesh` project and on each of the eight published `rlmesh*` crates and the six npm packages (`rlmesh` and the five `@rlmesh/{platform}-{arch}`; see [npm](#npm)) (repository `ArenaX-Labs/rlmesh`, workflow `release.yml`, environment `release`), plus the `release` environment created in the repository settings (add required reviewers there if you want a manual approval gate before publish).
- Break-glass local publish, from the tagged commit: no single host builds the full wheel matrix anymore, so first assemble `python/rlmesh/dist` from the release run's artifacts (`gh run download <run-id> -p 'wheels-*'`, then move the `.whl` files in) or build each half on its own host without re-running `release:python:prepare` in between (it wipes dist). Then `python scripts/release.py X.Y.Z --publish-crates` (needs `CARGO_REGISTRY_TOKEN`) and `--github-release` (needs `gh` auth); wheels would need a manual `twine upload`.

## Curate the changelog

The changelog is hand-written. `git-cliff` is gone.

1. `mise run changelog:draft` appends draft bullets under `## [Unreleased]` in `CHANGELOG.md`, one per user-facing commit since the last `v*` tag, each marked `<!-- DRAFT -->`.
2. Rewrite each bullet in your own words, drop internal-only changes, and group them under the Keep a Changelog sections. **Delete every `<!-- DRAFT -->` marker** — the release driver refuses to ship while any remain.
3. Rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a fresh empty `## [Unreleased]` above it, and add the `[X.Y.Z]: https://github.com/ArenaX-Labs/rlmesh/releases/tag/vX.Y.Z` link reference at the bottom.

A breaking change to a stable symbol gets a `### Breaking` entry with a before/after migration note (see [versioning](versioning.md)).

## Bump the version

`release.py` runs this for you, or run it alone:

```bash
mise run bump X.Y.Z
```

It rewrites every manifest and install snippet, updates the workflow cohort, runs `cargo update` and `uv lock`, then `policy:check` — the backstop that fails loudly if any version-bearing spot was missed. Prereleases use an exact provisional cohort (`YYYY.MM-X.Y.Z-beta.N`); a stable release seals the bare edition in `rlmesh.toml` with `sealed_in` and `spec_sha256`. See [workflow editions](editions/index.md).

## Tag scheme

One unscoped annotated tag per release: `vX.Y.Z`. The legacy `rust/v*` and `python/v*` tags are history — do not add more.

## Wheels

RLMesh publishes wheels only; do not build or upload an sdist. Wheel builds are host-specific, and CI runs both halves in parallel:

- macOS (`mise run release:python:wheels:macos`) builds the macOS arm64 and x86_64 wheels.
- Linux (`mise run release:python:wheels:linux`) builds the Linux and Windows subset.

`python scripts/check_python_wheels.py python/rlmesh/dist` validates ABI/platform tags and payload contents. Release validation rejects plain `linux_*` tags; uploadable Linux wheels use `manylinux` or `musllinux`. The publish job validates license payloads (`release:artifacts:licenses`) over the merged wheel matrix and freshly packaged crates before anything uploads.

## Publish order

The publish job runs `release.py --publish-crates`, which publishes the crates in dependency order (`rlmesh-proto`, `rlmesh-spaces`, `rlmesh-session-core`, `rlmesh-adapters`, `rlmesh-session`, `rlmesh-cli`, `rlmesh-grpc`, `rlmesh`), then uploads the wheels to PyPI, publishes NuGet packages when configured, publishes the npm packages, then cuts the GitHub Release via `release.py --github-release`. `cargo publish` waits for each crate to appear in the index before the next one builds, so the ordered run is safe to leave unattended.

## GitHub Releases

Every release gets a GitHub Release built from its tag. The Linux wheel runner
produces `rlmesh-X.Y.Z-x86_64-unknown-linux-gnu.tar.gz` and
`rlmesh-X.Y.Z-x86_64-pc-windows-msvc.tar.gz` via `release:cxx:package` and
`release:cxx:package --target windows`. The macOS wheel runner produces
`rlmesh-X.Y.Z-universal-apple-darwin.tar.gz` via `release:cxx:package --target macos`.
All three tarballs and their `.sha256` files are attached alongside the C# packages.
A rerun uploads only missing assets and never replaces one: CMake `FetchContent`
users pin their SHA-256. A break-glass `--github-release` needs all three tarballs
and the final NuGet/UPM packages downloaded from the original build first.
Prereleases (`-beta.N`, `-rc.N`) are marked `--prerelease` so they stay out of
"Latest"; a final `X.Y.Z` becomes Latest automatically.

**Re-run failed jobs only** for releases. Keep the successful builder artifacts
from the original run. Rebuilding a successful macOS job changes signing timestamps
and therefore tarball/nupkg bytes; nuget.org skips the already-published version,
while a later GitHub upload could otherwise attach a different rebuild of that
same version. Retry only the failed publish job with the original inputs. If a
successful builder must be replaced after publishing began, stop and plan a new
version instead of rebuilding the release in place.

## C# and Unity distributions

The Linux builder also runs `csharp:pack` and `csharp:upm`, installing the scoped
dotnet SDK only for packaging. These preliminary Linux/Windows packages verify
assembly locally on that builder and are not uploaded; publish rebuilds the final
packages after downloading both builders' C tarballs. Final assembly uses
`--require-all`: `RLMesh.X.Y.Z.nupkg`, `RLMesh.X.Y.Z.snupkg`, and
`com.rlmesh.rlmesh-X.Y.Z.tgz`, each attached to the GitHub Release with a SHA-256
sidecar. Final assembly refuses a missing native target. NuGet includes both
`netstandard2.1` and `net10.0`, and native assets for `linux-x64`, `win-x64`,
`osx-arm64` and `osx-x64`. The two macOS RIDs contain the same universal dylib.
UPM contains the same netstandard DLL and native bytes, with Unity importer
settings; no source compilation or asmdef is required for a prebuilt DLL.

### Enable nuget.org trusted publishing

The existing tag-only publish job uses GitHub OIDC via `NuGet/login` (pinned to
v1.2.0's full commit SHA), then pushes the nupkg and snupkg to
`https://api.nuget.org/v3/index.json` with `--skip-duplicate`. It already grants
`id-token: write` and uses the GitHub environment **release**. There is no
long-lived NuGet API key. Login and push run only when the repository variable
`NUGET_USER` is nonempty; otherwise the job emits a notice and still attaches
the packages to the GitHub Release.

Complete this one-time setup before setting that variable:

1. Sign into nuget.org as the account that will publish `RLMesh`. Ensure that
   account has permission to publish under the intended package owner (the user
   or organization). In **Trusted Publishing**, create a GitHub policy owned by
   that NuGet user/organization; the policy's owner is the package owner, not a
   GitHub repository name. Select the `RLMesh` package scope (or the narrowly
   matching package-ID pattern if it has not yet been published), with permission
   to push new packages/versions.
2. Set **Repository Owner** to `ArenaX-Labs`, **Repository** to `rlmesh`,
   **Workflow File** to `release.yml` (filename only, without `.github/workflows/`),
   and **Environment** to `release`, matching `jobs.publish.environment`.
3. In GitHub's `ArenaX-Labs/rlmesh` repository settings, under **Secrets and
   variables → Actions → Variables**, add the repository variable `NUGET_USER`.
   Its value is the policy creator's nuget.org profile username, not an email
   address or organization name. That user must be authorized for the policy's
   NuGet owner. Keep this variable unset/empty until the policy is ready.
4. Confirm the policy is active, then publish a release tag or retry only the
   publish job in the existing tag run. If nuget.org reports pending full activation, publish during its
   activation window (or restart the window in nuget.org before rerunning).

See [NuGet's trusted-publishing setup](https://learn.microsoft.com/en-us/nuget/nuget-org/trusted-publishing).
The action returns a temporary key used only by the push step; do not print it,
turn on shell tracing, or store it as a repository secret. Reruns skip existing
package and symbol versions. Until configured, use GitHub's nupkg as a local feed.

### macOS signing and notarization

Only the existing tag-only `wheels-macos` job builds macOS artifacts. It builds
`rlmesh-capi` for `aarch64-apple-darwin` and `x86_64-apple-darwin`, combines both
the dylib and static archive with `lipo`, and changes the dylib ID only if it
differs from `@rpath/librlmesh_capi.dylib`, before signing. `release:cxx:package --target macos`
requires Darwin and these prepared universal libraries; it preserves signed bytes.
The CMake and pkg-config metadata support the packaged headers and libraries.

Configure these secrets for the tag build. Prefer a protected GitHub `release`
environment with required reviewers and deployment tag rules for Apple credentials.
If using environment secrets, assign the existing `wheels-macos` job to that
environment before removing repository secrets; currently it reads repository
secrets. This keeps the protection change explicit and avoids unexpectedly gating
all macOS wheel builds during this fix round:

- `APPLE_DEVELOPER_ID_P12`: base64-encoded Developer ID Application `.p12`.
- `APPLE_DEVELOPER_ID_P12_PASSWORD`: its import password.
- `APPLE_TEAM_ID`: the team ID in the `Developer ID Application: … (TEAM_ID)` identity.
- `APPLE_NOTARY_KEY`: App Store Connect `.p8` contents.
- `APPLE_NOTARY_KEY_ID` and `APPLE_NOTARY_ISSUER_ID`: the corresponding API identifiers.

The runner exclusively creates mode-0600 key files, imports into a temporary
keychain, and uses `codesign --options runtime --timestamp`. Signing has a 5-minute
step budget; notarization has a separate 17-minute step budget, including a
`notarytool submit --wait --timeout 15m` limit and retrieval of `notarytool log`
for non-Accepted responses. The universal build has a 40-minute timeout. An
`always()` step deletes the keychain and credential files after either phase.
Apple's `security` CLI takes the import/keychain passwords in argv. That is
accepted only on these ephemeral, isolated release runners: never enable tracing,
echo credentials, or reuse this flow on a shared persistent host.

The user approved this release policy on 2026-10-20. On `refs/tags/v*`, any
missing Apple secret **fails** signing by default. The
explicit repository variable `RLMESH_ALLOW_UNSIGNED_MACOS=true` permits an
exception: the workflow warns, ad-hoc signs with `codesign --force --sign -`,
verifies that signature, and skips notarization. It produces **no Developer ID or
notarization assurance**, and published versions are immutable; remove the
exception before the next release. Non-tag local script probes may also use the
ad-hoc branch. Complete credentials always select Developer ID signing.

The separate `macos-notarization` workflow artifact holds `notarization.json`
and any failure log. It deliberately does not match `capi-*`: each C artifact
uploads only direct children of `target/capi-pkg`, preserving the flat extraction
layout final NuGet/UPM assembly needs. A cheap macOS smoke extracts the C tarball,
loads its dylib with Python ctypes and checks `rlmesh_abi_version`, then compiles
and runs a tiny C consumer using its pkg-config metadata and an executable rpath.

Verify the extracted release on macOS:

```sh
lipo -archs lib/librlmesh_capi.dylib
lipo -archs lib/librlmesh_capi.a
otool -D lib/librlmesh_capi.dylib
codesign --verify --strict --verbose=2 lib/librlmesh_capi.dylib
codesign -d --verbose=4 lib/librlmesh_capi.dylib
xcrun notarytool info SUBMISSION_ID --key /secure/AuthKey.p8 --key-id KEY_ID --issuer ISSUER_ID
xcrun notarytool log SUBMISSION_ID --key /secure/AuthKey.p8 --key-id KEY_ID --issuer ISSUER_ID
```

Check the expected team/authority and Accepted status. A loose dylib cannot have
a stapled notarization ticket; do not use `stapler validate` as its acceptance
test. The submitted zip and all distributions contain the same signed dylib.
Unity/IL2CPP execution remains a separate engine integration gate; NativeAOT is
only a supplement.

## npm

The Linux builder runs `build:js` (the wasm and the bundles) and
`release:js:native` for `linux-x64`, `linux-arm64` (cargo-zigbuild, glibc 2.28,
Node 22's own floor) and `win32-x64` (cargo-xwin); the macOS builder adds
`darwin-arm64` and `darwin-x64`. Publish runs `release:js:package --require-all`,
which packs one `@rlmesh/{platform}-{arch}` per addon and `rlmesh` with those
five as exact-version optional dependencies, all at the Cargo workspace
version. npm installs only the addon whose `os`/`cpu`/`libc` match, and
`native.ts` loads it. `release:js:publish` publishes the platform packages
first, skips any version already on npm, and sends a pre-release to the `next`
dist-tag so `npm install rlmesh` keeps the last stable release.

npm configures a trusted publisher per package, and only on a package that
already exists. `rlmesh` was reserved with a placeholder `0.0.1`; before the
first release, publish a placeholder `0.0.1` of each `@rlmesh/{platform}-{arch}`
the same way, then add the trusted publisher (repository `ArenaX-Labs/rlmesh`,
workflow `release.yml`, environment `release`) in each package's settings.

## Post-publish smoke

```bash
python -m venv /tmp/rlmesh-smoke
/tmp/rlmesh-smoke/bin/python -m pip install rlmesh
/tmp/rlmesh-smoke/bin/python -c "import rlmesh; print(rlmesh.__version__)"
```

## Publish the docs

rlmesh.dev/docs is built from the platform repo, which keeps a frozen copy of the docs for the latest release at `/docs/` and the working copy for `main` at `/docs/dev/`. After a final `X.Y.Z` release, cut the docs there with this checkout at the tag:

```bash
RLMESH_OSS_DIR=/path/to/rlmesh mise run docs:cut -- X.Y.Z   # OSS checkout at vX.Y.Z
RLMESH_OSS_DIR=/path/to/rlmesh mise run gen:api-snapshot:dev   # OSS checkout back on main
```

Open the result as a `docs: cut X.Y.Z` PR in the platform repo. It goes live on the next platform promote. Skip this for pre-releases.

The Rust, C, and C++ API references are early access and come from the same extractor: `uv run rlmesh-api-surface docs-api-surface --language rust|c|cpp --output-dir <dir>`. Rust documents the `rlmesh` facade crate from rustdoc JSON (it runs `cargo rustdoc` itself); C and C++ parse `rlmesh.h` and `rlmesh.hpp` with libclang, one page per `---- section ----` divider. Each snapshot carries the `access` label and summary from `[api_surface.<language>]` in `rlmesh.toml`, and the extractor warns about every undocumented item.

The C++ reference uses C++20. Header extraction discovers system include paths through `CC` or `CXX` (defaulting to `cc` or `c++`); select a compiler whose standard-library headers the pinned libclang can parse.

## Recovery

- A bad crates.io or PyPI publish cannot be deleted, only yanked. Yank it and ship a fixed patch release.
- If the seal gate blocks the release, the edition metadata does not match the version: rerun `mise run bump X.Y.Z` and review `rlmesh.toml`, or keep the release a pre-release.
