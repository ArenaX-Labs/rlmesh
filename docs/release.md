# Release Process

RLMesh is versioned, gated, and tagged from a maintainer's machine; pushing the tag hands off to CI (`.github/workflows/release.yml`), which builds the wheel matrix on Linux and macOS runners and publishes the Python wheels to PyPI, the Rust crates to crates.io and, when switched on, the npm packages to npm. SemVer is the source of truth (`Cargo.toml [workspace.package].version`); the PEP 440 spelling for PyPI is derived from it. The version contract is in [versioning](versioning.md).

The mechanical steps are scripted. You still own the changelog prose and the version number. The one irreversible action, pushing the tag, never runs without you asking.

## One-command flow

```bash
# Curate the changelog first (see below), then:
python scripts/release.py X.Y.Z --dry-run   # bump + full gate; no commit, tag, or publish
python scripts/release.py X.Y.Z             # also commit and tag vX.Y.Z (does NOT push)
git push origin HEAD --tags                 # you push; CI does the rest
```

The gate is `mise run release:check`. Among its steps, `test:sandbox:docker --require` runs `SandboxEnv` and `SandboxModel` against real containers built from the release's wheel (see [testing](testing.md#sandbox-docker-gate)); it is required, so run the release from a machine with a Docker daemon. A failure there blocks the release like any other.

`release.py` refuses to proceed if the working tree is dirty, the `vX.Y.Z` tag already exists, the changelog still has `<!-- DRAFT -->` markers, or there is no `## [X.Y.Z]` changelog section.

The tag push triggers the release workflow: Linux/Windows wheels build on a Linux runner and macOS wheels on a macOS runner (each system-tested against the built artifacts), then a publish job validates the assembled 14-wheel matrix and publishes to crates.io and PyPI through trusted publishing, then creates the GitHub Release with the curated release notes. When npm is switched on, its own jobs build, pack and smoke-test the npm packages beside the wheels and publish them after that (see [npm](#npm)). Retry failed jobs only, preserving successful builder artifacts (see GitHub Releases below). Registry pushes skip existing package versions. In 0.2 the GitHub Release has no native distribution assets.

## Prerequisites

- **Future Unity release gate (outside 0.2):** complete the [Unity release checklist](unity-release-checklist.md): Linux IL2CPP player smoke, Mono Editor reload, and failed-barrier reload. Unity support must not be claimed until all three pass; .NET tests and NativeAOT do not substitute for Unity verification.

- Nothing local: no publish tokens. crates.io, PyPI and npm authenticate via OIDC trusted publishing scoped to `release.yml` and the `release` GitHub environment; the GitHub Release uses the workflow's own token.
- One-time registry setup: a trusted publisher configured on the PyPI `rlmesh` project, on each of the nine published `rlmesh*` crates and on the npm `rlmesh` and five `@rlmesh/{platform}-{arch}` packages (repository `ArenaX-Labs/rlmesh`, workflow `release.yml`, environment `release`), plus the `release` environment created in the repository settings (add required reviewers there if you want a manual approval gate before publish).
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

It rewrites every manifest and install snippet and the release status, runs `cargo update` and `uv lock`, then `policy:check` — the backstop that fails loudly if any version-bearing spot was missed. A version bump never changes the workflow edition. A stable release seals a provisional current edition in `rlmesh.toml` with `sealed_in` and `spec_sha256`. See [workflow editions](editions/index.md).

**Before 0.2.0, a push that breaks the session wire bumps `WIRE_REVISION`.** Until 0.2.0 freezes the realtime protocol, any change that stops a build from talking to the previous one (an encoding, a key, the transport) bumps `WIRE_REVISION` in `crates/rlmesh-session-core/src/wire/version.rs` in the same push, with a line in its `REVISIONS` naming what changed, so two builds that cannot talk refuse each other by name instead of timing out. The prerelease number (`mise run bump 0.2.0-alpha.N+1`) is bumped when the release is cut, together with any wire revision change since the last release, so two releases never share a number across a wire break. A change that keeps the wire compatible needs neither.

## Freeze a wire protocol

The real-time session protocol (`rlmesh.session.v2`, `[protocols."rlmesh-realtime-v1"]` in `rlmesh.toml`) is provisional until the 0.2.0 release freezes it. Anything in `session.proto` then is permanent, so delete dead messages and values before. In the release commit:

1. Set its `status = "frozen"` and `frozen_in = "v0.2.0"`.
2. Run `mise run protocol:baseline-regen rlmesh-realtime-v1` to snapshot `crates/rlmesh-session-core/baselines/rlmesh-realtime-v1`.
3. Run `mise run check`. From then on `protocol:baseline-verify` keeps the live tree byte-identical to the baseline, and `protocol:breaking` checks it against the tag. The byte goldens (`crates/rlmesh-session-core/tests/vectors/wire`) already catch any change to an encoding.

## Tag scheme

One unscoped annotated tag per release: `vX.Y.Z`. The legacy `rust/v*` and `python/v*` tags are history — do not add more.

## Wheels

RLMesh publishes wheels only; do not build or upload an sdist. Wheel builds are host-specific, and CI runs both halves in parallel:

- macOS (`mise run release:python:wheels:macos`) builds the macOS arm64 and x86_64 wheels.
- Linux (`mise run release:python:wheels:linux`) builds the Linux and Windows subset. The Windows wheels cross-build through maturin's embedded cargo-xwin and, like the other Windows cross-builds, archive and link with the pinned Rust toolchain's `llvm-tools` and `rust-lld` (`scripts/xwin-env.sh`); the host supplies only `clang`.

`python scripts/check_python_wheels.py python/rlmesh/dist` validates ABI/platform tags and payload contents. Release validation rejects plain `linux_*` tags; uploadable Linux wheels use `manylinux` or `musllinux`. The publish job validates license payloads (`release:artifacts:licenses`) over the merged wheel matrix and freshly packaged crates before anything uploads.

## Publish order

The publish job uploads the wheels to PyPI first. When the `RLMESH_PUBLISH_CRATES` repository variable is `1`, it then runs `release.py --publish-crates`, which publishes the crates in dependency order (`rlmesh-spaces`, `rlmesh-proto`, `rlmesh-tether`, `rlmesh-session-core`, `rlmesh-adapters`, `rlmesh-session`, `rlmesh-cockpit`, `rlmesh`, `rlmesh-cli`); otherwise it skips them with a notice. Last, it cuts the GitHub Release via `release.py --github-release`. When the `RLMESH_PUBLISH_NPM` repository variable is `1`, a separate `publish-npm` job then publishes the npm packages; otherwise the publish job skips them with a notice (see [npm](#npm)). Trusted publishing on crates.io can be configured only for a crate that already exists, so a crate never published before is published once by hand with a token, its trusted publisher configured, and only then is `RLMESH_PUBLISH_CRATES` set to `1`. `cargo publish` waits for each crate to appear in the index before the next one builds, so the ordered run is safe to leave unattended.

## npm

The `RLMESH_PUBLISH_NPM` repository variable switches the npm release, as `RLMESH_PUBLISH_CRATES` switches the crates: set it to `1` to build and publish the npm packages with the next tag, with no code change. Unset or anything else, a tag run builds no addons and publishes nothing to npm, and the publish job leaves a notice saying so.

When it is `1`, a tag run adds the `npm-*` jobs beside the wheel jobs. `npm-linux` runs `build:js` (the wasm and the bundles) and `release:js:native` for `linux-x64`, `linux-arm64` (cargo-zigbuild, glibc 2.28, Node 22's own floor) and `win32-x64` (cargo-xwin); `npm-macos` adds `darwin-arm64` and `darwin-x64`. `npm-pack` runs `release:js:package --require-all`, which packs one `@rlmesh/{platform}-{arch}` per addon and `rlmesh` with those five as exact-version optional dependencies, all at the Cargo workspace version. npm installs only the addon whose `os`/`cpu`/`libc` match, and `native.ts` loads it. `npm-pack` then checks the tarballs (one version, no `fleet.js`, an addon only in each platform package, built for its target, the glibc floor) and installs and runs `examples/node/hold.mjs` from them on linux-x64 (`release:js:smoke`); `npm-smoke-macos` does the same on darwin-arm64.

Last, `publish-npm` runs `release:js:publish`, which publishes the platform packages first, skips any version already on npm, and sends a pre-release to the `next` dist-tag so `npm install rlmesh` keeps the last stable release. It authenticates by trusted publishing, so it runs in the `release` environment with `id-token: write`; Node and npm come from `mise.lock` (npm 11.5.1 or newer is needed for trusted publishing). If the environment has required reviewers, `publish-npm` asks for its own approval.

`publish-npm` waits for the publish job, so npm goes out after PyPI, the crates and the GitHub Release, and the publish job needs none of the npm jobs. A failed npm build, smoke test or publish therefore never holds up or undoes the other registries or the Release: the run ends red, and re-running its failed jobs retries only npm. npm goes after the GitHub Release, not before it, because otherwise an npm failure would either hold the Release back or have to be hidden with `continue-on-error`; the cost is a few minutes in which the Release is up and npm is not.

npm configures a trusted publisher per package, and only on a package that already exists. `rlmesh` and each `@rlmesh/{platform}-{arch}` were reserved with a placeholder `0.0.1`, and each has a trusted publisher (repository `ArenaX-Labs/rlmesh`, workflow `release.yml`, environment `release`). A new platform package needs the same before it ships.

To rehearse it without publishing, dispatch the workflow on any branch with `gh workflow run release.yml --ref <branch> -f rehearse=npm`. Only `npm-linux`, `npm-macos`, `npm-pack` and `npm-smoke-macos` run, on a read-only token with no environment or secrets, with the same tasks as a release; `publish-npm` and the release jobs skip. `release:js:smoke` also runs locally after `release:js:package`. The tarballs stay on the run as `npm-packages` for three days.

### First npm release

1. Rehearse the release commit (`gh workflow run release.yml --ref <branch> -f rehearse=npm`) and check every `npm-*` job passes.
2. Check that `rlmesh` and the five `@rlmesh/{platform}-{arch}` packages each list the trusted publisher above.
3. Before pushing the tag, set the variable: `gh variable set RLMESH_PUBLISH_NPM --body 1`. A tag run that starts without it skips the npm jobs, and re-running its failed jobs does not add them.
4. After `publish-npm`, check `npm view rlmesh dist-tags`: a pre-release is on `next`, and `latest` stays on the placeholder `0.0.1`.
5. Point anyone who installs the placeholder at the pre-releases, from an npm account that owns `rlmesh` (trusted publishing only publishes):

   ```bash
   npm deprecate rlmesh@0.0.1 "placeholder: install rlmesh@next for the 0.2 pre-releases"
   ```

6. Install it from the registry in a scratch directory and open a host, which loads the addon through `rlmesh` and checks its version:

   ```bash
   mkdir /tmp/rlmesh-npm-smoke && cd /tmp/rlmesh-npm-smoke && npm init -y >/dev/null
   npm install rlmesh@next
   node --input-type=module -e "import { Host } from 'rlmesh'; const h = await Host.open(); console.log(h.address); await h.close();"
   ```

Leave the variable at `1` to publish npm with every release, or delete it (`gh variable delete RLMESH_PUBLISH_NPM`) to go back to Python only.

## GitHub Releases

Every release gets a GitHub Release built from its tag and curated changelog.
For 0.2, the published distributions are the Python wheels, the nine Rust
crates above once `RLMESH_PUBLISH_CRATES` is on, and the six npm packages once
`RLMESH_PUBLISH_NPM` is on (0.2.0-alpha.3 ships the wheels only). npm attaches
nothing to the Release. C/C++, C#, Unity, Unreal and board-library release assets are
fenced; their code and existing CI build/test coverage remain in the tree.
C ABI 7 is cut only when the native bindings ship.

`release.py --github-release` does not require or attach native packages in 0.2,
even if they exist locally. Prereleases (`-alpha.N`, `-beta.N`, `-rc.N`) are marked
`--prerelease` so they stay out of "Latest"; a final `X.Y.Z` becomes Latest.

**Re-run failed jobs only** for releases, preserving successful builder artifacts.
If a successful builder must be replaced after publishing began, stop and plan
a new version instead of rebuilding the release in place.

## Fenced distribution machinery

The sections below document retained packaging machinery for future releases.
These release workflow steps are disabled in 0.2, including signing, artifact
assembly, uploads and registry authentication. Configuring credentials or
`NUGET_USER` does not enable them; re-enabling publication needs a separate
release decision and the relevant integration gates.

## C# and Unity distributions

When re-enabled, the Linux builder also runs `csharp:pack` and `csharp:upm`, installing the scoped
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

The fenced tag-only publish steps use GitHub OIDC via `NuGet/login` (pinned to
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

### Signed macOS CLI in the wheels

On a tag, `wheels-macos` builds a universal `rlmesh` (`mise run release:cli:macos`, with the cockpit) and signs it as `dev.rlmesh.cli` with the Developer ID identity and hardened runtime, using the script and secrets described below. The macOS wheels then carry its thinned slice as `rlmesh/_bin/rlmesh`, and `rlmesh`, `rlmeshd`, `docker-credential-rlmesh` and `python -m rlmesh` exec it, so the process that touches the Keychain is the signed binary rather than the Python interpreter. Keychain approvals attach to that signature's designated requirement (identifier plus team), so they survive upgrades.

`docker-credential-rlmesh` and `rlmeshd` are the same binary under another name: `rlmesh` dispatches on `argv[0]`.

This step needs only the three certificate secrets (`APPLE_DEVELOPER_ID_P12`, `APPLE_DEVELOPER_ID_P12_PASSWORD`, `APPLE_TEAM_ID`). Notarization is optional here and runs when all three `APPLE_NOTARY_*` secrets are set: Gatekeeper consults notarization only for quarantined files, pip and uv do not quarantine what they install, and a bare Mach-O cannot carry a stapled ticket. A partial set of notary secrets fails the tag.

With `RLMESH_ALLOW_UNSIGNED_MACOS=true` and no certificate secrets, the wheels carry no binary and behave as before.

### macOS signing and notarization

The retained native macOS steps belong to the tag-only `wheels-macos` job. It builds
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

## Post-publish smoke

```bash
python -m venv /tmp/rlmesh-smoke
/tmp/rlmesh-smoke/bin/python -m pip install "rlmesh==VERSION"  # the version just published
/tmp/rlmesh-smoke/bin/python -c "import rlmesh; print(rlmesh.__version__)"
```

## Publish the docs

The published user documentation is at [rlmesh.dev/docs](https://rlmesh.dev/docs/). Generate the Python API reference from the release tag with `uv run rlmesh-api-surface docs-api-surface --output-dir <dir>` so it describes the released package.

The Rust, C, and C++ API references are early access and come from the same extractor: `uv run rlmesh-api-surface docs-api-surface --language rust|c|cpp --output-dir <dir>`. Rust documents the `rlmesh` facade crate from rustdoc JSON (it runs `cargo rustdoc` itself); C and C++ parse `rlmesh.h` and `rlmesh.hpp` with libclang, one page per `---- section ----` divider. Each snapshot carries the `access` label and summary from `[api_surface.<language>]` in `rlmesh.toml`, and the extractor warns about every undocumented item.

The C++ reference uses C++20. Header extraction discovers system include paths through `CC` or `CXX` (defaulting to `cc` or `c++`); select a compiler whose standard-library headers the pinned libclang can parse.

## Recovery

- A bad crates.io or PyPI publish cannot be deleted, only yanked. Yank it and ship a fixed patch release.
- A bad npm publish: `npm deprecate` the version and ship a fixed one. npm restricts unpublishing (its unpublish policy) and never reuses a version number.
- If the seal gate blocks the release, the edition metadata does not match the version: rerun `mise run bump X.Y.Z` and review `rlmesh.toml`, or keep the release a pre-release.
