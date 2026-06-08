# Contributing

RLMesh is currently in beta. Issues and focused pull requests are welcome. Maintainers may be
selective while the API and package structure are still settling.

For larger API, architecture, dependency, or roadmap changes, please open an issue before starting
implementation work. Small bug fixes, docs improvements, tests, and narrowly scoped compatibility
fixes can come directly as pull requests.

Please do not report vulnerabilities, credential exposure, sandbox escape concerns, or
dependency-chain compromises in public issues. Use the repository security policy or contact
research@competesai.com.

## Pull Request Expectations

Use focused pull requests rather than broad refactors. Internal and external contributors should
follow the same PR template and validation expectations.

PRs should explain the intent, scope, user-facing impact, and checks run. Maintainers may ask for
smaller scope, additional tests, or follow-up issues before merge.

Changes that touch public Python or Rust APIs should update API surface snapshots or generated
native stubs when needed. Changes that touch packaging, wheels, sandboxing, transport, or
compatibility should explain which system validation was run or why it was skipped.

## Local Setup

Install local tools, Python development dependencies, and git hooks:

```bash
mise run setup
```

More setup and build details live in [docs/local-dev.md](docs/local-dev.md).

## Checks

Run these before opening a pull request:

```bash
mise run check
mise run test
mise run release:check
```

Use targeted tests while iterating, then run the full checks before asking for review. If a check
cannot be run locally, explain why in the PR template. Test layers and system profiles are described
in [docs/testing.md](docs/testing.md).

Maintainer release process notes live in [docs/release.md](docs/release.md).

## Development Notes

- Keep Python public API changes reflected in
  `python/rlmesh/tests/api_surface/snapshots/api_surface.json`.
- Keep generated native stubs current with `mise run stubs:generate`.
- Keep release-prep changes scoped and testable.
