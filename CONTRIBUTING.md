# Contributing

RLMesh is currently in beta. Issues and focused pull requests are welcome, and
maintainers may be selective while the API and package structure are still
settling.

For larger API, architecture, dependency, or roadmap changes, please open an
issue before starting implementation work. Small bug fixes, docs improvements,
tests, and narrowly scoped compatibility fixes can come directly as pull
requests.

Default project contact is research@competesai.com.

## Pull Request Expectations

Use focused pull requests rather than broad refactors. Internal contributors and
external contributors should follow the same PR template and validation
expectations.

PRs should explain the intent, scope, user-facing impact, and checks run.
Maintainers may ask for smaller scope, additional tests, or follow-up issues
before merge.

Changes that touch public Python or Rust APIs should update API snapshots and
generated native stubs when needed. Changes that touch packaging, wheels,
sandboxing, transport, or compatibility should explain which system runner
profile was run or why it was skipped.

## Local Setup

```bash
mise run setup
```

## Checks

Run these before opening a pull request:

```bash
mise run check
mise run test
```

For release-oriented validation, build wheels and run the installed-artifact
system runner:

```bash
mise run release:check
```

Maintainer release process notes live in [docs/release.md](docs/release.md).

## Development Notes

- Keep Python public API changes reflected in
  `python/rlmesh/tests/api_contract/snapshots/public_api.json`.
- Keep generated native stubs current with `mise run stubs:generate`.
- Avoid broad refactors in release-prep changes; keep changes scoped and
  testable.
