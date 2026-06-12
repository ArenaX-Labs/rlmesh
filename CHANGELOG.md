# Changelog

All notable RLMesh changes are tracked here.

## Unreleased

- **Breaking (beta):** workflow editions are now negotiated. Clients offer
  `supported_workflow_editions` in the handshake and servers return the highest mutual edition in
  `selected_workflow_edition`; the scalar `workflow_edition` fields are reserved, the `2026` legacy
  alias is removed, and servers no longer accept unknown future editions. Editions are documented in
  `docs/editions/` and tracked in `rlmesh.toml`; `2026.06` stays provisional until v0.1.0 seals it.

## 0.1.0-beta1

Initial OSS beta release.

- Python SDK and native extension package.
- Rust SDK, protocol, runtime, sandbox, and CLI crates.
- Installed-wheel validation harness for basic and optional heavy profiles.
- Public API snapshot tests for the Python package.
