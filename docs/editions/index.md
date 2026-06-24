# Workflow Editions

A workflow edition is a named behavioral contract for RLMesh workflow semantics. The base edition (`YYYY.MM`) identifies one spec document in this section; prerelease and local builds append a cohort suffix so moving builds fail closed unless both sides are from the same cohort. Exactly one edition governs a session, chosen during handshake.

```{note}
In 0.1.0-rc.1 the active cohort is `2026.06-0.1.0-rc.1`; the bare `2026.06` edition seals at the final 0.1.0.
```

Editions answer a different question than the protocol generation. The protocol generation (`rlmesh.protocol.v1`) names the wire shape: which services, messages, and fields exist. The edition names what a conforming interaction over that shape _means_: lifecycle, ordering, episode accounting, and error semantics.

## Negotiation

The client sends every edition it can operate under in `HandshakeRequest.supported_workflow_editions`. The server intersects that offer with its own supported set and selects the highest mutual edition. A matching suffixed cohort wins over its sealed fallback; if prerelease cohorts differ, peers can only interoperate through a sealed edition that both sides explicitly advertise. The selection is returned in `HandshakeResponse.selected_workflow_edition`. The runtime currently supports a single edition and refuses any other; making the selected edition drive runtime behavior is on the roadmap (see {doc}`../compatibility`).

- An empty intersection means `compatible = false`. The response lists the server's supported editions for diagnostics, but there is no second round trip because the client's offer was already complete.
- Servers accept only editions they explicitly support. A server never accepts an unknown edition on the assumption that it is probably compatible; forward compatibility lives in the client's offer set, not in server leniency.

## Edition vs. Capability vs. Bug Fix

Most development never touches the edition:

- A change to the meaning of an existing, conforming interaction mints a new edition. This is rare, and breaking semantic changes batch into at most one new edition per release.
- A new addition that is ignorable or detectable, such as a new RPC, a new field, or an opt-in behavior, is a capability or a plain feature. No edition.
- An implementation that deviates from the governing spec document has a bug. Fixing it needs no edition.

## Lifecycle: Provisional, Then Sealed

An edition is **provisional** while no stable release has shipped it: prerelease builds use exact release-cohort names such as `2026.06-0.1.0-rc.1`, and local source builds use exact `dev.<git>` cohort names. This prevents accidental interoperability between moving builds that have not had stable-release scrutiny. The first stable release that ships an edition **seals** the bare `YYYY.MM` name permanently: the spec document becomes immutable (enforced by checksum), and any later semantic change mints a new edition.

`2026.06` uses provisional cohorts through the 0.1 beta and release-candidate series and seals at v0.1.0. After sealing it remains valid indefinitely; a new edition is minted only by a deliberate semantic redesign, never on a schedule.

## Support Window

Sealing freezes an edition's spec by checksum. The intended support window — every later release keeps offering and accepting a sealed edition, including betas for a later edition, and sealed editions are never pruned — is a forward-compatibility guarantee on the roadmap (see {doc}`../compatibility`); it becomes binding at 1.0, not today. A provisional cohort, which no stable release has sealed, may change or be dropped and interoperates only with the same cohort unless both sides implement and advertise a sealed fallback.

## Enforcement

`rlmesh.toml` records the base edition, current official release cohort, supported editions, and each edition's `status` (`provisional` or `sealed`) plus `spec` path. A sealed edition also records `sealed_in` and `spec_sha256`. `scripts/check_rlmesh_policy.py` verifies sealed spec checksums, rejects provisional editions in stable releases, and checks that prerelease cohorts match the workspace SemVer. Local dev cohorts are generated at build time and are not committed to the manifest.

```{toctree}
:maxdepth: 1

2026.06
```
