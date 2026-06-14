# Workflow Editions

A workflow edition is a named, immutable behavioral contract for RLMesh workflow semantics. The
edition string (`YYYY.MM`) identifies one spec document in this section; that document is the
contract, not the implementation. Exactly one edition governs a session, chosen during handshake.

Editions answer a different question than the protocol generation. The protocol generation
(`rlmesh.protocol.v1`) names the wire shape: which services, messages, and fields exist. The edition
names what a conforming interaction over that shape _means_: lifecycle, ordering, episode
accounting, and error semantics.

## Negotiation

The client sends every edition it can operate under in
`HandshakeRequest.supported_workflow_editions`. The server intersects that offer with its own
supported set and selects the highest mutual edition; the zero-padded `YYYY.MM` format makes
lexicographic order chronological order. The selection is returned in
`HandshakeResponse.selected_workflow_edition` and governs the rest of the session.

- An empty intersection means `compatible = false`. The response lists the server's supported
  editions for diagnostics, but there is no second round trip because the client's offer was already
  complete.
- Servers accept only editions they explicitly support. A server never accepts an unknown edition on
  the assumption that it is probably compatible; forward compatibility lives in the client's offer
  set, not in server leniency.

## Edition vs. Capability vs. Bug Fix

Most development never touches the edition:

- A change to the meaning of an existing, conforming interaction mints a new edition. This is rare,
  and breaking semantic changes batch into at most one new edition per release.
- A new addition that is ignorable or detectable, such as a new RPC, a new field, or an opt-in
  behavior, is a capability or a plain feature. No edition.
- An implementation that deviates from the governing spec document has a bug. Fixing it needs no
  edition.

## Lifecycle: Provisional, Then Sealed

An edition is **provisional** while no stable release has shipped it: its spec document may still be
edited in place, and beta releases may change its semantics as a hard break. The first stable
release that ships an edition **seals** it permanently: the spec document becomes immutable
(enforced by checksum), and any later semantic change mints a new edition.

`2026.06` is provisional during the 0.1 beta series and seals when v0.1.0 ships. After sealing it
remains valid indefinitely; a new edition is minted only by a deliberate semantic redesign, never on
a schedule.

## Support Window

Once a stable release seals an edition, every later release keeps offering and accepting it. That
includes betas for a later edition, so newer builds can still negotiate with older peers. Sealed
editions are never pruned. Provisional editions, which no stable release has sealed, may change or
be dropped because they are content-pinned and interoperate only between matching builds.

## Enforcement

`rlmesh.toml` records every known edition under `[workflow.editions."<edition>"]` with its `status`
(`provisional` or `sealed`), `spec` path, and `spec_sha256`; a sealed edition also records
`sealed_in`, the stable release that sealed it. `scripts/check_rlmesh_policy.py` verifies the spec
files against those checksums and keeps the `rlmesh-proto` constants in sync with the manifest. It
also rejects provisional editions in stable releases, checks that sealed editions were sealed by
stable versions, and confirms `SUPPORTED_WORKFLOW_EDITIONS` matches the manifest.

```{toctree}
:maxdepth: 1

2026.06
```
