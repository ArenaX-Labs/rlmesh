# The browser binding and its gateway

Status: Experimental (protocol v2). The `rlmesh` npm package
(`javascript/rlmesh`) ships with each release; the API may change.

A browser page cannot open a v2 session's own transport: it has no raw TCP
or UDP, so neither Zenoh nor gRPC can run there. The browser binding is
three pieces instead:

- **The sans-IO core in wasm.** `javascript/rlmesh/wasm` (`rlmesh-wasm`) wraps `rlmesh-session-core`
  (the v2 wire and the state machines: the clock
  offset estimator, the holder side of a lease, the receive rules) with
  wasm-bindgen. One wasm Module is compiled once and instantiated in three
  Workers.
- **The browser's own WebSockets**, one per traffic class: `/control`,
  `/state` and `/bulk`, each in its own Worker, so a camera can only slow its
  own socket and the main thread never decodes a sample.
- **`rlmesh-gateway`**, a session participant on the session's own
  transport. It is not an authority of its own: the session host still
  grants, renews and lapses every lease.

```
 browser                                   rlmesh-gateway                          session (Zenoh, per class)
 ┌──────────────────────────────┐          ┌───────────────────────────────┐       ┌───────────────────────┐
 │ main thread: rlmesh API      │          │ /control: one participant per │──ctl──│ session host          │
 │  control Worker (wasm) ──────┼─/control─│  connection (hello → attach), │──hb───│  (authority, leases,  │
 │   heartbeats, leases, chunks │          │  heartbeats and chunks as sent│──c/───│   clock mappings)     │
 │  state Worker (wasm) ────────┼─/state───│ /state, /bulk: samples as     │──d/───│ robot (session edge)  │
 │  bulk Worker (wasm) ─────────┼─/bulk────│  published, tagged per sub    │       │  publishers, edges    │
 └──────────────────────────────┘          └───────────────────────────────┘       └───────────────────────┘
```

## What the browser is to the session

Each control connection is a participant of its own, attached with its
token's credential (see the security model): the gateway runs a separate
handshake for it and seals and opens for it, so the wasm holds no keys. On
the hello (`{"type": "hello", "priority": "model"}`) the gateway attaches a
controller at that class (capped by `--max-priority`, `model` by default,
and by the host at the token's device's cap; a page that takes over from a
model asks for `operator`, which the gateway must allow with
`--max-priority operator`), declares an identity binding over every
commandable part, and answers with the participant id. `"observer"`
attaches one that takes no leases. Each data connection reads through an
observer of its own under the same credential, so the host hands it only
the stream keys the token's device may read. From then on:

- **Leases are the host's.** `lease(path)` sends a real `GrantLease` (with
  the binding's fingerprint); the gateway forwards it through the
  participant and returns the host's `CtlReply`. Preemption, command groups
  and fencing are the host's rules. The browser's
  `rlmesh_session_core::lease::LeaseClient` only tracks a local deadline
  that is a conservative lower bound of the host's expiry
  (`expires − θ − ε`, and the request's send time plus the ttl): a late
  renewal never extends a lease that already expired locally.
- **Heartbeats are the browser's own.** The participant's own heartbeat loop
  stays off; the control Worker sends a heartbeat every 50 ms (in the
  browser's clock, with the previous four-timestamp exchange), the gateway
  forwards it as sent, and the host's echo comes back as sent. So the host
  renews a lease only while the browser itself is alive, and its clock
  mapping is the browser's, which the edge uses to time the browser's
  chunks. A browser whose heartbeats stop loses its lease after the
  session's lease grace (250 ms by default): the session goes SAFE.
- **Commands are timed action chunks.** `lease.drive(values, {hz})` runs in
  the control Worker: each chunk carries the target twice, one drive period
  apart, from session now, and is void after both. The robot's edge fences
  it (epoch, token, source, binding, its own clock rules), clamps it and
  holds the last target up to the part's `hold_max` before it goes safe. The
  gateway refuses a chunk naming another source and reports (but forwards) a
  chunk the current snapshot would fence.
- **Samples arrive as published.** The gateway subscribes on the stream's
  class and prepends a subscription id and a flags byte; the wasm decodes the
  bytes the publisher sent (`Bytes` slices, no re-encoding) and applies the
  session's receive rules (epochs, seq gaps as losses, layout checks,
  freshness against `max_age`). Capture and publish times convert to the
  browser's clock through the host's mappings.
- **A disconnect ends the participant.** When the control socket closes,
  the gateway drops the participant; the host sees its heartbeats stop and
  ends its leases (SAFE when one was active). The client reports the lease
  as revoked (`cause: 'disconnected'`) and never resumes it: the reconnect
  is a new participant, and `arm()` then `lease()` take control again.
- **A participant the host detaches closes its socket** with `4409`: its
  token's device was revoked (`rlmesh devices revoke` on the robot), or its
  entry changed. The socket reconnects, under the new entry, or is refused
  with `4401` once the host no longer knows the token's key.

Every binary frame is one tag byte and a v2 message
(`javascript/rlmesh/wasm/src/frame.rs` has the table); text frames are the
gateway's JSON (hello, subscribe, errors, fence reports).

## The browser capability profile

The binding is a client of a session, not a full participant runtime:

- **It can** observe (subscribe, `latest`, `peek`), hold leases and command
  commandable parts, arm, disarm and e-stop the session, and estimate its
  clock.
- **It cannot** serve environments, declare robot parts, run an edge for
  hardware, publish samples, or bind a policy map (the gateway's binding is
  an identity binding over the parts; one member of a command group is
  leased at a time). A robot's edge must run next to the robot, where a
  process stall can be bounded: never in a tab.
- **It is async-first.** Every call that waits on the network returns a
  `Promise`, an exception to the parity rule R-16 (sync first): a browser
  main thread cannot block.

## When a robot or the session goes away

A robot that detaches takes its parts with it: the page gets a `robot`
event (`{robot, present: false}`), its leases end (`cause: 'robot_left'`)
and its subscriptions close; when it declares again, `robot` reports it
back. When the session's host is gone for 2 s, the session has ended: the
gateway closes every socket with `4410` (reason `host_lost: ...`), the page gets one `ended` event
(its leases end with `cause: 'session_ended'`), and `rlmesh-gateway`, which
serves one session, exits. The gateway takes the session as its host
prints it, `--address {ip}:{port}[/{session}]`, or as `--connect`
with `--session`/`--session-file`, and joins the host as a client, as
every participant does (through the password gate of a loopback host, and
without one to rlmeshd).

## Security model

`rlmesh-gateway` exposes a robot session to web pages, so it is locked down
by default (`crates/rlmesh-session/src/gateway/security.rs`):

- **Bind:** `127.0.0.1` by default. A non-loopback `--bind` refuses to
  start without TLS (a token would cross the network in the clear) unless
  `--insecure-http` says so. A browser on another machine runs a gateway
  of its own, on its own loopback, rather than
  reaching the robot's.
- **Token:** every socket needs one: there is no tokenless loopback, and
  the gateway refuses to start with no token to accept. The client offers
  the WebSocket subprotocols `rlmesh.v2, rlmesh.token.<token>`; the gateway
  selects `rlmesh.v2`, never echoes or logs the token, and keeps only
  hashes of tokens. A subprotocol rather than a `?token=` query: URLs end
  up in access logs, proxy logs, tracing spans and resource-timing entries,
  the `Sec-WebSocket-Protocol` header normally does not. A token is one of:
  - **this machine's page token**, from `RLMESH_GATEWAY_TOKEN` or
    `--token-file` (never a command-line value: argv is world-readable),
    RFC 7230 token characters, at least 16 long. Whoever starts the
    gateway mints it and opens the page as `http://127.0.0.1:P/#t=…`: the
    fragment never reaches a server or its logs, and `connect()` moves it
    to `sessionStorage` and out of the URL. Its browsers attach with the
    gateway's own credential (the session address's key: device `local`
    on the robot's machine, or this machine's pairing), at
    `--max-priority` at most;
  - **a paired browser token**: `rlmesh pair HOST --browser [--name NAME]`
    on the machine whose gateway serves the browser pairs it with the
    robot's rlmeshd as machines pair (design §3, §5): it shows a code, the
    owner types `rlmesh pair approve CODE [--max CLASS]` on the robot, and
    rlmeshd enrolls a device of `kind = "browser"` (default cap `model`,
    or `observer`), with its own `parts` and `verbs` (`rlmesh devices
allow`). This machine keeps the device's key and the token's hash in
    `~/.config/rlmesh/browsers.toml` (`--browsers`), which the gateway
    reads at every upgrade, and the command prints the page URL with the
    token once. The gateway attaches that browser under the device's key,
    so the host applies its cap, parts and verbs, and `rlmesh devices
revoke` detaches exactly that browser.
- **Origin:** an allowlist, by default the origins the gateway serves its
  own page from, computed from the bind address and port, never from the
  request's `Host` (so DNS rebinding, `evil.example` resolving to
  127.0.0.1, does not pass). `--allow-origin` adds origins and `'*'` allows
  any. A request without an `Origin` is not a browser page (Node's
  `WebSocket` sends none); the token still applies to it.
- **What a connection may do:** at most `--max-message-bytes` per inbound
  message and frame (64 KiB) and `--max-subscriptions` per data connection
  (32; duplicates ignored). Its participant attaches at most at
  `--max-priority` (`model` by default) and its token's device's cap.
  Heartbeats and chunks must name the connection's own
  participant, and only `GrantLease`, `ReleaseLease`, `Arm`, `Disarm` and
  `Estop` are forwarded (declaring parts, bindings or acknowledgements is
  refused).
- **Refusals carry their reason.** A browser cannot read the status of a
  refused WebSocket upgrade, so the gateway completes a refused upgrade and
  closes it at once, before any route sees it: `4401` for the token, `4403`
  for the origin. A participant the host refuses closes its socket the same
  way: `4401` for a key it doesn't know (`UNAUTHENTICATED`: a revoked
  token), `4403` above the device's cap (`NOT_AUTHORIZED`). `connect()`
  rejects with that reason instead of waiting out its timeout, and the
  socket does not retry.
- **TLS:** `--tls-cert` and `--tls-key` serve https and wss with rustls (the
  ring provider) on the same port; handshakes run off the accept loop with a
  10 s timeout. A page is cross-origin isolated (COOP/COEP/CORP are always
  sent), which enables `SharedArrayBuffer` for `latest()`, only in a secure
  context: loopback http qualifies, any other host needs TLS.

## Running it

```sh
mise run demo:web     # the example robot and the gateway; open the http://127.0.0.1:18080/#t=… it prints
mise run test:js      # wasm tests under Node, the .wasm checks, strict tsc, JS unit tests
mise run test:js:e2e  # Node recovery tests against the example robot and a real gateway
mise run check:sans-io
```

The tools (wasm-bindgen 0.2.129, which must equal the crate pin, and
binaryen's wasm-opt) are scoped in `javascript/mise.toml` with their own
lock, like the C# tools: `setup:js` installs them, the wasm32 Rust target
and the npm dev dependencies, and no other check needs them. The CI
`browser` job runs the browser gates only when the wasm, gateway,
session-core, wire or JS inputs change.

The sans-IO crate (`rlmesh-session-core`: the wire, the spec and the state
machines) carries its own `clippy.toml` that bans `Instant::now`,
`SystemTime::now`, `thread::spawn` and `thread::sleep` (clippy does not merge
config files, so it repeats the workspace settings). `check:sans-io` builds it for wasm32-unknown-unknown,
runs clippy with those bans denied, and refuses an IO dependency (tokio, mio,
socket2, zenoh, tonic, hyper, async-std, getrandom, wasm-bindgen, js-sys,
web-sys, libc) on the wasm target.
