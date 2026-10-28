# rlmesh-cockpit

The cockpit: a native (egui) viewer for one live RLMesh session (protocol v2). It joins as an observer, never a controller, so it adds nothing to the hot loop and can attach late or to a remote robot.

This crate is an internal component of `rlmesh`. The Rust API exposed here is unstable and will have frequent breaking changes.

## Run it

Start a session, for example the UR10e demo with its host in its own terminal:

```sh
uv run python examples/python/v2/ur10e_mujoco/demo.py robot --seconds 120 --camera off
# session: 127.0.0.1:41000/local/01a15b0a-...      (printed on stderr)
uv run python examples/python/v2/ur10e_mujoco/demo.py policy --address ADDR
```

Then open the cockpit on the printed address:

```sh
mise run cockpit -- 127.0.0.1:41000/local/01a15b0a-...
# or: cargo run -p rlmesh-cockpit -- ADDRESS [--id cockpit] [--timeout 10] [--record-dir DIR]
```

`RUST_LOG=rlmesh_cockpit=debug` logs the feed.

The header says whether the host records the session (`REC`, or "recording paused: disk low"). **Pause** scrubs the last minute (up to 512 MB of samples) while the session keeps running underneath; only E-STOP stays on until **Back to live**. **Keep** keeps the host's recording of the last 2 minutes as an incident; it needs the host on this machine, recording into `--record-dir` (default: the host's default).

## Replay a recording

```sh
mise run cockpit -- --replay session.mcap   # the demo's --record file, or any rlmesh Recorder output
mise run cockpit -- --replay ~/.local/share/rlmesh/recordings/rolling/SESSION   # what `rlmesh host` recorded
mise run cockpit -- --replay ~/.local/share/rlmesh/recordings/incidents/SESSION/INCIDENT   # what it kept
```

The same views over a recorded session, with play/pause, speed, previous/next message and a scrub bar instead of the verbs. Scrubbing shows exactly what playing straight through shows (checkpoints every 10 s of session time).

`rlmesh host` keeps an incident by itself on an e-stop, safe mode, a lease preemption or 5 missed edge deadlines within 10 s: the 2 minutes before and 30 s after, under `incidents/SESSION/UNIXMS-LABEL/`, for 14 days or 10 GB in all.

## What it shows

- **Live channels** (left): the robot path tree, each part's kind, latest value (numbers with a 10 s plot, or a camera thumbnail), age since capture and seq gaps. Commandable parts list their `_applied` stream: what the edge dispatched on each tick.
- **Declared vs measured** (§8.10), each counter kept separate and labelled with its clock:
  - per stream: declared source rate; produced and published (the publisher's own count, from its observer reports); the admitted delivery contracts and the publication plan; unique received here; per second of capture time; each subscriber's seq advance (from its heartbeats); age, freshness bound and losses; `rate_drift` outside `[0.8, 1.25]`.
  - per controller: starts, completions and busy skips per second, other skipped slots by reason (`waiting_for`, `stale_input`, `window_gap`, `no_lease`), inference pending, latency p50/p99, chunks per second, chunk age at receipt and valid coverage remaining (an observer's estimate, §9.2).
  - per edge: declared and measured control rate (the edge's own tick stamps), deadline misses (a tick more than two control periods late, §9.1), longest tick gap, edge mode, lease holder, fenced chunks, queue depth, hold and safe behaviour.
  - per participant: clock offset, error bound and RTT, health, heartbeat age.
- **Lease and mode timeline**: lanes of session mode, lease holder and edge mode, and every change derived from consecutive authority snapshots: epochs, mode transitions, leases granted, preempted, rebound, expired, revoked or released, e-stop, bindings and roster.
- **Bindings**: each controller's binding table in the §6.8 layout (entries, input contracts, action grids).
- **Arm / e-stop**: E-STOP is one click and always visible. Arm asks for confirmation. Disarm is offered whenever the protocol allows it. A latched e-stop is cleared at the robot (a local clear, §11.1); the cockpit says so and then offers disarm and arm.

In a sim slower than real time, "per second" means two things: rates on the cockpit's or publisher's clock are per wall second, "per capture s" is per second of capture time, and in virtual and lockstep modes the header shows session seconds per cockpit second.

## Headless checks

The state is a sans-IO model (`Model::ingest`, `Model::view` with an injected `now_ns`); the UI renders a `View`. `cargo test -p rlmesh-cockpit` runs the model tests and drives the real widgets with egui_kittest, rendering every tab offscreen with wgpu (lavapipe without a GPU) into `target/tmp/cockpit/` (or `$RLMESH_COCKPIT_SNAPSHOTS`). Against a live session:

```sh
RLMESH_COCKPIT_ADDRESS=ADDR RLMESH_COCKPIT_SNAPSHOTS=/tmp/cockpit \
  cargo test -p rlmesh-cockpit --test render -- --ignored live_session
```

## License

Licensed under either of Apache License, Version 2.0 or the MIT license, at your option. See [LICENSE-APACHE](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-APACHE) and [LICENSE-MIT](https://github.com/ArenaX-Labs/rlmesh/blob/main/LICENSE-MIT).
