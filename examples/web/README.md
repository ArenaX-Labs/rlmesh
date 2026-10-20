# A v2 session in the browser

A live RLMesh real-time session (protocol v2) in a web page, through
`rlmesh-gateway`, with the `rlmesh` npm package (`javascript/rlmesh`). See
[the browser binding](../../docs/browser.md) for how it works and the
gateway's security model.

```sh
mise run demo:web     # builds the wasm, the package and this page, then runs the robot and the gateway
# open http://127.0.0.1:18080/
```

`run.sh` starts two processes (logs in `~/.cache/rlmesh-web/`):

- **The example robot** (`crates/rlmesh-gateway/examples/web_robot`): a session
  host, the robot `arm7` (`rlmesh_edge::Robot`) and a model policy, in one
  process. `arm7/arm` is 7 joints (`pos` and `vel`) at 1 kHz with position
  commands (`safe: stop`, held 100 ms); `arm7/gripper` is 100 Hz; and
  `arm7/scene/cam` is a 640x480 camera at 20 Hz, JPEG q85 (`CAM=raw` sends raw
  RGB). The policy holds the arm at the model class and sways it; it takes
  the arm back whenever it is free.
- **The gateway**, serving `examples/web/dist` with COOP/COEP, so the page is
  cross-origin isolated and `latest()` reads a SharedArrayBuffer.

The page:

- draws the camera from the bulk Worker (`ImageBitmap`s) and plots the joints
  from `session.subscribe('arm7/arm')`;
- shows each stream's rate, publish → receipt latency, capture → receipt age,
  late share and losses, measured in its class Worker;
- **Take control** leases `arm7/arm` at the operator class, preempting the
  policy, and drives the sliders' targets at 100 Hz from the control Worker
  (keys Q/A … U/J nudge them). **Release** hands the arm back to the policy.
- **Pause heartbeats** stops this participant's heartbeats: the host lapses
  the lease after its grace (250 ms) and the session goes SAFE. SAFE stays
  until **Re-arm** (a fresh epoch); then take control again.
- **Stale-token cmd** sends one chunk under the previous token: the gateway
  reports it fenced, and the robot's edge refuses it (its log counts
  `StaleToken`).

Environment knobs: `ROBOT_BASE` (per-class TCP base, default 17600, four
ports), `HTTP_PORT` (18080), `CAM=jpeg|raw`, `POLICY=on|off`, `DURATION`
(seconds), `SKIP_BUILD=1`, and `BIND` (default 127.0.0.1; any other address
needs `RLMESH_GATEWAY_TOKEN`, then open `http://HOST:PORT/#token=…`).

`window.__rlmesh` exposes the hooks the automated checks use: `measure(s)`,
`take()`, `release()`, `arm()`, `setTarget([...])`, `pauseHeartbeats(p)`,
`stale()`, `trap(cls)`, `dropSocket(cls)` and a `timeline` of mode and edge
transitions.
