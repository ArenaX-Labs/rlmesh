# The Node binding

Status: Experimental (protocol v2). The `rlmesh` npm package
(`javascript/rlmesh`) is not published yet; the API may change.

In Node 22+ and Electron's main process, `rlmesh` runs real-time sessions
in-process: a session host, robots with their edges, and bound policies, on
the session's own transport (Zenoh, one session per traffic class). It is
Python's `rlmesh.v2`, one to one, over a Node-API addon
(`javascript/rlmesh/native`, napi-rs) that binds `rlmesh-session`
(its edge included) directly, as PyO3 does. Every rule (normalisation, binding,
conversions, leases, the edge, the run loop) runs in Rust.

A page cannot do this: it has no raw TCP or UDP. Browsers join through
`rlmesh-gateway` instead ([the browser binding](browser.md)); in Node the
same package exports both, `connect` for a gateway and the classes below for
the session itself.

`examples/node/hold.mjs` runs it end to end (`node examples/node/hold.mjs`
after `mise run build:js:native`): the conformance UR10e, faked, and a policy
that holds the arm. Its core:

```js
import { Host, Policy, Rate, Robot } from "rlmesh";

const host = await Host.open();
const robot = await Robot.connect("ur10e", UR10E, { session: host });
const policy = new Policy(HOLD, { session: host, name: "hold" });
try {
  await robot.arm();
  // The robot's control loop: publish its sensors, play what its edge says is due.
  (async () => {
    const rate = new Rate(500);
    for (;;) {
      robot.publish("arm", { pos: q, vel: dq, effort });
      const due = robot.due("arm"); // { mode, value: { dtype, shape, data }, ... } or null
      if (due?.value) q = due.value.data;
      await rate.sleep();
    }
  })();
  console.log(String(await policy.bind("ur10e"))); // the binding table
  await policy.run(
    async (obs) => {
      const arm = new Float32Array(60); // [horizon 10, 6 joints]
      for (let k = 0; k < 10; k++) arm.set(obs.inputs.q.data, k * 6);
      return { arm: { data: arm, shape: [10, 6] } };
    },
    { duration: 5 },
  );
} finally {
  await policy.close();
  await robot.close();
  await host.close();
}
```

On Node 24 and later, `await using host = await Host.open()` (and the same
for `Robot`, `Policy` and `Recorder`) closes each one at the end of the
block instead of the `finally`.

## How it maps to Python

| Python (`rlmesh.v2`)                               | Node (`rlmesh`)                                               |
| -------------------------------------------------- | ------------------------------------------------------------- |
| `with Host() as host`                              | `const host = await Host.open()` … `await host.close()`       |
| `Robot(name, parts, session=host)`                 | `await Robot.connect(name, parts, { session: host })`         |
| `robot.publish("arm", pos=q, vel=dq)`              | `robot.publish('arm', { pos: q, vel: dq })`                   |
| `robot.due("arm").value` (NumPy array)             | `robot.due('arm').value` (`{ dtype, shape, data }`)           |
| `robot.arm()`, `estop(reason)`, `clear()`          | `await robot.arm()`, `estop(reason)`, `clear()`               |
| `robot.now_ns()` (`time.monotonic_ns()`)           | `robot.nowNs()` (`process.hrtime.bigint()`)                   |
| `Policy(map, session=host).bind(robot, **opts)`    | `await new Policy(map, { session: host }).bind(robot, opts)`  |
| `obs["q"]`, `obs.meta.time_ns`                     | `obs.inputs.q`, `obs.meta.timeNs`                             |
| `predict` returns `{"arm": array}`                 | `predict` returns (or resolves to) `{ arm: { data, shape } }` |
| `policy.run(predict, duration=5)`                  | `await policy.run(predict, { duration: 5 })`                  |
| `policy.acquire()` → `{"arm": token}`              | `await policy.acquire()` → `{ arm: token }` (`bigint`)        |
| `Rate(500).sleep()`                                | `await new Rate(500).sleep()`                                 |
| `with Recorder(path, session=host)`                | `const rec = await Recorder.start(path, { session: host })`   |
| `read_recording(path)`                             | `await readRecording(path)`                                   |
| `SessionRefused`, `BindingRefused`, `StartSkipped` | the same classes; `ValueError` is `RangeError`                |

The rules are R-30 in the [parity spec](specs/language-parity/rules.md).

## Threads

Nothing blocks the JS thread on the network. Calls that wait (`Host.open`,
`Robot.connect`, `bind`, `arm`, `acquire`, `close`) run on the addon's
session runtime and return a `Promise`. `publish`, `due`, `observe` and
`send` never wait. `Rate.sleep` waits on a native thread (a coarse sleep,
then a short spin) and resolves on the JS thread. A JPEG part's frames
are encoded on a thread of the robot's own (encoding a 1280x720 frame takes
about 6 ms, which on the JS thread would cost a 1 kHz loop one tick in four),
so `publish` returns `null` for them at once.

`policy.run` runs the run loop (§8.9) on a native thread of its own. At each
start it hands the observation to `predict` on the JS thread and waits for
the answer; `predict` may be `async` (a model on another thread, a GPU
call), and one inference is in flight at a time. If it throws or rejects, the
policy stops sending, releases its leases, and `run` rejects with that error.

A robot's control loop and a policy in the same process share the JS
thread: a long synchronous `predict` delays the robot's ticks. Run the model
off the thread (a `worker_threads` Worker, an async runtime), or the robot
and the policy in separate processes, as on real hardware.

## Arrays

A tensor is `{ dtype, shape, data }`, its elements flat and row-major in a
typed array of the dtype (`Float32Array` for `float32`, `BigInt64Array` for
`int64`; `float16` is a `Float16Array` where the runtime has one, else the
raw bits in a `Uint16Array`). `publish` takes typed arrays, number arrays or
a number, casts them to the declared dtype as NumPy's `same_kind` does (a
float never becomes an int), checks the element count, and copies them
before it returns. Observation arrays and `due` values are the caller's.

## Building and testing

The addon is built per platform with cargo (no `@napi-rs/cli`); the package
loads `dist/rlmesh.{platform}-{arch}.node` on first use, or the path in
`RLMESH_NATIVE`. `connect` works without it.

```sh
mise run build:js:native   # the addon, profile node (LTO, stripped)
mise run test:js:native    # the twin of test_v2_session.py, in Node
mise run bench:js:native   # the same 1 kHz robot and 50 Hz policy, in Node and in Python
```

The addon has no npm dependencies. On linux-x64 it is 9.3 MB stripped (4.1
MB gzipped). Publishing per-platform prebuilds is future work.
