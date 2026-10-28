# The Node binding

Status: Experimental (protocol v2). The `rlmesh` npm package
(`javascript/rlmesh`) ships with each release; the API may change.

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

## Managed platform

`rlmesh/managed` is the managed platform's public HTTP API as an SDK, in Node
and in browsers alike: plain `fetch`, no wasm and no native addon, so
importing it loads neither. Its functions, types and fetch client are
generated by [Hey API](https://heyapi.dev) from the platform's OpenAPI
contract (`src/managed/gen`), one function per operation, named by its
`operationId`, taking `{ client, path, query, body }`:

```js
import {
  createClient,
  createEvaluation,
  getEvaluation,
  getEvaluationScores,
  listEvaluations,
} from "rlmesh/managed";

const client = createClient({ apiKey }); // or { getAccessToken }; in Node, optional
const { data, error, response } = await createEvaluation({ client, body: request });
if (error) throw error; // a ManagedError: status, code, message, requestId
const evaluation = (await getEvaluation({ client, path: { id: data.id } })).data;
const scores = await getEvaluationScores({
  client,
  path: { id: data.id },
  query: { groupBy: "task" },
});
const page = await listEvaluations({ client, query: { tag: ["team:vla"], limit: 50 } });
```

Without `client`, a function uses the package's default client. Each call
resolves to `{ data, error, response }`; pass `throwOnError: true` to have it
throw instead. Either way a non-2xx answer is a `ManagedError` with the HTTP
`status`, the platform's `code` and `message`, `error` (the body's
`{ code, message, details }`) and `requestId`. `streamEvaluationEvents`
returns `{ stream }`, an async iterator of parsed frames
(`EvaluationStreamFrame`, the first an `EvaluationStreamSnapshot`) that
reconnects with backoff. Anything else Hey API's client offers
(interceptors, `client.setConfig`, `client.get` for a raw request) works as
[its docs](https://heyapi.dev/openapi-ts/clients/fetch) describe.

Credentials: `apiKey`, else `getAccessToken` (a browser session). In Node an
unset key falls back to `RLMESH_API_KEY`, then to the signed-in CLI
(`rlmesh token --json`, after `rlmesh login`; `profile` picks a CLI profile),
whose token is refetched once on a 401, and an unset `baseUrl` to
`RLMESH_PLATFORM_URL`, then to that profile's platform. The default client
does the same, so in Node a call with no `client` works once you are signed
in. Browsers get none of these fallbacks: pass a `client`, or set the default
one's token with `client.setConfig({ auth })`.

`mise run js:managed:generate` regenerates `src/managed/gen` from
`https://api.rlmesh.dev/openapi.yaml` (or a contract file given as its
argument), keeping only the public paths.

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
