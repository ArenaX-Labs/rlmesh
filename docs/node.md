# The Node binding

Status: Experimental (protocol v2). The `rlmesh` npm package
(`javascript/rlmesh`) is not on npm yet: the release workflow's npm steps are
fenced. Its pre-releases will install with `npm install rlmesh@next` (npm's
`latest` is a deprecated `0.0.1` placeholder). Until then, build from source to
experiment. CI builds the addon for linux-x64 and runs the native tests below
when the binding, its examples or the session crates change. The API may
change.

In Node 22+ and Electron's main process, `rlmesh` runs real-time sessions
in-process: a session host, robots with their edges, and bound policies, on
the session's own transport (Zenoh, one session per traffic class). It is
Python's `rlmesh.realtime`, one to one, over a Node-API addon
(`javascript/rlmesh/native`, napi-rs) that binds `rlmesh-session`
(its edge included) and `rlmesh::realtime` directly, as PyO3 does. Every rule (normalisation, binding,
conversions, leases, the edge, the run loop) runs in Rust.

A page cannot do this: it has no raw TCP or UDP. Browsers join through
`rlmesh-gateway` instead ([the browser binding](browser.md)); in Node the
same package exports both, `connect` for a gateway and the classes below for
the session itself.

`examples/node/hold.mjs` runs it end to end (`node examples/node/hold.mjs`
after `mise run build:js:native` and `node javascript/rlmesh/build.mjs
--node-only`): the conformance UR10e, faked, and a policy that holds the arm.
Its core:

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
  await policy.bind("ur10e");
  console.log(String(policy.binding)); // the binding table
  await policy.run(
    async (obs) => {
      const arm = new Float32Array(60); // [chunk 10, 6 joints]
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
for `Robot`, `Policy` and `SessionRecorder`) closes each one at the end of the
block instead of the `finally`.

`Robot`, `Policy` and `SessionRecorder` place their session by the hosting
rule, as in Python. A `Host` or its address is joined. An empty session (`''`,
a robot's default) joins the local host on `RLMESH_PORT` (7447 by default),
and a loopback `:port` joins whatever listens there. When nothing listens, the
participant hosts the session in this process until it closes. So a script
ported from Python with no session, or with `session=""`, runs the same.
`robot.address` is the robot's participant address on that host, its key
included.

## How it maps to Python

| Python (`rlmesh.realtime`)                         | Node (`rlmesh`)                                                    |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| `with Host() as host`                              | `const host = await Host.open()` … `await host.close()`            |
| `Robot(name, parts, session=host)`                 | `await Robot.connect(name, parts, { session: host })`              |
| `Robot(name, parts)` (the local session)           | `await Robot.connect(name, parts)`                                 |
| `robot.publish("arm", pos=q, vel=dq)`              | `robot.publish('arm', { pos: q, vel: dq })`                        |
| `robot.due("arm").value` (NumPy array)             | `robot.due('arm').value` (`{ dtype, shape, data }`)                |
| `robot.arm()`, `estop(reason)`, `clear_estop()`    | `await robot.arm()`, `estop(reason)`, `clearEstop()`               |
| `robot.now_ns()` (`time.monotonic_ns()`)           | `robot.nowNs()` (`process.hrtime.bigint()`)                        |
| `Policy(map, session=host).bind(robot, **opts)`    | `await new Policy(map, { session: host }).bind(robot, opts)`       |
| `policy.bind("sim").run(predict, episodes=20)`     | `await (await policy.bind('sim')).run(predict, { episodes: 20 })`  |
| `policy.binding` (the table bound)                 | `policy.binding`                                                   |
| `obs["q"]`, `obs.meta.time_ns`                     | `obs.inputs.q`, `obs.meta.timeNs`                                  |
| `predict` returns `{"arm": array}`                 | `predict` returns (or resolves to) `{ arm: { data, shape } }`      |
| `policy.run(predict, duration=5)`                  | `await policy.run(predict, { duration: 5 })`                       |
| `policy.run(predict, mode=RunMode.SHADOW)`         | `await policy.run(predict, { mode: 'shadow' })`                    |
| `policy.run(predict, resume=Resume.AUTO)`          | `await policy.run(predict, { resume: 'auto' })`                    |
| `policy.clock` (`Clock.LOCKSTEP`)                  | `policy.clock` (`'lockstep'`)                                      |
| `policy.run(predict, episodes=20, seed=7)`         | `await policy.run(predict, { episodes: 20, seed: 7 })`             |
| `stats["episodes"][0]["return"]`                   | `stats.episodes[0].return`                                         |
| `policy.resume()` → `["ur10e:arm"]`                | `policy.resume()` → `['ur10e:arm']`                                |
| `policy.health(wait=True)["links"]["ur10e"]`       | `(await policy.health({ wait: true })).links['ur10e']`             |
| `on_step=f` (`step.shadow`, `step.actions`)        | `onStep: f` (`step.shadow`, `step.actions`)                        |
| `policy.acquire()` → `{"ur10e:arm": token}`        | `await policy.acquire()` → `{ 'ur10e:arm': token }` (`bigint`)     |
| `Rate(500).sleep()`                                | `await new Rate(500).sleep()`                                      |
| `with SessionRecorder(path, session=host)`         | `const rec = await SessionRecorder.start(path, { session: host })` |
| `read_recording(path)`                             | `await readRecording(path)`                                        |
| `SessionRefused`, `BindingRefused`, `StartSkipped` | the same classes; `ValueError` is `RangeError`                     |

The rules are R-30 in the [parity spec](specs/language-parity/rules.md).

### Not yet in Node

These parts of `rlmesh.realtime` have no Node counterpart yet:

- **Serving a lockstep robot.** A Node `Robot` runs on the wall clock;
  `Robot(..., clock=Clock.LOCKSTEP)` with `serve` and `serve_lockstep` is
  Python's. A Node policy runs a lockstep robot served elsewhere (below).
- **Builders.** Python's `parts`, `inputs` and `actions` modules
  (`parts.joints(...)`) return the plain dicts; Node takes those dicts as
  written.
- **Holds and hand-backs.** `robot.rearm(name)`, `robot.held` and
  `policy.held`, `policy.release(to="model")`, and `robot.part(path)` with
  `robot.command(path)`. A held robot is re-armed from Python or the cockpit.
- **A timeout per bind.** Python's `policy.bind(..., timeout=)`; in Node,
  `bind` waits for the `Policy`'s own `timeout`.
- **A two-argument `predict`.** Python's `predict(obs, policy)`; in Node,
  `predict` takes the observation only. Close over the policy instead.

## Lockstep robots

A policy follows its robot's clock, as in Python ([one policy on both
clocks](clocks.md#one-policy-on-both-clocks)): `bind` learns it from what the
robot declared on joining (`policy.clock`, `'wall'` or `'lockstep'`), and
`run` follows it. A lockstep robot (a Gym env served with Python's
`rlmesh.EnvServer`, or a Python `Robot(..., clock=Clock.LOCKSTEP)`) runs by
episodes, each from a reset seeded from `seed` as a model run seeds it, and
an env that declared the `trial_index` reset option gets each episode's
ordinal in it:

```js
const policy = new Policy(BALANCE, { session: "127.0.0.1:5555", name: "balance" });
await policy.bind("cartpole");
const stats = await policy.run(lean, { episodes: 10, seed: 7 }); // stats.clock === 'lockstep'
for (const episode of stats.episodes) console.log(episode.return, episode.steps);
```

`duration`, `mode: 'shadow'` and `resume` throw a `RangeError` on a lockstep
robot, as `episodes` and `seed` do on the wall clock; `run` and `stats`
resolve to `LockstepStats` or `PolicyStats`, told apart by `clock`. A Node
`Robot` runs on the wall clock: serving a lockstep robot from Node is future
work.

## Threads

Nothing blocks the JS thread on the network. Calls that wait (`Host.open`,
`Robot.connect`, `bind`, `arm`, `acquire`, `close`) run on the addon's
session runtime and return a `Promise`. `publish`, `due`, `observe` and
`send` never wait. `Rate.sleep` waits on a native thread (a coarse sleep,
then a short spin) and resolves on the JS thread. A JPEG part's frames
are encoded on a thread of the robot's own (encoding a 1280x720 frame takes
about 6 ms, which on the JS thread would cost a 1 kHz loop one tick in four),
so `publish` returns `null` for them at once.

`policy.run` runs the run loop on a native thread of its own. At each
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
argument), keeping only the public paths and fields and the first sentence of
each description. It fails, changing nothing, when wording about how the
platform runs evaluations would reach the generated code.

## Building and testing

The addon is built per platform with cargo (no `@napi-rs/cli`). On first use
the package loads the path in `RLMESH_NATIVE` if set, else
`dist/rlmesh.{platform}-{arch}.node` (a local build), else the
`@rlmesh/{platform}-{arch}` package npm installs as an optional dependency.
It refuses an addon whose version is not the package's, such as a stale local
build. `connect` works without it.

```sh
mise run build:js:native   # the addon, profile node (LTO, stripped)
mise run test:js:native    # the twin of test_v2_session.py, in Node (the lockstep test runs a Python env through uv)
mise run bench:js:native   # the same 1 kHz robot and 50 Hz policy, in Node and in Python
```

`test:js:native` bundles `dist/node.js` alone (`build.mjs --node-only`), so it
needs no wasm toolchain.

The addon has no npm dependencies. On linux-x64 it is 15.3 MB stripped (6.8
MB gzipped). The release workflow builds one per platform (`mise run
release:js:native`: linux-x64 and linux-arm64 against glibc 2.28, win32-x64,
and darwin-x64 and darwin-arm64 on macOS) and packs each as
`@rlmesh/{platform}-{arch}` (`mise run release:js:package`). `rlmesh` names
them all as optional dependencies at its own version, and npm installs the
one for your platform.
