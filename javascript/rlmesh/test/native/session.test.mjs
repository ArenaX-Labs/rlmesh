// The native binding on loopback: a Node robot and Node policies in one session, the twin of
// python/rlmesh/tests/integration/test_v2_session.py. The robot is the §14.1 UR10e (the conformance
// fixture), faked: its control loop publishes joints, wrench, gripper and two cameras, and applies
// whatever its edge says is due. The policies bind to it by kind and path, lease its parts and
// stream timed chunks. Needs dist/node.js and the addon (`mise run test:js:native`).
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { inspect } from 'node:util';
import { afterEach, beforeEach, test } from 'node:test';
import { setTimeout as sleep } from 'node:timers/promises';
import * as rlmesh from '../../dist/node.js';

const SPEC = new URL('../../../../crates/rlmesh-session-core/tests/', import.meta.url);
const WAIT_MS = 10_000;

const UR_JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'];
const HOME = Float32Array.of(0, -1.57, 1.57, -1.57, -1.57, 0);

/** A small sine around the measured pose on the arm, the gripper half open. */
const SINE = {
  hz: 50,
  inputs: {
    q: { kind: 'joints', at: 'arm', fields: ['pos'], names: UR_JOINTS },
    ft: { kind: 'wrench', at: 'wrist/ft', frame: 'tool0' },
    wrist: { kind: 'camera', at: 'wrist/cam', size: [96, 96] },
    prompt: { kind: 'text', at: '/task' },
  },
  actions: {
    arm: { kind: 'joints', at: 'arm', command: 'position', names: UR_JOINTS, hz: 100, horizon: 10 },
    gripper: { kind: 'gripper', at: 'gripper', command: 'position', hz: 100, horizon: 10 },
  },
  allow: ['resize', 'crop'],
};

/** Holds one target on the arm. */
const HOLD = {
  hz: 50,
  inputs: { q: { kind: 'joints', at: 'arm', fields: ['pos'], names: UR_JOINTS } },
  actions: { arm: { kind: 'joints', at: 'arm', command: 'position', names: UR_JOINTS, hz: 100, horizon: 10 } },
};

const fixture = (kind, name) => JSON.parse(readFileSync(new URL(`fixtures/${kind}/${name}.json`, SPEC), 'utf8'));
const golden = (name) => readFileSync(new URL(`golden/${name}.txt`, SPEC), 'utf8').replace(/\n+$/, '');

/** Polls `done` every 10 ms; `context()` goes into the timeout's message. */
async function waitUntil(what, done, context = () => '') {
  const deadline = Date.now() + WAIT_MS;
  while (!done()) {
    assert.ok(Date.now() < deadline, `timed out waiting for ${what} ${context()}`);
    await sleep(10);
  }
}

/** Why a start now would be skipped, for a timeout's message. */
function skipped(policy) {
  try {
    policy.observe();
    return null;
  } catch (e) {
    return { reason: e.reason, waiting: e.waiting, stale: e.stale, gaps: e.gaps };
  }
}

/** `[horizon, 6]` of one value. */
const hold = (target, horizon = 10) => ({ arm: { data: new Float32Array(horizon * 6).fill(target), shape: [horizon, 6] } });
const first = (due) => due.value.data[0];

/** The §14.1 UR10e without physics: commanded joints become measured ones. */
class FakeUR10e {
  q = HOME.slice();
  gripper = 1;
  log = [];
  stopped = false;

  static async start(host, name = 'ur10e') {
    const ur = new FakeUR10e();
    ur.robot = await rlmesh.Robot.connect(name, fixture('robots', 'ur10e'), { session: host });
    await ur.robot.arm();
    ur.robot.publish('task', 'wave the arm');
    ur.loops = [ur.control(), ur.cameras()];
    return ur;
  }

  async control() {
    const rate = new rlmesh.Rate(500);
    const zeros = new Float32Array(6);
    while (!this.stopped) {
      this.robot.publish('arm', { pos: this.q, vel: zeros, effort: zeros });
      this.robot.publish('wrist/ft', { force: [0, 0, 9.8], torque: [0, 0, 0] });
      this.robot.publish('gripper', [this.gripper]);
      const due = this.robot.due('arm');
      if (due) {
        if (due.value) this.q = Float32Array.from(due.value.data);
        this.log.push(due);
      }
      const gripper = this.robot.due('gripper');
      if (gripper?.value) this.gripper = gripper.value.data[0];
      await rate.sleep();
    }
  }

  async cameras() {
    const wrist = new Uint8Array(480 * 640 * 3);
    const scene = new Uint8Array(720 * 1280 * 3);
    const rate = new rlmesh.Rate(10);
    for (let frame = 0; !this.stopped; frame++) {
      for (let i = 0; i < wrist.length; i += 3) wrist[i] = frame % 255;
      this.robot.publish('wrist/cam', wrist);
      this.robot.publish('scene/cam', scene);
      // Again with every frame: under load a policy that subscribes late can miss a task published
      // only once at start (it then waits on `prompt` for good).
      this.robot.publish('task', 'wave the arm');
      await rate.sleep();
    }
  }

  driven(predicate, n = 5) {
    const recent = this.log.slice(-n);
    return recent.length === n && recent.every((d) => d.mode === 'driving' && predicate(d));
  }

  async [Symbol.asyncDispose]() {
    this.stopped = true;
    await Promise.all(this.loops);
    await this.robot.close();
  }
}

let host;
beforeEach(async () => {
  host = await rlmesh.Host.open();
});
afterEach(async () => {
  await host.close();
});

test('a Node policy drives a Node robot on loopback', async () => {
  const seen = [];
  await using ur = await FakeUR10e.start(host);
  await using policy = new rlmesh.Policy(SINE, { session: host, name: 'sine' });
  const binding = await policy.bind('ur10e');
  assert.match(String(binding), /wrist +in +wrist\/cam +bound +decode\(jpeg\) → crop\(640x480 → 96x96, bilinear\)/);
  assert.equal(binding.json.actions[0].horizon, 10);
  assert.equal(binding.fingerprint.length, 16);

  // An async predict: the run loop waits for it off this thread.
  const running = policy.run(
    async (obs) => {
      seen.push(obs);
      const t0 = Number(obs.meta.timeNs) / 1e9;
      const arm = new Float32Array(60);
      for (let k = 0; k < 10; k++) for (let j = 0; j < 6; j++) arm[k * 6 + j] = HOME[j] + 0.1 * Math.sin(2 * Math.PI * (t0 + k * 0.01));
      return { arm: { data: arm, shape: [10, 6] }, gripper: { data: new Float64Array(10).fill(0.5), shape: [10, 1] } };
    },
    { duration: 30 },
  );
  const state = () => JSON.stringify({ stats: policy.stats(), skip: skipped(policy), recent: ur.log.slice(-5).map((d) => [d.mode, d.value && first(d)]) });
  await waitUntil('the arm driven by the sine', () => ur.driven((d) => Math.abs(first(d)) <= 0.1001), state);
  await waitUntil('the gripper commanded', () => Math.abs(ur.gripper - 0.5) < 1e-6, state);
  policy.stop();
  const stats = await running;
  assert.ok(stats.completions > 0 && stats.chunksSent > 0, JSON.stringify(stats));
  assert.equal(stats.sendErrors, 0);
  assert.equal(policy.holds, false, 'a stopped run releases its leases');

  // predict saw what the binding table promises.
  const obs = seen.at(-1);
  assert.deepEqual(Object.keys(obs.inputs).sort(), ['ft', 'prompt', 'q', 'wrist']);
  assert.deepEqual(obs.inputs.q.shape, [6]);
  assert.equal(obs.inputs.q.dtype, 'float32');
  assert.ok(obs.inputs.q.data instanceof Float32Array);
  assert.deepEqual(obs.inputs.ft.shape, [6]);
  assert.deepEqual(obs.inputs.wrist.shape, [96, 96, 3]);
  assert.ok(obs.inputs.wrist.data instanceof Uint8Array);
  assert.equal(obs.inputs.prompt, 'wave the arm');
  assert.deepEqual(Object.keys(obs.meta.agesNs).sort(), ['ft', 'prompt', 'q', 'wrist']);
  assert.equal(obs.meta.stale.size, 0);
  assert.equal(typeof obs.meta.timeNs, 'bigint');

  // The robot applied the policy's chunks, and knows what they came from.
  const driving = ur.log.filter((d) => d.mode === 'driving');
  assert.ok(driving.length && driving.every((d) => d.path === 'arm'));
  assert.ok(driving.at(-1).basedOn.some(([path]) => path === 'arm'));
  const edge = ur.robot.edgeStats('arm');
  assert.ok(edge.accepted > 0);
  assert.equal(edge.rejected, 0, JSON.stringify(edge.rejectedBy));
  assert.deepEqual(edge.modes.safe, [0, 0], 'a released lease never latches safe');
});

test('the binding table and refusals match the conformance goldens', async () => {
  await using robot = await rlmesh.Robot.connect('ur10e', fixture('robots', 'ur10e'), { session: host });
  await using rtde = await rlmesh.Robot.connect('ur10e_rtde', fixture('robots', 'ur10e_rtde'), { session: host });
  assert.equal(robot.parts['wrist/cam'], 'camera');
  assert.equal(rtde.name, 'ur10e_rtde');
  {
    await using pi0 = new rlmesh.Policy(fixture('policies', 'pi0_single'), { session: host, name: 'pi0_single' });
    const binding = await pi0.bind('ur10e');
    assert.equal(binding.table.replace(/\n+$/, ''), golden('pi0_single__ur10e'));
    assert.ok(binding.warnings.length && binding.warnings[0].includes('gripper'));
  }

  const refused = new rlmesh.Policy(fixture('policies', 'force_policy'), { session: host, name: 'force_policy' });
  const err = await refused.bind('ur10e_rtde').then(
    () => assert.fail('bound'),
    (e) => e,
  );
  assert.ok(err instanceof rlmesh.BindingRefused, String(err));
  assert.ok(err instanceof rlmesh.SessionException);
  assert.equal(err.message.replace(/\n+$/, ''), golden('force_policy__ur10e_rtde__refused'));
  assert.deepEqual(err.problems, [['incompatible', 'input', 'ft']]);
  // The fix the message suggests is a different robot: the same policy binds where the wrench is in tool0.
  assert.match(String(await refused.bind('ur10e')), /wrist\/ft/);
  await refused.close();

  await assert.rejects(
    new rlmesh.Policy(HOLD, { session: host, name: 'typo' }).bind('ur10e', { allow: ['resise'] }),
    (e) => e instanceof rlmesh.BindingRefused && /unknown conversion/.test(e.message),
  );
});

test('an operator preempts the model and hands back under a fresh token', async () => {
  await using ur = await FakeUR10e.start(host);
  await using model = new rlmesh.Policy(HOLD, { session: host, name: 'model' });
  await using operator = new rlmesh.Policy(HOLD, { session: host, name: 'teleop', priority: 'operator' });
  await model.bind('ur10e');
  await operator.bind('ur10e');
  const running = model.run(() => hold(0.25), { duration: 30 });
  const state = () => JSON.stringify({ model: model.stats(), recent: ur.log.slice(-5).map((d) => [d.mode, d.value && first(d)]) });
  await waitUntil('the model driving', () => ur.driven((d) => first(d) === 0.25), state);
  const modelToken = ur.log.at(-1).leaseToken;
  assert.equal(typeof modelToken, 'bigint');

  // Taking over: the operator class preempts the model class.
  const tokens = await operator.acquire();
  assert.ok(tokens.arm > modelToken);
  await waitUntil("the model's lease gone", () => !model.holds);
  let streaming = true;
  const teleop = (async () => {
    while (streaming) {
      try {
        const sent = operator.send(hold(0.75), operator.observe());
        assert.notEqual(sent.arm, null);
      } catch (e) {
        // A start can be skipped under load (an input not yet fresh): try again next tick.
        if (!(e instanceof rlmesh.StartSkipped)) throw e;
      }
      await sleep(10);
    }
  })();
  // A failure below stops the stream before the operator is disposed, so it is the one reported.
  await using _stopTeleop = { [Symbol.asyncDispose]: () => ((streaming = false), teleop) };
  await waitUntil('the operator driving', () => ur.driven((d) => first(d) === 0.75));
  assert.equal(ur.log.at(-1).leaseToken, tokens.arm);
  assert.ok(model.stats().skipped.no_lease > 0, 'a preempted model stays warm, skips its starts');

  // A model cannot take it back while the operator holds it.
  await assert.rejects(model.acquire(), (e) => e instanceof rlmesh.SessionRefused && e.status === 'LEASE_BUSY');

  // Handing back: the model re-requests and resumes under a fresh token.
  streaming = false;
  await teleop;
  await operator.release();
  await waitUntil('the model driving again', () => ur.driven((d) => first(d) === 0.25));
  const resumed = ur.log.at(-1).leaseToken;
  assert.ok(resumed !== null && resumed > tokens.arm);
  model.stop();
  assert.equal(typeof (await running).starts, 'number');
});

test('a recorder writes the session to MCAP and reads it back', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'rlmesh-record-'));
  try {
    const path = join(dir, 'session.mcap');
    const recorder = await rlmesh.Recorder.start(path, { session: host });
    {
      await using ur = await FakeUR10e.start(host);
      await using model = new rlmesh.Policy(HOLD, { session: host, name: 'model' });
      await model.bind('ur10e');
      const running = model.run(() => hold(0.25), { duration: 30 });
      await waitUntil('the model driving', () => ur.driven(() => true, 50));
      model.stop();
      const stats = await running;
      assert.equal(stats.alignedTo, 'q', 'HOLD reads only the joints: its starts are phased after them');
      assert.equal(typeof stats.phaseOffsetMs, 'number');
    }
    const stats = await recorder.close();
    assert.ok(stats && stats.messages > 0);
    assert.deepEqual(await recorder.close(), stats, 'closing twice keeps the stats');

    const summary = await rlmesh.readRecording(path);
    assert.equal(summary.messages, stats.messages);
    const topics = Object.fromEntries(summary.topics.map((t) => [t.topic, t]));
    for (const topic of ['/ur10e/arm', '/ur10e/arm/_applied', '/ur10e/arm/_chunks', '/ur10e/wrist/cam', '/_session/authority', '/_session/bindings']) {
      const row = topics[topic];
      assert.ok(row?.messages > 0, topic);
      assert.equal(row.undecodable, 0, topic);
      assert.ok(row.logTimeMonotonic && row.publishTimeMonotonic, topic);
    }
    assert.equal(topics['/ur10e/arm/_chunks'].schema, 'rlmesh.session.v2.ActionChunk');
    assert.ok(summary.metadata.includes('rlmesh.robot.ur10e'));
    const arm = summary.leases.filter((l) => l.part === 'ur10e/arm');
    assert.equal(arm[0].holder, 'model');
    assert.equal(arm[0].priority, 'model');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('a throwing predict releases the leases and the run rejects with its error', async () => {
  let calls = 0;
  await using _ur = await FakeUR10e.start(host);
  await using policy = new rlmesh.Policy(HOLD, { session: host, name: 'crashy' });
  await policy.bind('ur10e');
  const crash = new Error('the model crashed');
  await assert.rejects(
    policy.run(
      (obs) => {
        if (++calls === 3) throw crash;
        const arm = new Float32Array(60);
        for (let k = 0; k < 10; k++) arm.set(obs.inputs.q.data, k * 6);
        return { arm: { data: arm, shape: [10, 6] } };
      },
      { duration: WAIT_MS / 1000 },
    ),
    (e) => e === crash,
  );
  assert.equal(policy.holds, false);
  // A wrong shape is refused before anything is sent.
  assert.throws(() => policy.send({ arm: { data: new Float32Array(6), shape: [6] } }, policy.observe()), (e) => e instanceof RangeError && /expected \[10, 6\]/.test(e.message));
  assert.throws(() => policy.send({ legs: { data: new Float32Array(60), shape: [10, 6] } }, policy.observe()), /not an action/);
});

test('publish checks fields, dtypes and sizes', async () => {
  await using robot = await rlmesh.Robot.connect('ur10e', fixture('robots', 'ur10e'), { session: host });
  assert.equal(robot.parts['wrist/cam'], 'camera');
  robot.publish('arm', { pos: HOME, vel: new Float64Array(6), effort: new Float64Array(6) });
  assert.notEqual(robot.publish('wrist/ft', { force: [1, 2, 3], torque: [0, 0, 0] }), null);
  assert.throws(() => robot.publish('arm', { pos: HOME }), (e) => e instanceof TypeError && /missing/.test(e.message));
  assert.throws(() => robot.publish('arm', HOME), (e) => e instanceof TypeError && /publish\('arm', \{ pos: \.\.\. \}\)/.test(e.message));
  assert.throws(() => robot.publish('gripper', new Float32Array(7)), (e) => e instanceof RangeError && /7 elements/.test(e.message));
  assert.throws(() => robot.publish('wrist/cam', new Float64Array(480 * 640 * 3)), (e) => e instanceof TypeError && /cannot publish float64 as uint8/.test(e.message));
  assert.throws(() => robot.publish('task', 3), (e) => e instanceof TypeError && /publish a string/.test(e.message));
  assert.throws(() => robot.publish('elbow', 1), (e) => e instanceof RangeError && /no part 'elbow'/.test(e.message));
  assert.equal(robot.due('arm'), null, 'nothing is leased: nothing is due');
  assert.throws(() => robot.due('wrist/ft'), (e) => e instanceof RangeError && /not a commandable part/.test(e.message));
  const robotNs = robot.nowNs();
  const hrtime = process.hrtime.bigint();
  assert.ok(hrtime >= robotNs && hrtime - robotNs < 1_000_000n, "the robot's clock is process.hrtime's");
});

test("the session's key is in the host's address, never in what inspecting shows", async () => {
  const key = host.address.split('#')[1];
  assert.ok(key && key.length > 52);
  const policy = new rlmesh.Policy(HOLD, { session: host });
  for (const shown of [inspect(host), inspect(policy), String(host), String(policy)]) assert.ok(!shown.includes(key), shown);
});

/** A loopback port whose next three are free too (a daemon's classes). */
async function freeBase() {
  const { createServer } = await import('node:net');
  const listen = (port) =>
    new Promise((resolve) => {
      const server = createServer();
      server.once('error', () => resolve(null));
      server.listen(port, '127.0.0.1', () => resolve(server));
    });
  for (let attempt = 0; attempt < 50; attempt++) {
    const probe = await listen(0);
    const base = probe.address().port;
    await new Promise((r) => probe.close(r));
    if (base > 65531) continue;
    const held = await Promise.all([0, 1, 2, 3].map((i) => listen(base + i)));
    await Promise.all(held.filter(Boolean).map((s) => new Promise((r) => s.close(r))));
    if (held.every(Boolean)) return base;
  }
  throw new Error('no four free ports');
}

/** An rlmeshd on loopback, whose `local.key` this process attaches with (Python's `rlmesh daemon`). */
async function startDaemon() {
  const { spawn } = await import('node:child_process');
  // Short: its local channel is a Unix socket in there.
  const state = mkdtempSync('/tmp/rlmd-');
  process.env.RLMESH_STATE_DIR = state;
  const base = await freeBase();
  const repo = new URL('../../../../', import.meta.url).pathname;
  const proc = spawn('uv', ['run', '--no-sync', 'python', '-m', 'rlmesh', 'daemon', '--listen', `127.0.0.1:${base}`], {
    cwd: repo,
    env: process.env,
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  await new Promise((resolve, reject) => {
    proc.stdout.once('data', (line) => (String(line).includes('rlmeshd on') ? resolve() : reject(new Error(String(line)))));
    proc.once('exit', (code) => reject(new Error(`rlmeshd exited with ${code}`)));
  });
  return {
    home: `127.0.0.1:${base}`,
    async [Symbol.asyncDispose]() {
      proc.kill();
      await new Promise((r) => proc.once('exit', r));
      delete process.env.RLMESH_STATE_DIR;
      rmSync(state, { recursive: true, force: true });
    },
  };
}

test('a scene runs a lobby robot and a Node policy on one machine', async () => {
  await using daemon = await startDaemon();
  const dir = mkdtempSync(join(tmpdir(), 'rlmesh-scene-'));
  const path = join(dir, 'bench.toml');
  writeFileSync(
    path,
    '[scene]\nformat = 1\nname = "bench"\nwait = "20s"\n\n[members.arm]\nrole = "robot"\n\n[members.pilot]\nrole = "controller"\ndevice = "local"\n',
  );
  const ARM = { arm: { kind: 'joints', count: 2, fields: ['pos'], hz: 100, command: 'position', hold_max_ms: 200 } };
  const PILOT = {
    hz: 20,
    inputs: { q: { kind: 'joints', at: 'arm', fields: ['pos'] } },
    actions: { arm: { kind: 'joints', at: 'arm', command: 'position', hz: 20, horizon: 2 } },
  };
  const robot = (async () => {
    await using arm = await rlmesh.Robot.waitInLobby('arm', ARM, { home: daemon.home, timeout: 3 * WAIT_MS / 1000 });
    while (arm.leaving === null) {
      arm.publish('arm', { pos: new Float32Array(2) });
      await sleep(10);
    }
    return arm.leaving;
  })();
  try {
    await using scene = await rlmesh.runScene(path, { home: daemon.home });
    const binding = await scene.waitUntil('binding', { timeout: WAIT_MS / 1000 });
    assert.deepEqual(binding.members[0], { name: 'arm', stage: 'declared', error: null });
    const policy = new rlmesh.Policy(PILOT, { session: await scene.address(), name: 'pilot' });
    await policy.bind('arm');
    assert.equal((await scene.waitUntil('running', { timeout: WAIT_MS / 1000 })).error, null);
    const stats = await policy.run(() => ({ arm: { data: new Float32Array(4), shape: [2, 2] } }), { duration: 1 });
    assert.ok(stats.starts > 0, JSON.stringify(stats));
    await assert.rejects(rlmesh.runScene(path, { home: daemon.home }), (e) => e instanceof rlmesh.SessionRefused && e.status === 'ROBOT_BUSY');
    await policy.close();
    assert.equal((await scene.stop()).stage, 'ended');
    assert.equal(await robot, 'scene_ended');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
