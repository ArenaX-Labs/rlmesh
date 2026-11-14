// The native binding on loopback: a Node robot and Node policies in one session, the twin of
// python/rlmesh/tests/integration/test_v2_session.py. The robot is the UR10e (the conformance
// fixture), faked: its control loop publishes joints, wrench, gripper and two cameras, and applies
// whatever its edge says is due. The policies bind to it by kind and path, lease its parts and
// stream timed chunks. A lockstep env, served by Python (`trials_env.py`, through `uv run`), is run
// by episodes. Needs dist/node.js and the addon (`mise run test:js:native`).
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { createRequire } from 'node:module';
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
    arm: { kind: 'joints', at: 'arm', command: 'position', names: UR_JOINTS, hz: 100, chunk: 10 },
    gripper: { kind: 'gripper', at: 'gripper', command: 'position', hz: 100, chunk: 10 },
  },
  allow: ['resize', 'crop'],
};

/** Holds one target on the arm. */
const HOLD = {
  hz: 50,
  inputs: { q: { kind: 'joints', at: 'arm', fields: ['pos'], names: UR_JOINTS } },
  actions: { arm: { kind: 'joints', at: 'arm', command: 'position', names: UR_JOINTS, hz: 100, chunk: 10 } },
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

/** `[chunk, 6]` of one value. */
const hold = (target, chunk = 10) => ({ arm: { data: new Float32Array(chunk * 6).fill(target), shape: [chunk, 6] } });
const first = (due) => due.value.data[0];

/** The UR10e without physics: commanded joints become measured ones. */
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
  assert.equal(await policy.bind('ur10e'), policy);
  const binding = policy.binding;
  assert.equal(policy.clock, 'wall');
  await assert.rejects(policy.run(() => ({}), { episodes: 2 }), (e) => e instanceof RangeError && /episodes= is for a lockstep robot/.test(e.message));
  assert.match(String(binding), /wrist +in +wrist\/cam +bound +decode\(jpeg\) → crop\(640x480 → 96x96, bilinear\)/);
  assert.equal(binding.json.actions[0].chunk, 10);
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
  assert.equal(stats.clock, 'wall');
  assert.ok(stats.completions > 0 && stats.chunksSent > 0, JSON.stringify(stats));
  assert.equal(stats.sendErrors, 0);
  const q = stats.inputs.find((i) => i.entry === 'q');
  assert.equal(q.part, 'ur10e:arm');
  assert.ok(q.received > 0 && q.lost >= 0, JSON.stringify(stats.inputs));
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

test('drive binds and runs a policy in one call, closing it when the run ends', async () => {
  await using ur = await FakeUR10e.start(host);
  const hold = () => ({ arm: { data: new Float32Array(60).map((_, i) => HOME[i % 6]), shape: [10, 6] }, gripper: { data: new Float64Array(10).fill(0.5), shape: [10, 1] } });
  const stats = await rlmesh.drive(SINE, 'ur10e', hold, { session: host, name: 'drive', duration: 1 });
  assert.equal(stats.clock, 'wall');
  assert.ok(stats.completions > 0 && stats.chunksSent > 0, JSON.stringify(stats));
  assert.ok(ur.log.some((d) => d.mode === 'driving'), 'the robot played the chunks');
});

test('the binding table and refusals match the conformance goldens', async () => {
  await using robot = await rlmesh.Robot.connect('ur10e', fixture('robots', 'ur10e'), { session: host });
  await using rtde = await rlmesh.Robot.connect('ur10e_rtde', fixture('robots', 'ur10e_rtde'), { session: host });
  assert.equal(robot.parts['wrist/cam'], 'camera');
  assert.equal(rtde.name, 'ur10e_rtde');
  {
    await using pi0 = new rlmesh.Policy(fixture('policies', 'pi0_single'), { session: host, name: 'pi0_single' });
    const binding = (await pi0.bind('ur10e')).binding;
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
  assert.match(String((await refused.bind('ur10e')).binding), /wrist\/ft/);
  await refused.close();

  await assert.rejects(
    new rlmesh.Policy(HOLD, { session: host, name: 'typo' }).bind('ur10e', { allow: ['resise'] }),
    (e) => e instanceof rlmesh.BindingRefused && /unknown conversion/.test(e.message),
  );
});

test('an operator preempts the model, which stays stopped until resumed', async () => {
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
  assert.ok(tokens['ur10e:arm'] > modelToken);
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
  assert.equal(ur.log.at(-1).leaseToken, tokens['ur10e:arm']);
  assert.ok(model.stats().skipped.parked > 0, 'a preempted model stays warm, skips its starts');
  assert.deepEqual(model.stats().parked, ['ur10e:arm']);

  // A model cannot take it back while the operator holds it.
  await assert.rejects(model.acquire(), (e) => e instanceof rlmesh.SessionRefused && e.status === 'LEASE_BUSY');

  // The operator lets go: the model stays stopped, and the arm goes safe.
  streaming = false;
  await teleop;
  await operator.release();
  const parked = model.stats().skipped.parked;
  await waitUntil(
    'the model parked and the arm safe',
    () => model.stats().skipped.parked >= parked + 25 && ur.log.at(-1).mode === 'safe',
    state,
  );
  assert.equal(model.holds, false);
  const events = model.events().filter((e) => e.kind === 'parked');
  assert.deepEqual(
    events.map((e) => [e.part, e.by, e.message]),
    [['ur10e:arm', 'teleop', 'ur10e:arm was taken by teleop at Operator; this policy stays stopped until policy.resume()']],
  );

  // Resumed, the model re-requests and drives under a fresh token.
  assert.deepEqual(model.resume(), ['ur10e:arm']);
  await waitUntil('the model driving again', () => ur.driven((d) => first(d) === 0.25));
  const resumed = ur.log.at(-1).leaseToken;
  assert.ok(resumed !== null && resumed > tokens['ur10e:arm']);
  model.stop();
  assert.equal(typeof (await running).starts, 'number');
});

test("a model set to resume on its own takes the arm back when the operator lets go", async () => {
  await using ur = await FakeUR10e.start(host);
  await using model = new rlmesh.Policy(HOLD, { session: host, name: 'model' });
  await using operator = new rlmesh.Policy(HOLD, { session: host, name: 'teleop', priority: 'operator' });
  await model.bind('ur10e');
  await operator.bind('ur10e');
  await assert.rejects(model.run(() => hold(0.25), { resume: 'later' }), RangeError);
  const running = model.run(() => hold(0.25), { duration: 30, resume: 'auto' });
  await waitUntil('the model driving', () => ur.driven((d) => first(d) === 0.25));
  await operator.acquire();
  await waitUntil("the model's lease gone", () => !model.holds);
  await operator.release();
  await waitUntil('the model driving again, unresumed', () => ur.driven((d) => first(d) === 0.25) && model.holds);
  assert.equal(model.stats().skipped.parked, 0);
  assert.deepEqual(model.stats().parked, []);
  model.stop();
  await running;
});

test("a viewer reads every link's health, each robot's refusals by reason", async () => {
  await using one = await FakeUR10e.start(host);
  await using two = await FakeUR10e.start(host, 'ur10e-2');
  await using model = new rlmesh.Policy(HOLD, { session: host, name: 'model' });
  await using viewer = new rlmesh.Policy({ hz: 25, inputs: HOLD.inputs }, { session: host, name: 'viewer' });
  await model.bind('ur10e');
  await viewer.bind('ur10e-2');
  const running = model.run(() => hold(0.25), { duration: 30 });
  await waitUntil('the model driving', () => one.driven((d) => first(d) === 0.25));

  // The viewer neither publishes nor commands; its first read subscribes and waits for one.
  let health = await viewer.health();
  while (!(health.links['ur10e']?.lastChunkAccepted > 0)) health = await viewer.health({ wait: true });
  const names = Object.keys(health.links).sort();
  assert.deepEqual(names, ['model', 'ur10e', 'ur10e-2', 'viewer']);
  for (const name of names) {
    const link = health.links[name];
    assert.ok(link.heartbeats > 0 && link.rttMs > 0 && link.offsetErrorMs !== null, inspect(link));
    assert.equal(link.health, 'ok');
  }
  assert.deepEqual(health.links['ur10e-2'].rejectedBy, {});
  assert.equal(health.links['ur10e'].rejected, 0);
  assert.ok(health.links['ur10e'].published['ur10e:arm'].published > 0);
  assert.ok(health.links.model.run.chunksSent > 0);
  assert.ok(health.links.viewer.lastSeqSeen['ur10e-2:arm'] > 0);
  assert.equal(health.links['ur10e'].run, null);

  // A robot reads the same; the next one comes later.
  const now = await two.robot.health();
  const later = await two.robot.health({ wait: true });
  assert.ok(later.hostNs > now.hostNs);
  await assert.rejects(viewer.health({ wait: true, timeout: 0.001 }), /health/);
  model.stop();
  await running;
});

test('a recorder writes the session to MCAP and reads it back', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'rlmesh-record-'));
  try {
    const path = join(dir, 'session.mcap');
    const recorder = await rlmesh.SessionRecorder.start(path, { session: host });
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

test('a camera publishes frames it already encoded', async () => {
  // A 16x8 JPEG, as a webcam in MJPEG mode hands it out.
  const frame = Buffer.from(
    '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wAARCAAIABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDHooorhPqD/9k=',
    'base64',
  );
  const dir = mkdtempSync(join(tmpdir(), 'rlmesh-jpeg-'));
  try {
    const path = join(dir, 'jpeg.mcap');
    const recorder = await rlmesh.SessionRecorder.start(path, { session: host });
    {
      await using camera = await rlmesh.Robot.connect(
        'webcam',
        { cam: { kind: 'camera', width: 16, height: 8, hz: 20, encoding: 'jpeg' } },
        { session: host },
      );
      for (let i = 0; i < 10; i++) {
        camera.publishJpeg('cam', frame);
        await sleep(60);
      }
      assert.throws(() => camera.publishJpeg('cam', Buffer.from('not a jpeg')), /not a JPEG/);
    }
    await recorder.close();
    const topics = Object.fromEntries((await rlmesh.readRecording(path)).topics.map((t) => [t.topic, t]));
    assert.ok(topics['/webcam/cam']?.messages > 0);
    assert.equal(topics['/webcam/cam'].undecodable, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('a policy publishes annotations under its name and never binds them', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'rlmesh-annotate-'));
  try {
    const path = join(dir, 'tracked.mcap');
    const recorder = await rlmesh.SessionRecorder.start(path, { session: host });
    {
      await using eye = await rlmesh.Robot.connect(
        'eye-1',
        { eyes: { kind: 'joints', names: ['pan', 'tilt'], fields: ['pos'], hz: 200, command: 'position', safe: 'stop' } },
        { session: host },
      );
      await using tracker = new rlmesh.Policy(
        { hz: 25, actions: { gaze: { kind: 'joints', at: 'eyes', command: 'position', hz: 50 } } },
        { session: host, name: 'tracker', publishes: { target: { kind: 'x/target', hz: 25, fields: { xy: ['float32', [2]] } } } },
      );
      // One robot besides the tracker's own parts: bound with no robot named, the eye alone.
      const binding = (await tracker.bind()).binding;
      assert.match(binding.toString(), /binding tracker on eye-1/);
      assert.throws(() => tracker.publish('boxes', Float32Array.of(0, 0)), /no part 'boxes'/);
      // For half a second: the recorder subscribes to a new part once it sees it.
      for (let i = 0; i < 25; i++) {
        tracker.publish('target', Float32Array.of(0.01 * i, 0.5));
        await sleep(20);
      }
      void eye;
    }
    await recorder.close();
    const summary = await rlmesh.readRecording(path);
    const target = summary.topics.find((t) => t.topic === '/tracker/target');
    assert.ok(target, summary.topics.map((t) => t.topic).join(', '));
    assert.ok(target.messages >= 1);
    assert.equal(target.undecodable, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('a shadow run predicts on a disarmed session, sends nothing, and commands once armed', async () => {
  const EYE = { eyes: { kind: 'joints', names: ['pan', 'tilt'], fields: ['pos'], hz: 200, command: 'position', safe: 'hold', hold_max_ms: 200 } };
  await using eye = await rlmesh.Robot.connect('eye-1', EYE, { session: host });
  const driven = [];
  let stopped = false;
  const loop = (async () => {
    const rate = new rlmesh.Rate(200);
    while (!stopped) {
      eye.publish('eyes', { pos: Float32Array.of(0, 0) });
      const due = eye.due('eyes');
      if (due?.mode === 'driving' && due.value) driven.push(first(due));
      await rate.sleep();
    }
  })();
  await using policy = new rlmesh.Policy(
    { hz: 25, inputs: { q: { kind: 'joints', at: 'eyes', fields: ['pos'] } }, actions: { gaze: { kind: 'joints', at: 'eyes', command: 'position', hz: 50 } } },
    { session: host, name: 'tracker' },
  );
  await policy.bind('eye-1');
  const steps = [];
  const run = policy.run(() => ({ gaze: { data: Float32Array.of(0.25, 0), shape: [1, 2] } }), {
    duration: 60,
    mode: 'shadow',
    onStep: (step) => steps.push(step),
  });
  await waitUntil('shadow predictions', () => policy.stats().shadowPredictions >= 10 && steps.length >= 10);
  const stats = policy.stats();
  assert.equal(stats.chunksSent, 0);
  assert.equal(stats.skipped.no_lease, 0);
  const edge = eye.edgeStats('eyes');
  assert.equal(edge.accepted + edge.rejected, 0, 'the edge got a chunk');
  assert.equal(steps[0].shadow, true);
  assert.deepEqual(steps[0].parts, { gaze: 'eye-1:eyes' });
  assert.deepEqual(steps[0].seqs, { gaze: null });
  assert.deepEqual(steps[0].actions.gaze.pos.shape, [1, 2]);
  assert.equal(steps[0].actions.gaze.pos.data[0], 0.25);
  const events = policy.events();
  assert.ok(events.some((e) => e.kind === 'shadowing' && /DISARMED/.test(e.message)), JSON.stringify(events));

  await eye.arm();
  await waitUntil('the eye driven by the same run', () => driven.length >= 5);
  assert.ok(policy.events().some((e) => e.kind === 'commanding'));
  assert.ok(steps.some((s) => !s.shadow && typeof s.seqs.gaze === 'number'));
  policy.stop();
  const final = await run;
  assert.ok(final.chunksSent > 0 && final.shadowPredictions >= 10);
  await assert.rejects(policy.run(() => ({}), { mode: 'sideways' }), RangeError);
  stopped = true;
  await loop;
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

/**
 * A local env served as a lockstep robot by Python's `rlmesh.EnvServer` (`trials_env.py`), in a
 * session of its own: its session's address and the resets it saw (`{ seed, trial }`), in order.
 */
async function serveTrials() {
  const { spawn } = await import('node:child_process');
  const { createInterface } = await import('node:readline');
  const base = `127.0.0.1:${await freeBase()}`;
  const repo = new URL('../../../../', import.meta.url).pathname;
  const script = new URL('trials_env.py', import.meta.url).pathname;
  const proc = spawn('uv', ['run', '--no-sync', 'python', script, `${base}/trials`], {
    cwd: repo,
    env: process.env,
    stdio: ['pipe', 'pipe', 'inherit'],
  });
  const exited = new Promise((resolve) => proc.once('exit', resolve));
  const resets = [];
  await new Promise((resolve, reject) => {
    createInterface({ input: proc.stdout }).on('line', (line) => {
      const message = JSON.parse(line);
      if (message.serving) resolve();
      else resets.push(message);
    });
    exited.then((code) => reject(new Error(`trials_env.py exited with ${code}`)));
  });
  return {
    base,
    resets,
    async [Symbol.asyncDispose]() {
      proc.stdin.end();
      await exited;
    },
  };
}

test('a Node policy runs a lockstep env by episodes, each reset as a model run resets it', async () => {
  await using env = await serveTrials();
  const box = { kind: 'gym/box', fields: { value: ['float32', [2]] } };
  await using policy = new rlmesh.Policy({ hz: 10, inputs: { x: { ...box, at: 'obs' } }, actions: { a: { ...box, at: 'action' } } }, { session: env.base, name: 'p' });
  await policy.bind('trials');
  assert.equal(policy.clock, 'lockstep');
  assert.equal(policy.holds, true, 'its lease is taken at bind');
  const still = () => ({ a: { data: new Float32Array(2), shape: [1, 2] } });
  const played = [];
  const stats = await policy.run(still, { episodes: 3, seed: 5, onStep: (step) => played.push(step.actions.a) });
  // Two steps an episode, one action a chunk: a predict a step, a reward of 1 each.
  assert.deepEqual(stats, {
    clock: 'lockstep',
    starts: 6,
    chunksSent: 6,
    episodes: Array(3).fill({ return: 2, steps: 2, terminated: true, truncated: false }),
  });
  assert.deepEqual(policy.stats(), stats);
  assert.equal(played.length, 6);
  // Each episode's reset is seeded from the base and carries its trial_index, numbered from 0.
  await waitUntil('three resets', () => env.resets.length === 3);
  const first = env.resets.splice(0);
  assert.deepEqual(first.map((r) => r.trial), [0, 1, 2]);
  assert.equal(new Set(first.map((r) => r.seed)).size, 3, JSON.stringify(first));
  // The same seed runs the same episodes again.
  assert.deepEqual(await policy.run(still, { episodes: 3, seed: 5 }), stats);
  await waitUntil('three resets again', () => env.resets.length === 3);
  assert.deepEqual(env.resets.splice(0), first);

  // The wall clock's settings throw at once, as do its verbs.
  for (const [opts, says] of [
    [{ duration: 1 }, /duration= is for a wall-clock robot/],
    [{ mode: 'shadow' }, /mode=shadow is for a wall-clock robot/],
    [{ resume: 'auto' }, /resume= is for a wall-clock robot/],
    [{ episodes: 0 }, /episodes=0/],
  ]) {
    await assert.rejects(policy.run(still, opts), (e) => e instanceof RangeError && says.test(e.message));
  }
  await assert.rejects(policy.acquire(), (e) => e instanceof rlmesh.SessionException && /runs in lockstep/.test(e.message));
  assert.deepEqual(policy.events(), []);
  assert.equal(env.resets.length, 0, 'nothing was reset');
});

/** A free loopback port (a daemon listens on it). */
async function freeBase() {
  const { createServer } = await import('node:net');
  const probe = createServer();
  await new Promise((resolve) => probe.listen(0, '127.0.0.1', resolve));
  const { port } = probe.address();
  await new Promise((r) => probe.close(r));
  return port;
}

/** A host with its lobby on loopback, whose `local.key` this process attaches with (`rlmesh host --lobby`). */
async function startDaemon() {
  const { spawn } = await import('node:child_process');
  // Short: its local channel is a Unix socket in there.
  const state = mkdtempSync('/tmp/rlmd-');
  process.env.RLMESH_STATE_DIR = state;
  const base = await freeBase();
  const repo = new URL('../../../../', import.meta.url).pathname;
  const proc = spawn('uv', ['run', '--no-sync', 'python', '-m', 'rlmesh', 'host', '--lobby', '--no-record', '--listen', `127.0.0.1:${base}`], {
    cwd: repo,
    env: process.env,
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  await new Promise((resolve, reject) => {
    proc.stdout.once('data', (line) => (String(line).includes('rlmesh host on') ? resolve() : reject(new Error(String(line)))));
    proc.once('exit', (code) => reject(new Error(`rlmesh host exited with ${code}`)));
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

// The lobby and scenes are unreleased (`fleet`): only an addon built with it runs this.
const fleetAddon = typeof createRequire(import.meta.url)(
  process.env.RLMESH_NATIVE ?? `../../dist/rlmesh.${process.platform}-${process.arch}.node`,
).runScene === 'function';
test('a scene runs a lobby robot and a Node policy on one machine', { skip: !fleetAddon && 'needs an addon built with fleet' }, async () => {
  await using daemon = await startDaemon();
  const dir = mkdtempSync(join(tmpdir(), 'rlmesh-scene-'));
  const path = join(dir, 'bench.toml');
  writeFileSync(
    path,
    '[scene]\nformat = 1\nname = "bench"\nwait = "20s"\n\n[members.arm]\nrole = "robot"\n\n[members.pilot]\nrole = "controller"\ndevice = "local"\n',
  );
  const ARM = { arm: { kind: 'joints', count: 2, fields: ['pos'], hz: 100, command: 'position', safe: 'stop', hold_max_ms: 200 } };
  const PILOT = {
    hz: 20,
    inputs: { q: { kind: 'joints', at: 'arm', fields: ['pos'] } },
    actions: { arm: { kind: 'joints', at: 'arm', command: 'position', hz: 20, chunk: 2 } },
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
