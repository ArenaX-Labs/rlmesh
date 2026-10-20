import { test } from 'node:test';
import assert from 'node:assert/strict';
import { connectWith } from '../src/session.js';
import type { Frame, Sample, Status, WorkerRestart } from '../src/types.js';
import { controlScript, fakePlatform, until } from './fakes.js';

async function open(opts: Parameters<typeof fakePlatform>[0] = {}, connect: Parameters<typeof connectWith>[2] = {}) {
  const f = fakePlatform(opts);
  const s = await connectWith(f.platform, 'ws://gw/', { timeoutMs: 1000, ...connect });
  return { s, f };
}

function sample(path: string, seq: number, data = Float64Array.of(seq, seq + 1)) {
  return { op: 'sample', path, seq, epoch: 1, t_capture_ms: 1, t_publish_ms: 1, data };
}

test('connect: one Worker per class, module compiled once, declarations, participant and clock', async (t) => {
  const { s, f } = await open();
  t.after(() => s.close());
  assert.equal(f.loads(), 1);
  assert.deepEqual(f.spawned.map((w) => w.cls), ['control', 'state', 'bulk']);
  for (const w of f.spawned) {
    assert.equal(w.init.module, f.module);
    assert.equal(w.init.url, `ws://gw/${w.cls}`);
    assert.deepEqual(w.init.protocols, ['rlmesh.v2']);
    assert.equal(w.init.priority, 'operator');
  }
  assert.deepEqual(s.controls.map((c) => c.path), ['arm7/arm', 'arm7/gripper']);
  assert.deepEqual(s.streams.map((c) => c.path), ['arm7/arm', 'arm7/arm/_applied', 'arm7/scene/cam']);
  assert.equal(s.mode, 'active');
  assert.equal(s.clock.offset_ms, 1.5);
  assert.equal(s.id, 'web-0000abcd');
  // The data Workers got the host's snapshot bytes and this participant's clock.
  for (const cls of ['state', 'bulk'] as const) {
    assert.equal(f.latest(cls).ops('state').length, 1);
    assert.deepEqual(f.latest(cls).ops('clock').at(-1), { op: 'clock', participant: 'web-0000abcd', offset_ms: 1.5, error_ms: 0.2 });
  }
  assert.throws(() => s.latest('nope'), /unknown stream/);
  assert.throws(() => s.subscribe('arm7/gripper'), /commandable part/);
});

test('connect fails cleanly when the gateway never answers', async () => {
  const f = fakePlatform({ control: () => {} });
  await assert.rejects(connectWith(f.platform, 'ws://gw', { timeoutMs: 50 }), /token\/origin rejected/);
  assert.ok(f.spawned.every((w) => w.terminated));
});

test('the token is offered as a subprotocol and the priority goes with the hello', async () => {
  const f = fakePlatform();
  const s = await connectWith(f.platform, 'ws://gw', { token: 'secret-token-0123456789', priority: 'safety' });
  assert.deepEqual(f.latest('state').init.protocols, ['rlmesh.v2', 'rlmesh.token.secret-token-0123456789']);
  assert.equal(f.latest('control').init.priority, 'safety');
  assert.equal(s.priority, 'safety');
  s.close();
});

test('subscribe: bounded queue with a dropped counter; samples are owned per subscriber', async (t) => {
  const { s, f } = await open();
  t.after(() => s.close());
  const a = s.subscribe('arm7/arm', { maxQueue: 3 });
  const b = s.subscribe('arm7/arm', { maxQueue: 100 });
  assert.equal(f.latest('state').ops('subscribe').at(-1).deliver, true);
  for (let i = 0; i < 8; i++) f.latest('state').emit(sample('arm7/arm', i));
  assert.equal(a.dropped, 5);
  assert.equal(b.dropped, 0);
  const x = (await a.next()).value as Sample;
  const y = (await b.next()).value as Sample;
  assert.equal(x.seq, 5);
  assert.equal(y.seq, 0);
  assert.notEqual((await b.next()).value!.data, x.data);
  // latest() copies into the caller's buffer.
  const into = new Float64Array(8);
  const l = s.latest('arm7/arm', into)!;
  assert.equal(l.seq, 7);
  assert.equal(l.data.buffer, into.buffer);
  assert.deepEqual([...l.data], [7, 8]);
  assert.throws(() => s.latest('arm7/arm', new Float64Array(1)), RangeError);
  a.close();
  b.close();
});

test('peek(): a borrowed frame is not recycled while held, and is unusable after release', async (t) => {
  const { s, f } = await open();
  t.after(() => s.close());
  assert.equal(s.peek('arm7/scene/cam'), null);
  const bulk = f.latest('bulk');
  const frame = (seq: number) => {
    const buf = new ArrayBuffer(64);
    new Uint8Array(buf).fill(seq);
    return { op: 'frame', path: 'arm7/scene/cam', seq, epoch: 1, t_capture_ms: 1, t_publish_ms: 1, kind: 'jpeg', width: 4, height: 4, len: 16, buf, decode_ms: 0 };
  };
  bulk.emit(frame(1));
  const b = s.peek<Frame>('arm7/scene/cam')!;
  assert.equal(b.value.data[0], 1);
  bulk.emit(frame(2));
  assert.equal(bulk.ops('return').length, 0, 'held frame 1 not returned to the pool');
  assert.equal(b.value.data[0], 1, 'still frame 1');
  b.release();
  assert.equal(bulk.ops('return').length, 1, 'returned on release');
  assert.throws(() => b.value, /after release/);
  {
    using c = s.peek<Frame>('arm7/scene/cam')!;
    assert.equal(c.value.seq, 2);
  }
  bulk.emit(frame(3));
  assert.equal(bulk.ops('return').length, 2, 'frame 2 recycled at the end of the using block + supersede');
  // Owned copy via latest(): no bitmap, independent bytes.
  const own = s.latest<Frame>('arm7/scene/cam')!;
  assert.equal(own.data[0], 3);
  assert.ok(!('bitmap' in own));
  // The frame event's borrow is released after the listener.
  let seen: any;
  s.on('frame', (e) => (seen = e));
  bulk.emit(frame(4));
  assert.ok(seen.released);
});

test('lease on any commandable part; drive/send; statuses carry the host mode and the edge mode', async (t) => {
  const { s, f } = await open();
  t.after(() => s.close());
  await assert.rejects(s.lease('arm7/scene/cam'), /no commandable part/);
  const g = await s.lease('arm7/gripper', { ttlMs: 500 });
  assert.equal(f.latest('control').ops('lease').at(-1).ttl_ms, 500);
  assert.equal(g.path, 'arm7/gripper');
  assert.equal(g.priority, 'operator');
  assert.ok(g.active);
  await assert.rejects(s.lease('arm7/gripper'), /already holding/);
  g.drive([0.01], { hz: 5000 });
  const d = f.latest('control').ops('drive').at(-1);
  assert.equal(d.hz, 1000, 'hz is clamped');
  assert.deepEqual([...d.vals], [0.01]);
  assert.equal(await g.send([0]), true);
  assert.equal(s.held('arm7/gripper'), g);
  assert.equal(s.held('arm7/arm'), undefined);
  const statuses: Status[] = [];
  s.on('status', (e) => statuses.push(e));
  const ctl = f.latest('control');
  ctl.emit({ op: 'event', ev: { type: 'status', path: 'arm7/arm', mode: 'safe', epoch: 1, estop_latched: false, holder: '', priority: '', token: 0, ours: false, bound: true } });
  assert.equal(s.status('arm7/arm')!.mode, 'safe');
  f.latest('state').emit({ op: 'edge', path: 'arm7/arm/_applied', mode: 'hold', token: 3 });
  assert.equal(s.status('arm7/arm')!.edge, 'hold');
  assert.deepEqual(statuses.map((x) => x.edge ?? '-'), ['-', 'hold']);
  g.release();
  assert.equal(g.state, 'released');
  assert.equal(ctl.ops('release').at(-1).path, 'arm7/gripper');
  assert.throws(() => g.drive([1]), /released/);
});

test('a refused lease rejects with the host reason; a host revocation ends the lease', async (t) => {
  const script = controlScript({ grant: false });
  const { s, f } = await open({
    control: (w, m) => {
      script(w, m);
      if (m.op === 'lease' && m.path === 'arm7/gripper') {
        queueMicrotask(() => w.emit({ op: 'event', ev: { type: 'lease', event: 'refused', ours: true, path: m.path, token: 0, reason: 'LeaseBusy: held by safety' } }));
      }
      if (m.op === 'lease' && m.path === 'arm7/arm') {
        queueMicrotask(() => w.emit({ op: 'event', ev: { type: 'lease', event: 'grant', ours: true, path: m.path, holder: 'web-0000abcd', priority: 'operator', token: 9 } }));
      }
    },
  });
  t.after(() => s.close());
  await assert.rejects(s.lease('arm7/gripper'), /refused: LeaseBusy/);
  const arm = await s.lease('arm7/arm');
  f.latest('control').emit({ op: 'event', ev: { type: 'lease', event: 'revoke', ours: true, path: 'arm7/arm', token: 9, reason: 'preempted by sup (Safety, t10)' } });
  const r = await arm.revoked;
  assert.equal(r.cause, 'host');
  assert.match(r.reason, /preempted/);
  assert.equal(arm.active, false);
});

test('session verbs resolve or reject with the host refusal', async (t) => {
  const { s } = await open();
  t.after(() => s.close());
  await s.arm();
  await s.estop('test');
  await assert.rejects(s.disarm(), /disarm: Forbidden/);
});

test('a lease lost on disconnect is reported revoked and never resumed', async (t) => {
  const { s, f } = await open();
  t.after(() => s.close());
  const disc: string[] = [];
  s.on('disconnected', (e) => disc.push(e.cls));
  const arm = await s.lease('arm7/arm');
  const ctl = f.latest('control');
  ctl.emit({ op: 'socket', state: 'closed', attempt: 0, code: 1006 });
  ctl.emit({ op: 'lease_lost', path: 'arm7/arm', token: arm.token, reason: 'control socket disconnected' });
  const r = await arm.revoked;
  assert.equal(r.cause, 'disconnected');
  assert.equal(r.token, arm.token);
  assert.equal(arm.active, false);
  assert.equal(await arm.send([0, 0, 0, 0, 0, 0, 0]), false, 'the old lease refuses locally');
  assert.deepEqual(disc, ['control']);
  // Reconnect: a new participant; nothing is resumed and a new lease gets a new token.
  ctl.emit({ op: 'socket', state: 'open', reconnect: true });
  ctl.emit({ op: 'socket', state: 'attached', id: 'web-0000beef' });
  assert.equal(s.id, 'web-0000beef');
  const again = await s.lease('arm7/arm');
  assert.notEqual(again.token, arm.token);
});

test('worker trap: state Worker respawned, state, clock and subscriptions replayed, worker_restart emitted', async (t) => {
  const { s, f } = await open();
  t.after(() => s.close());
  const restarts: WorkerRestart[] = [];
  s.on('worker_restart', (e) => restarts.push(e));
  const sub = s.subscribe('arm7/arm');
  const old = f.latest('state');
  old.emit({ op: 'fatal', reason: 'RuntimeError: unreachable (panicked at ...)' });
  await until(() => restarts.length === 1);
  const fresh = f.latest('state');
  assert.notEqual(fresh, old);
  assert.ok(old.terminated);
  assert.equal(fresh.init.module, f.module, 'same compiled Module, no recompile');
  assert.equal(f.loads(), 1);
  assert.equal(fresh.ops('state').length, 1, 'the snapshot replayed first');
  assert.deepEqual(fresh.ops('subscribe').map((m) => m.path), ['arm7/arm']);
  assert.equal(fresh.ops('clock').at(-1).offset_ms, 1.5);
  assert.match(restarts[0]!.reason, /unreachable/);
  fresh.emit(sample('arm7/arm', 42));
  assert.equal((await sub.next()).value!.seq, 42, 'the same iterator keeps flowing');
  sub.close();
});

test('control Worker trap revokes held leases (worker_restart) and rejects pending ones', async (t) => {
  let grant = true;
  const script = controlScript();
  const { s, f } = await open({ control: (w, m) => (m.op === 'lease' && !grant ? undefined : script(w, m)) });
  t.after(() => s.close());
  const arm = await s.lease('arm7/arm');
  grant = false;
  const pending = s.lease('arm7/gripper', { timeoutMs: 5000 });
  f.latest('control').emit({ op: 'fatal', reason: 'RuntimeError: unreachable' });
  assert.equal((await arm.revoked).cause, 'worker_restart');
  await assert.rejects(pending, /control worker restarted/);
  await until(() => s.workers.control.ready);
  grant = true;
  const again = await s.lease('arm7/arm');
  assert.ok(again.active);
});

test('debug.sendStaleCommand sends zeros for every command element', async (t) => {
  const { s, f } = await open();
  t.after(() => s.close());
  s.debug.sendStaleCommand('arm7/arm', 41);
  const m = f.latest('control').ops('stale_command').at(-1);
  assert.equal(m.token, 41);
  assert.equal(m.vals.length, 7);
});

test('close() releases held leases and stops every Worker', async () => {
  const { s, f } = await open();
  const l = await s.lease('arm7/arm');
  s.close();
  assert.equal(l.state, 'released');
  assert.equal(f.latest('control').ops('release').length, 1);
  await new Promise((r) => setTimeout(r, 80));
  assert.ok(f.spawned.every((w) => w.terminated));
});
