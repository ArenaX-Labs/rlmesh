// Recovery end to end, under Node: the package's `default` entry (dist/node.js: worker_threads class
// Workers, the real wasm, Node's WebSocket) against the example robot (a real session host, an
// rlmesh-session edge robot and a model policy) and the real rlmesh-gateway.
//
//   1. a real wasm trap in the state Worker -> respawned, the same subscription keeps flowing;
//   2. a trap in the control Worker -> the lease is revoked (worker_restart); its participant is gone,
//      so the host goes SAFE; arm + a new lease (a new token) -> ACTIVE;
//   3. SIGKILL the gateway -> every class socket reports disconnected, the lease is revoked
//      (disconnected); restart it -> sockets reconnect with backoff, data resumes, the host went SAFE,
//      arm + a fresh lease bring it back to ACTIVE.
//
// The page token (`RLMESH_GATEWAY_TOKEN`, which a page gets in its URL fragment) never reaches the
// gateway's logs, at debug.
//
// `mise run test:js:e2e` builds the release binaries and dist/ and sets RLMESH_E2E_TARGET.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { connect } from '../../dist/node.js';

const root = new URL('../../../../', import.meta.url).pathname;
const targetDir = process.env.RLMESH_E2E_TARGET ?? `${root}target/release`;
const bin = { robot: `${targetDir}/examples/web_robot`, gateway: `${targetDir}/rlmesh-gateway` };
const base = 20000 + Math.floor(Math.random() * 20000);
const ROBOT_BASE = base;
const HTTP_PORT = base + 10;
const scratch = mkdtempSync(join(process.env.XDG_CACHE_HOME ?? join(process.env.HOME ?? tmpdir(), '.cache'), 'rlmesh-e2e-'));
const sessionFile = join(scratch, 'session.id');
const TOKEN = randomBytes(24).toString('hex');
const procs = new Set();
/** Everything every gateway logged. */
let gatewayLog = '';

function start(name, path, args, env = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(path, args, { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, RUST_LOG: process.env.RUST_LOG ?? 'warn', ...env } });
    procs.add(p);
    let log = '';
    const t = setTimeout(() => reject(new Error(`${name} not READY: ${log}`)), 30000);
    const onData = (d) => {
      log += d;
      if (name === 'rlmesh-gateway') gatewayLog += d;
      if (log.includes('READY')) {
        clearTimeout(t);
        resolve(p);
      }
    };
    p.stdout.on('data', onData);
    p.stderr.on('data', onData);
    p.on('exit', (code) => {
      procs.delete(p);
      clearTimeout(t);
      if (!log.includes('READY')) reject(new Error(`${name} exited ${code}: ${log}`));
    });
  });
}

const startGateway = () =>
  start('rlmesh-gateway', bin.gateway, ['--connect', `127.0.0.1:${ROBOT_BASE}`, '--session-file', sessionFile, '--port', String(HTTP_PORT), '--max-priority', 'operator', '--browsers', join(scratch, 'none.toml')], {
    RLMESH_GATEWAY_TOKEN: TOKEN,
    RUST_LOG: 'debug,zenoh=warn',
  });
const waitFor = (session, type, pred = () => true, ms = 10000) =>
  new Promise((resolve, reject) => {
    const t = setTimeout(() => { off(); reject(new Error(`no ${type} event within ${ms} ms`)); }, ms);
    const off = session.on(type, (e) => { if (pred(e)) { clearTimeout(t); off(); resolve(e); } });
  });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Next sample from a subscription with seq greater than `after`. */
async function nextAfter(sub, after, ms = 5000) {
  const deadline = Date.now() + ms;
  for (;;) {
    const r = await Promise.race([sub.next(), sleep(Math.max(1, deadline - Date.now())).then(() => null)]);
    if (!r) throw new Error(`no sample after seq ${after} within ${ms} ms`);
    if (r.value.seq > after) return r.value;
  }
}

/** Wait for the session mode (from the declarations' state events). */
async function modeIs(session, mode, ms = 5000) {
  if (session.mode === mode) return;
  await waitFor(session, 'state', (s) => s.mode === mode, ms);
}

/** Re-arm after a fallback and take the lease (preempting the policy, which re-requests too). */
async function rearmAndLease(session, path, ms = 5000) {
  const deadline = Date.now() + ms;
  for (;;) {
    try {
      // Right after a reconnect the new participant may not be attached yet: retry both.
      if (session.mode === 'safe' || session.mode === 'disarmed') await session.arm();
      return await session.lease(path, { timeoutMs: 1000 });
    } catch (e) {
      if (Date.now() > deadline) throw e;
      await sleep(150);
    }
  }
}

let robot, gateway, session;
before(async () => {
  robot = await start('web_robot', bin.robot, ['--listen-base', `127.0.0.1:${ROBOT_BASE}`, '--session-file', sessionFile]);
  gateway = await startGateway();
  session = await connect(`ws://127.0.0.1:${HTTP_PORT}`, { token: TOKEN, priority: 'operator', backoff: { initialMs: 50, maxMs: 400 }, timeoutMs: 10000 });
});
after(() => {
  session?.close();
  for (const p of procs) p.kill('SIGKILL');
  rmSync(scratch, { recursive: true, force: true });
});

test('connects under Node: declarations, a participant of its own, the clock, and samples', async () => {
  assert.deepEqual(session.controls.map((c) => c.path).sort(), ['arm7/arm', 'arm7/gripper']);
  // connect resolves on the declarations; the gateway's binding can land a snapshot later.
  if (!session.controls.every((c) => c.bound)) await waitFor(session, 'state', () => session.controls.every((c) => c.bound));
  assert.ok(session.controls.every((c) => c.bound), 'the gateway bound every commandable part');
  assert.match(session.id, /^web-/);
  const sub = session.subscribe('arm7/gripper');
  const s = await nextAfter(sub, 0);
  assert.equal(s.data.length, 1);
  sub.close();
  assert.ok(Number.isFinite(session.clock.offset_ms));
  assert.ok(session.clock.error_ms < 5, `clock error ${session.clock.error_ms} ms`);
});

test('wasm trap in the state Worker: respawned, the same subscription keeps flowing', async () => {
  const sub = session.subscribe('arm7/arm', { maxQueue: 64 });
  const before = await nextAfter(sub, 0);
  const restarted = waitFor(session, 'worker_restart', (e) => e.cls === 'state');
  session.debug.trap('state');
  const r = await restarted;
  assert.match(r.reason, /RuntimeError/);
  assert.equal(r.restarts, 1);
  const afterTrap = await nextAfter(sub, before.seq + 50);
  assert.ok(afterTrap.seq > before.seq);
  console.log(`  state worker: ${r.reason.slice(0, 120)}; back in ${r.downtime_ms.toFixed(1)} ms; seq ${before.seq} -> ${afterTrap.seq}`);
  sub.close();
});

test('wasm trap in the control Worker: the lease is revoked (worker_restart), the host goes SAFE, a new lease has a new token', async () => {
  const lease = await rearmAndLease(session, 'arm7/arm');
  await modeIs(session, 'active');
  lease.drive([0.2, 0, 0, 0, 0, 0, 0], { hz: 100 });
  await sleep(300);
  const oldId = session.id;
  const restarted = waitFor(session, 'worker_restart', (e) => e.cls === 'control');
  session.debug.trap('control');
  const rev = await lease.revoked;
  assert.equal(rev.cause, 'worker_restart');
  await restarted;
  assert.equal(lease.active, false);
  // The dead Worker's participant left: the host ended its lease and went SAFE.
  await modeIs(session, 'safe', 5000);
  assert.notEqual(session.id, oldId, 'a new participant');
  const again = await rearmAndLease(session, 'arm7/arm');
  assert.notEqual(again.token, lease.token);
  await modeIs(session, 'active', 3000);
  console.log(`  control worker: old token ${lease.token} revoked (${rev.cause}); host SAFE; new participant ${session.id}, token ${again.token}, ACTIVE`);
  again.release();
});

test('gateway killed: sockets report down, the lease is revoked (disconnected); restart: reconnect, resume, re-arm, re-lease', async () => {
  const sub = session.subscribe('arm7/gripper', { maxQueue: 64 });
  const s0 = await nextAfter(sub, 0);
  const grip = await rearmAndLease(session, 'arm7/gripper');
  grip.drive([0.8], { hz: 50 });
  await sleep(300);
  const down = new Set();
  const allDown = new Promise((r) => session.on('disconnected', (e) => { down.add(e.cls); if (down.size === 3) r(); }));
  const t0 = Date.now();
  gateway.kill('SIGKILL');
  const rev = await grip.revoked;
  assert.equal(rev.cause, 'disconnected');
  await allDown;
  assert.equal(await grip.send([0]), false, 'an abandoned lease refuses locally');
  await sleep(1000);
  const up = new Set();
  const allUp = new Promise((r) => session.on('reconnected', (e) => { up.add(e.cls); if (up.size === 3) r(); }));
  gateway = await startGateway();
  await allUp;
  const back = Date.now() - t0;
  const s1 = await nextAfter(sub, s0.seq + 10);
  await modeIs(session, 'safe', 5000);
  const again = await rearmAndLease(session, 'arm7/gripper');
  assert.notEqual(again.token, grip.token);
  await modeIs(session, 'active', 3000);
  console.log(`  gateway: ${down.size} sockets down, reconnected after ${back} ms; gripper seq ${s0.seq} -> ${s1.seq}; host SAFE -> ACTIVE with token ${again.token}`);
  again.release();
  sub.close();
});

test('session host killed: the gateway ends the session (4410), the page is told once, and the gateway exits', async () => {
  const sub = session.subscribe('arm7/gripper', { maxQueue: 64 });
  await nextAfter(sub, 0);
  const ended = new Promise((r) => session.on('ended', r));
  const exited = new Promise((r) => gateway.once('exit', (code) => r(code)));
  const t0 = Date.now();
  robot.kill('SIGKILL');
  const e = await ended;
  assert.equal(e.code, 4410);
  assert.equal((await sub.next()).done, true, 'subscriptions end with the session');
  assert.equal(await exited, 0, 'a gateway serves one session: it exits when that ends');
  console.log(`  host killed: ended (${e.code}: ${e.reason}) after ${Date.now() - t0} ms; gateway exited`);
});

test('the page token never reaches the gateway\'s logs', () => {
  assert.match(gatewayLog, /browser attached/, 'the gateways logged at debug');
  assert.ok(!gatewayLog.includes(TOKEN), 'the token was logged');
});
