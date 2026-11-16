// The lobby and scene runs, unreleased (`fleet`), as python/rlmesh/tests/integration/
// test_v2_session.py runs them through `rlmesh.realtime._fleet`: a lobby robot and a Node policy in
// one scene on a loopback rlmesh host. dist/fleet.js is the package plus the private src/fleet.ts;
// the npm package exports none of it. Needs an addon built with the `fleet` feature (a release
// addon skips it) and `uv` with the in-tree Python env for the host.
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { setTimeout as sleep } from 'node:timers/promises';
import * as rlmesh from '../../dist/fleet.js';

const WAIT_MS = 10_000;

test('the package exports none of the lobby and scenes', async () => {
  const pkg = await import('../../dist/node.js');
  for (const name of ['waitInLobby', 'leaving', 'runScene', 'Scene']) assert.equal(name in pkg, false, name);
  assert.equal('leaving' in pkg.Robot.prototype, false);
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

// Only an addon built with `fleet` has the lobby and scenes.
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
    await using arm = await rlmesh.waitInLobby('arm', ARM, { home: daemon.home, timeout: 3 * WAIT_MS / 1000 });
    while (rlmesh.leaving(arm) === null) {
      arm.publish('arm', { pos: new Float32Array(2) });
      await sleep(10);
    }
    return rlmesh.leaving(arm);
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
