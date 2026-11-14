import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Supervisor } from '../src/supervisor.js';
import type { TrafficClass, WorkerRestart } from '../src/types.js';
import { FakeWorker } from './fakes.js';

function setup(opts: { failInit?: boolean } = {}) {
  const spawned: FakeWorker[] = [];
  const downs: string[] = [];
  const restarts: WorkerRestart[] = [];
  const messages: any[] = [];
  const timers: { fn: () => void; ms: number }[] = [];
  let t = 0;
  const sup = new Supervisor(
    {
      spawn: (cls) => {
        const w = new FakeWorker(cls);
        if (opts.failInit) w.postMessage = (m: any) => queueMicrotask(() => w.emit({ op: 'ready', ok: false, error: `boom ${m.op}` }));
        spawned.push(w);
        return w;
      },
      initMessage: (cls) => ({ op: 'init', cls, module: 'the-module' }),
      onMessage: (_cls, m) => messages.push(m),
      onDown: (cls, reason) => downs.push(`${cls}: ${reason}`),
      onRestarted: (i) => restarts.push(i),
    },
    { stallMs: 1000, now: () => t, setTimeout: (fn, ms) => timers.push({ fn, ms }) },
  );
  return { sup, spawned, downs, restarts, messages, timers, advance: (ms: number) => (t += ms) };
}

test('a wasm trap (fatal) respawns the class Worker from the same Module', async () => {
  const s = setup();
  await s.sup.start('state' as TrafficClass);
  const first = s.spawned[0]!;
  first.emit({ op: 'sample', n: 1 });
  assert.equal(s.messages.length, 1);
  first.emit({ op: 'fatal', reason: 'RuntimeError: unreachable' });
  assert.ok(first.terminated);
  assert.deepEqual(s.downs, ['state: RuntimeError: unreachable']);
  assert.equal(s.sup.post('state', { op: 'x' }), false, 'posts are dropped while down');
  assert.equal(s.timers[0]!.ms, 0, 'first restart is immediate');
  s.timers.shift()!.fn();
  const second = s.spawned[1]!;
  assert.equal(second.init.module, 'the-module');
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(s.restarts.length, 1);
  assert.equal(s.restarts[0]!.cls, 'state');
  assert.equal(s.restarts[0]!.restarts, 1);
  assert.ok(s.sup.post('state', { op: 'x' }));
  // Messages from the dead Worker are ignored.
  first.terminated = false;
  first.emit({ op: 'sample', n: 2 });
  assert.equal(s.messages.length, 1);
});

test('crash loops back off; onerror and stalls also restart', async () => {
  const s = setup();
  await s.sup.start('bulk' as TrafficClass);
  for (let i = 0; i < 3; i++) {
    s.spawned.at(-1)!.onerror?.({ message: 'uncaught' });
    s.timers.shift()!.fn();
    await new Promise((r) => setTimeout(r, 0));
    s.advance(100);
  }
  assert.deepEqual(
    s.downs.map((d) => d.split(':')[0]),
    ['bulk', 'bulk', 'bulk'],
  );
  assert.equal(s.restarts.length, 3);
  // Stall: no message for > stallMs (checked on time) -> restart.
  s.sup.checkStalls();
  s.advance(800);
  s.sup.checkStalls();
  s.advance(800);
  s.sup.checkStalls();
  assert.match(s.downs.at(-1)!, /stalled/);
  assert.ok(s.timers.at(-1)!.ms > 0, 'repeated restarts within the window are delayed');
});

test('a blocked caller does not count as a stall', async () => {
  const s = setup();
  await s.sup.start('state' as TrafficClass);
  s.advance(5000); // the main thread was blocked: the check itself is late
  s.sup.checkStalls();
  assert.equal(s.downs.length, 0);
});

test('a Worker that cannot instantiate fails start() instead of looping', async () => {
  const s = setup({ failInit: true });
  await assert.rejects(s.sup.start('control' as TrafficClass), /failed to start: init failed: boom init/);
  assert.equal(s.timers.length, 0);
});
