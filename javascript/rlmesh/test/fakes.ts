// Fake class Workers for unit tests: a scripted stand-in for src/worker.ts on the other side of postMessage.
import type { Platform } from '../src/session.js';
import type { WorkerLike } from '../src/supervisor.js';
import type { ControlDecl, StreamDecl, TrafficClass } from '../src/types.js';

export class FakeWorker implements WorkerLike {
  onmessage: ((e: { data: any }) => void) | null = null;
  onerror: ((e: { message?: string }) => void) | null = null;
  posted: any[] = [];
  terminated = false;
  init: any;
  constructor(
    readonly cls: TrafficClass,
    private script: (w: FakeWorker, m: any) => void = () => {},
  ) {}
  postMessage(m: any) {
    if (this.terminated) throw new Error('posted to a terminated worker');
    this.posted.push(m);
    if (m.op === 'init') {
      this.init = m;
      queueMicrotask(() => this.emit({ op: 'ready', cls: this.cls, ok: true }));
    }
    this.script(this, m);
  }
  emit(m: any) {
    if (!this.terminated) this.onmessage?.({ data: m });
  }
  terminate() {
    this.terminated = true;
  }
  ops(op: string) {
    return this.posted.filter((m) => m.op === op);
  }
}

const field = (name: string, len: number, offset = 0) => ({ name, dtype: 'f32', shape: [len], unit: '', offset, len });

export const STREAMS: StreamDecl[] = [
  { path: 'arm7/arm', robot: 'arm7', part: 'arm', applied: false, channel: 1, class: 'state', kind: 'f64', rate_hz: 1000, max_age_ms: 20, fields: [field('pos', 7), field('vel', 7, 7)], width: 0, height: 0 },
  { path: 'arm7/skin', robot: 'arm7', part: 'skin', applied: false, channel: 4, class: 'state', kind: 'f64', rate_hz: 100, max_age_ms: 50, fields: [field('pressure', 32)], width: 0, height: 0 },
  { path: 'arm7/arm/_applied', robot: 'arm7', part: 'arm', applied: true, channel: 2, class: 'state', kind: 'f64', rate_hz: 1000, max_age_ms: Number.NaN, fields: [field('pos', 7)], width: 0, height: 0 },
  { path: 'arm7/scene/cam', robot: 'arm7', part: 'scene/cam', applied: false, channel: 5, class: 'bulk', kind: 'jpeg', rate_hz: 20, max_age_ms: 200, fields: [], width: 640, height: 480 },
];

export const CONTROLS: ControlDecl[] = [
  { path: 'arm7/arm', robot: 'arm7', part: 'arm', channel: 1, class: 'control', kind: 'joints', mode: 'position', group: null, fields: [field('pos', 7)], rate_hz: 1000, hold_max_ms: 100, bound: true },
  { path: 'arm7/gripper', robot: 'arm7', part: 'gripper', channel: 3, class: 'control', kind: 'gripper', mode: 'position', group: null, fields: [field('pos', 1)], rate_hz: 100, hold_max_ms: 200, bound: true },
];

export const STATE = { session: '0192f3a4-5b6c-7d8e-9f00-112233445566', revision: 3, epoch: 1, mode: 'active', estop_latched: false };

/** A control worker that answers like the real one: hello, declarations and a clock on init; grants every lease request. */
export function controlScript(opts: { grant?: boolean } = {}) {
  let token = 100;
  return (w: FakeWorker, m: any) => {
    if (m.op === 'init') {
      queueMicrotask(() => {
        w.emit({ op: 'socket', state: 'open', reconnect: false });
        w.emit({ op: 'socket', state: 'attached', id: 'web-0000abcd', session: STATE.session });
        w.emit({ op: 'decls', decls: { session: STATE.session, streams: STREAMS, controls: CONTROLS }, state: STATE });
        w.emit({ op: 'state_bytes', bytes: new ArrayBuffer(8) });
        w.emit({ op: 'clock', participant: 'web-0000abcd', clock: { offset_ms: 1.5, rtt_ms: 0.3, error_ms: 0.2, samples: 4 } });
      });
    }
    if (m.op === 'lease' && opts.grant !== false) {
      const t = ++token;
      queueMicrotask(() => w.emit({ op: 'event', ev: { type: 'lease', event: 'grant', ours: true, path: m.path, holder: 'web-0000abcd', priority: m.priority ?? 'operator', token: t } }));
    }
    if (m.op === 'send') queueMicrotask(() => w.emit({ op: 'reply', rid: m.rid, ok: true }));
    if (m.op === 'verb') queueMicrotask(() => w.emit({ op: 'reply', rid: m.rid, ok: m.name !== 'disarm', error: m.name === 'disarm' ? 'Forbidden: nope' : '' }));
  };
}

export function fakePlatform(opts: { canShareMemory?: boolean; control?: (w: FakeWorker, m: any) => void } = {}) {
  const spawned: FakeWorker[] = [];
  const module = { fake: 'module' } as unknown as WebAssembly.Module;
  let loads = 0;
  const platform: Platform = {
    createWorker: (cls) => {
      const w = new FakeWorker(cls, cls === 'control' ? (opts.control ?? controlScript()) : undefined);
      spawned.push(w);
      return w;
    },
    loadModule: async () => {
      loads++;
      return module;
    },
    canShareMemory: opts.canShareMemory ?? false,
  };
  const latest = (cls: TrafficClass) => spawned.filter((w) => w.cls === cls).at(-1)!;
  return { platform, spawned, module, latest, loads: () => loads };
}

export const until = async (cond: () => boolean, ms = 2000) => {
  const t0 = Date.now();
  while (!cond()) {
    if (Date.now() - t0 > ms) throw new Error('timed out waiting');
    await new Promise((r) => setTimeout(r, 2));
  }
};
