// Keeps one Worker per traffic class alive. A wasm trap (a Rust panic under panic=abort surfaces as
// `RuntimeError: unreachable`) poisons that instance, so the class Worker reports `fatal` and the
// supervisor respawns it from the already-compiled Module. Also restarts on `onerror` (uncaught error,
// Node worker exit) and on a stall (no message for `stallMs`).

import { backoffDelay } from './socket.js';
import type { TrafficClass, WorkerRestart } from './types.js';

export interface WorkerLike {
  postMessage(m: unknown, transfer?: Transferable[]): void;
  terminate(): void;
  onmessage: ((e: { data: any }) => void) | null;
  onerror: ((e: { message?: string }) => void) | null;
}

export interface SupervisorHooks {
  spawn(cls: TrafficClass): WorkerLike;
  /** The `init` message for a (re)spawned Worker; includes the shared compiled Module. */
  initMessage(cls: TrafficClass): unknown;
  /** Every message except `ready` and `fatal`. */
  onMessage(cls: TrafficClass, m: any): void;
  /** The Worker is gone (trap, error, stall); called before the respawn delay. */
  onDown(cls: TrafficClass, reason: string): void;
  /** A respawned Worker is ready; replay subscriptions etc. */
  onRestarted(info: WorkerRestart): void;
}

export interface SupervisorOptions {
  /** Restart a Worker that posted nothing for this long (ms). 0 disables. Default 10000. */
  stallMs?: number;
  /** Two restarts closer than this count as a crash loop and back off. Default 30000. */
  crashLoopWindowMs?: number;
  now?: () => number;
  setTimeout?: (fn: () => void, ms: number) => unknown;
}

interface Slot {
  w: WorkerLike | null;
  ready: boolean;
  restarts: number;
  consecutive: number;
  lastRestartAt: number;
  lastSeen: number;
  downAt: number;
  reason: string;
  /** Resolves the first start. */
  first?: { resolve: (m: any) => void; reject: (e: Error) => void };
}

export class Supervisor {
  private slots = new Map<TrafficClass, Slot>();
  private closed = false;
  private lastCheck: number;
  private readonly now: () => number;
  private readonly st: (fn: () => void, ms: number) => unknown;

  constructor(
    private h: SupervisorHooks,
    private opts: SupervisorOptions = {},
  ) {
    this.now = opts.now ?? (() => performance.now());
    this.st = opts.setTimeout ?? ((fn, ms) => setTimeout(fn, ms));
    this.lastCheck = this.now();
  }

  /** Spawn a class Worker; resolves with its `ready` message, rejects if it cannot instantiate. */
  start(cls: TrafficClass): Promise<any> {
    const slot: Slot = { w: null, ready: false, restarts: 0, consecutive: 0, lastRestartAt: -Infinity, lastSeen: this.now(), downAt: 0, reason: '' };
    this.slots.set(cls, slot);
    const p = new Promise<any>((resolve, reject) => (slot.first = { resolve, reject }));
    this.spawn(cls, slot);
    return p;
  }

  isReady(cls: TrafficClass) {
    return !!this.slots.get(cls)?.ready;
  }

  restarts(cls: TrafficClass) {
    return this.slots.get(cls)?.restarts ?? 0;
  }

  /** Post to a ready Worker. Returns false (message dropped) while it is down or restarting. */
  post(cls: TrafficClass, m: unknown, transfer: Transferable[] = []): boolean {
    const s = this.slots.get(cls);
    if (!s?.w || !s.ready) return false;
    s.w.postMessage(m, transfer);
    return true;
  }

  /** Tear down and respawn `cls` (also the entry point for `fatal`, `onerror` and stalls). */
  restart(cls: TrafficClass, reason: string) {
    const s = this.slots.get(cls);
    if (!s || this.closed || !s.w) return;
    if (s.first) {
      // Never came up: fail `connect()` instead of looping.
      s.w.terminate();
      s.w = null;
      s.first.reject(new Error(`${cls} worker failed to start: ${reason}`));
      s.first = undefined;
      return;
    }
    const w = s.w;
    s.w = null;
    s.ready = false;
    w.onmessage = w.onerror = null;
    w.terminate();
    s.downAt = this.now();
    s.reason = reason;
    s.restarts++;
    s.consecutive = s.downAt - s.lastRestartAt < (this.opts.crashLoopWindowMs ?? 30000) ? s.consecutive + 1 : 0;
    s.lastRestartAt = s.downAt;
    this.h.onDown(cls, reason);
    const delay = s.consecutive === 0 ? 0 : backoffDelay(s.consecutive - 1, { initialMs: 250, maxMs: 10000, jitter: 0 });
    this.st(() => {
      if (!this.closed) this.spawn(cls, s);
    }, delay);
  }

  /** Call periodically. Skips a round when the caller's own timer ran late (a blocked main thread). */
  checkStalls() {
    const stallMs = this.opts.stallMs ?? 10000;
    const now = this.now();
    const late = now - this.lastCheck > 3000;
    this.lastCheck = now;
    if (!stallMs) return;
    for (const [cls, s] of this.slots) {
      if (late) {
        s.lastSeen = now;
        continue;
      }
      if (s.ready && now - s.lastSeen > stallMs) this.restart(cls, `stalled: no message for ${Math.round(now - s.lastSeen)} ms`);
    }
  }

  close() {
    this.closed = true;
    for (const s of this.slots.values()) {
      if (s.w) {
        s.w.onmessage = s.w.onerror = null;
        s.w.terminate();
      }
      s.w = null;
      s.ready = false;
    }
  }

  private spawn(cls: TrafficClass, s: Slot) {
    let w: WorkerLike;
    try {
      w = this.h.spawn(cls);
    } catch (e) {
      s.first?.reject(e instanceof Error ? e : new Error(String(e)));
      s.first = undefined;
      return;
    }
    s.w = w;
    s.lastSeen = this.now();
    w.onmessage = (e) => {
      if (s.w !== w) return;
      const m = e.data;
      s.lastSeen = this.now();
      if (m?.op === 'ready') {
        if (!m.ok) return this.restart(cls, `init failed: ${m.error}`);
        s.ready = true;
        if (s.first) {
          s.first.resolve(m);
          s.first = undefined;
        } else {
          this.h.onRestarted({ cls, reason: s.reason, restarts: s.restarts, downtime_ms: this.now() - s.downAt });
        }
        return;
      }
      if (m?.op === 'fatal') return this.restart(cls, String(m.reason));
      this.h.onMessage(cls, m);
    };
    w.onerror = (e) => {
      if (s.w !== w) return;
      this.restart(cls, `worker error: ${e?.message ?? 'unknown'}`);
    };
    w.postMessage(this.h.initMessage(cls));
  }
}
