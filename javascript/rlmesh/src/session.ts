// The `rlmesh` session: the public API on the main thread, over one supervised Worker (and one
// reconnecting WebSocket) per traffic class. Platform specifics (how Workers are created, how the wasm
// Module is loaded) come in through `Platform`, so the same code runs in browsers and under Node.
//
// The control Worker's connection is a session participant of its own (the gateway attaches it): its
// leases are the host's, its heartbeats renew them, and its chunks are fenced by the robot's edge.

import { Lease } from './lease.js';
import { BoundedQueue, type Subscription } from './queue.js';
import { protocolsFor } from './socket.js';
import { Supervisor, type WorkerLike } from './supervisor.js';
import type {
  Borrowed,
  Clock,
  ConnectOptions,
  ConsoleLine,
  ControlDecl,
  EdgeMode,
  Frame,
  LeaseEvent,
  LeaseOptions,
  Priority,
  Revocation,
  Sample,
  SessionEnded,
  SessionEvents,
  SessionState,
  Status,
  StreamDecl,
  SubscribeOptions,
  TrafficClass,
  WorkerRestart,
} from './types.js';

export interface Platform {
  createWorker(cls: TrafficClass): WorkerLike;
  loadModule(): Promise<WebAssembly.Module>;
  /** Whether a SharedArrayBuffer may be shared with Workers (browsers: `crossOriginIsolated`). */
  canShareMemory: boolean;
}

const CLASSES: readonly TrafficClass[] = ['control', 'state', 'bulk'];
const SLOT_BYTES = 256;
const SLOTS = 32;
const SLOT_VALUES = 24;
const nowLocal = () => performance.timeOrigin + performance.now();

/** Fixed-size sample of recent values for p50/p99 on the main thread. */
class Window {
  private v: Float64Array;
  private n = 0;
  constructor(cap = 65536) {
    this.v = new Float64Array(cap);
  }
  push(x: number) {
    this.v[this.n++ % this.v.length] = x;
  }
  reset() {
    this.n = 0;
  }
  summary(threshold = 2) {
    const k = Math.min(this.n, this.v.length);
    if (!k) return { count: 0 };
    const s = this.v.slice(0, k).sort();
    let late = 0;
    for (let i = 0; i < k; i++) if ((s[i] as number) > threshold) late++;
    return { count: this.n, p50_ms: s[Math.floor(k * 0.5)], p99_ms: s[Math.min(k - 1, Math.floor(k * 0.99))], max_ms: s[k - 1], late_pct: (100 * late) / k };
  }
}

/** One received frame: pooled buffer + bitmap, recycled when superseded and no longer pinned. */
interface FrameRec {
  meta: Omit<Frame, 'data' | 'age_ms'>;
  buf: ArrayBuffer;
  len: number;
  bitmap: ImageBitmap | undefined;
  pins: number;
  superseded: boolean;
  recycled: boolean;
}

interface PinToken {
  rec: FrameRec;
  done: boolean;
}

class BorrowedValue<T extends Sample<ArrayBufferView>> implements Borrowed<T> {
  private _released = false;
  constructor(
    private _value: T,
    private _bitmap: ImageBitmap | undefined,
    private onRelease: () => void,
  ) {}
  get value(): T {
    if (this._released) throw new Error(`peek() value of ${this._value.path} used after release`);
    return this._value;
  }
  get bitmap() {
    if (this._released) throw new Error(`peek() bitmap of ${this._value.path} used after release`);
    return this._bitmap;
  }
  get released() {
    return this._released;
  }
  release() {
    if (this._released) return;
    this._released = true;
    this.onRelease();
  }
  [Symbol.dispose]() {
    this.release();
  }
}

type Listener<K extends keyof SessionEvents> = (e: SessionEvents[K]) => void;

interface Wanted {
  deliver: boolean;
}

export interface LoadInfo {
  compile_ms?: number;
  workers_ready_ms?: number;
  connect_ms?: number;
  cross_origin_isolated?: boolean;
  shared_memory: boolean;
  workers: Partial<Record<TrafficClass, any>>;
}

export class Session {
  /** Gateway base URL (`ws://` or `wss://`), without the class suffix. */
  readonly url: string;
  /** The class this connection's participant attaches at. */
  readonly priority: Priority;
  /** Every declared stream: each part's samples, and each commandable part's applied stream. */
  streams: StreamDecl[] = [];
  /** Every commandable part: what `lease()` accepts (when `bound`). */
  controls: ControlDecl[] = [];
  /** The session as the host's latest snapshot shows it. */
  state: SessionState = { session: '', revision: 0, epoch: 0, mode: 'disarmed', estop_latched: false };
  /** The session id (a UUIDv7). */
  sessionId = '';
  /** This connection's participant id. Changes if the control socket reconnects or its Worker restarts. */
  id = '';
  clock: Clock = { offset_ms: Number.NaN, rtt_ms: Number.NaN, error_ms: Number.NaN, samples: 0 };
  readonly load: LoadInfo = { workers: {}, shared_memory: false };
  /** Worker console errors/warnings and wasm panic messages (last 200). */
  readonly consoleLog: ConsoleLine[] = [];

  private module!: WebAssembly.Module;
  private sup: Supervisor;
  private listeners = new Map<string, Set<(e: any) => void>>();
  private statusByPath = new Map<string, Status>();
  private edges = new Map<string, EdgeMode>();
  private leases = new Map<string, Lease>();
  private pending = new Map<string, { resolve: (l: Lease) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> }>();
  private subs = new Map<string, Set<BoundedQueue<any>>>();
  private wanted = new Map<string, Wanted>();
  private latestSamples = new Map<string, Sample>();
  private frames = new Map<string, FrameRec>();
  private stateBytes: ArrayBuffer | undefined;
  private sab: SharedArrayBuffer | undefined;
  private sabSlots = new Map<string, number>();
  private scratch = new Map<string, Float64Array>();
  private reqs = new Map<number, { cls: TrafficClass; resolve: (v: any) => void; reject: (e: Error) => void }>();
  private nextRid = 1;
  private heartbeatsPaused = false;
  private mainLatency = new Window();
  private mainFrameAge = new Window(4096);
  private stallTimer: ReturnType<typeof setInterval> | undefined;
  private closed = false;
  private gotDecls!: () => void;
  private gotClock!: () => void;
  private readonly declsP = new Promise<void>((r) => (this.gotDecls = r));
  private readonly clockP = new Promise<void>((r) => (this.gotClock = r));
  private refused!: (e: Error) => void;
  /** Rejects when the gateway ends the connection for good before connect resolves. */
  private readonly refusedP = (() => {
    const p = new Promise<never>((_, rej) => (this.refused = rej));
    p.catch(() => {}); // only connect awaits it; afterwards the `ended` event reports the same
    return p;
  })();
  /** Set once the gateway ended the connection for good. */
  ended: SessionEnded | null = null;
  private robots = new Set<string>();
  private readonly pins = new FinalizationRegistry<PinToken>((t) => {
    if (!t.done) {
      t.done = true;
      t.rec.pins--;
      this.recycle(t.rec);
    }
  });

  /** @internal Use `connect()`. */
  constructor(
    url: string,
    private platform: Platform,
    private opts: ConnectOptions = {},
  ) {
    this.url = url.replace(/\/$/, '');
    this.priority = opts.priority ?? 'model';
    this.sup = new Supervisor(
      {
        spawn: (cls) => this.platform.createWorker(cls),
        initMessage: (cls) => ({
          op: 'init',
          cls,
          module: this.module,
          url: `${this.url}/${cls}`,
          protocols: protocolsFor(this.opts.token),
          backoff: this.opts.backoff,
          sab: cls === 'state' ? this.sab : undefined,
          heartbeats: !this.heartbeatsPaused,
          priority: this.priority,
        }),
        onMessage: (cls, m) => this.onWorker(cls, m),
        onDown: (cls, reason) => this.onWorkerDown(cls, reason),
        onRestarted: (info) => this.onWorkerRestarted(info),
      },
      { stallMs: opts.workerStallMs ?? 10000 },
    );
  }

  // ------------------------------------------------------------------ events

  on<K extends keyof SessionEvents>(type: K, fn: Listener<K>): () => void {
    let set = this.listeners.get(type);
    if (!set) this.listeners.set(type, (set = new Set()));
    set.add(fn);
    return () => set.delete(fn);
  }

  private emit<K extends keyof SessionEvents>(type: K, e: SessionEvents[K]) {
    this.listeners.get(type)?.forEach((f) => {
      try {
        f(e);
      } catch (err) {
        queueMicrotask(() => {
          throw err;
        });
      }
    });
  }

  private hasListeners(type: keyof SessionEvents) {
    return (this.listeners.get(type)?.size ?? 0) > 0;
  }

  // ------------------------------------------------------------------ declarations, status, clock

  /** The session mode. */
  get mode() {
    return this.state.mode;
  }

  /** The host's view of a commandable part (holder, token, mode), with its edge mode when known. */
  status(path: string): Status | undefined {
    return this.statusByPath.get(path);
  }

  /** Session time now (ms since the UNIX epoch, in the host's clock domain). */
  nowSessionMs() {
    return nowLocal() + (Number.isFinite(this.clock.offset_ms) ? this.clock.offset_ms : 0);
  }

  /** Whether each class Worker is up (false while it is being respawned). */
  get workers(): Record<TrafficClass, { ready: boolean; restarts: number }> {
    const o = {} as Record<TrafficClass, { ready: boolean; restarts: number }>;
    for (const c of CLASSES) o[c] = { ready: this.sup.isReady(c), restarts: this.sup.restarts(c) };
    return o;
  }

  private decl(path: string): StreamDecl {
    const d = this.streams.find((s) => s.path === path);
    if (d) return d;
    if (this.controls.some((c) => c.path === path)) {
      throw new Error(`${path} is a commandable part: lease() it, or subscribe to ${path} for its state`);
    }
    throw new Error(`unknown stream ${path}; declared: ${this.streams.map((s) => s.path).join(', ')}`);
  }

  // ------------------------------------------------------------------ worker plumbing

  private post(cls: TrafficClass, m: unknown, t: Transferable[] = []) {
    return this.sup.post(cls, m, t);
  }

  private request(cls: TrafficClass, m: Record<string, unknown>): Promise<any> {
    return new Promise((resolve, reject) => {
      const rid = this.nextRid++;
      this.reqs.set(rid, { cls, resolve, reject });
      if (!this.post(cls, { ...m, rid })) {
        this.reqs.delete(rid);
        reject(new Error(`${cls} worker is restarting`));
      }
    });
  }

  private ensureSub(path: string) {
    const d = this.decl(path);
    const cls = d.class;
    const hasSubs = (this.subs.get(path)?.size ?? 0) > 0;
    let slot: number | undefined;
    // A slot holds SLOT_VALUES numbers; a larger stream is delivered per sample instead.
    const values = d.fields.reduce((n, f) => n + f.len, 0);
    if (cls === 'state' && d.kind === 'f64' && this.sab && values <= SLOT_VALUES) {
      slot = this.sabSlots.get(path);
      if (slot == null && this.sabSlots.size < SLOTS) this.sabSlots.set(path, (slot = this.sabSlots.size));
    }
    // State with an SAB slot only needs per-sample messages for subscribers; bulk frames always come.
    const deliver = cls === 'bulk' || hasSubs || slot == null;
    const prev = this.wanted.get(path);
    if (prev && prev.deliver === deliver) return;
    this.wanted.set(path, { deliver });
    this.post(cls, { op: 'subscribe', path, deliver, sab_slot: slot });
  }

  private forwardState(cls: 'state' | 'bulk') {
    if (!this.stateBytes) return;
    const copy = this.stateBytes.slice(0);
    this.post(cls, { op: 'state', bytes: copy }, [copy]);
  }

  private forwardClock(cls: 'state' | 'bulk') {
    if (Number.isFinite(this.clock.offset_ms)) {
      this.post(cls, { op: 'clock', participant: this.id, offset_ms: this.clock.offset_ms, error_ms: this.clock.error_ms });
    }
  }

  // ------------------------------------------------------------------ samples

  /**
   * Every sample of a stream as an async iterator, with a bounded queue: a slow consumer loses the
   * oldest samples (counted in `dropped`), the producer never blocks. Samples are owned by you. Frames
   * arrive as owned byte copies without a bitmap; use `peek()` or the `frame` event to draw.
   */
  subscribe<T extends ArrayBufferView = Float64Array>(path: string, opts: SubscribeOptions = {}): Subscription<Sample<T>> {
    this.decl(path);
    let set = this.subs.get(path);
    if (!set) this.subs.set(path, (set = new Set()));
    const q: BoundedQueue<Sample<T>> = new BoundedQueue<Sample<T>>(opts.maxQueue ?? 1024, () => {
      set.delete(q);
      // A stream whose robot left is gone: nothing to re-subscribe.
      if (!this.closed && this.streams.some((d) => d.path === path)) this.ensureSub(path);
    });
    set.add(q);
    this.ensureSub(path);
    return q;
  }

  private readSlot(path: string): { seq: number; tc: number; tp: number; n: number; vals: Float64Array } | null {
    const slot = this.sabSlots.get(path);
    if (!this.sab || slot == null) return null;
    const base = slot * SLOT_BYTES;
    const lock = new Int32Array(this.sab, base, 2);
    const meta = new Float64Array(this.sab, base + 8, 4);
    const vals = new Float64Array(this.sab, base + 40, SLOT_VALUES);
    let out = this.scratch.get(path);
    if (!out) this.scratch.set(path, (out = new Float64Array(SLOT_VALUES)));
    for (let tries = 0; tries < 100; tries++) {
      const a = Atomics.load(lock, 0);
      if (a & 1) continue;
      const seq = meta[0] as number, tc = meta[1] as number, tp = meta[2] as number, n = meta[3] as number;
      out.set(vals);
      if (Atomics.load(lock, 0) === a) return n ? { seq, tc, tp, n, vals: out.subarray(0, n) } : null;
    }
    return null;
  }

  private copyInto<T extends Float64Array | Uint8Array>(src: T, into: T | undefined): T {
    if (!into) return src.slice() as T;
    if (into.length < src.length) throw new RangeError(`latest(): buffer of ${into.length} < ${src.length}`);
    into.set(src as any);
    return into.subarray(0, src.length) as T;
  }

  /**
   * The newest value of a stream, as a copy you own (into `into` when given; it must be large
   * enough). Returns null until the first value arrives (the first call starts the subscription).
   * Numeric state streams read a SharedArrayBuffer slot (seqlock) when memory can be shared, so no
   * per-sample messages reach the main thread. Frames come without a bitmap; `peek()` has it.
   */
  latest<T extends Sample<ArrayBufferView> = Sample>(path: string, into?: Float64Array | Uint8Array): T | null;
  latest(path: string, into?: Float64Array | Uint8Array): Sample<ArrayBufferView> | null {
    const cls = this.decl(path).class;
    this.ensureSub(path);
    const h = nowLocal();
    if (cls === 'bulk') {
      const f = this.frames.get(path);
      if (!f) return null;
      const data = this.copyInto(new Uint8Array(f.buf, 0, f.len), into as Uint8Array | undefined);
      return { ...f.meta, data, age_ms: h - f.meta.t_capture_ms };
    }
    const r = this.readSlot(path);
    if (r) return { path, seq: r.seq, epoch: this.state.epoch, t_capture_ms: r.tc, age_ms: h - r.tc, latency_ms: h - r.tp, data: this.copyInto(r.vals, into as Float64Array | undefined) };
    const s = this.latestSamples.get(path);
    if (!s) return null;
    return { ...s, age_ms: h - s.t_capture_ms, data: this.copyInto(s.data, into as Float64Array | undefined) };
  }

  /**
   * The newest value without copying: `value.data` (and `bitmap` for frames) are the session's pooled
   * buffers, valid until you release it (`release()` or `using`). A held frame is not recycled; release
   * promptly. State values are a per-path scratch view, overwritten by the next `peek()` of that path.
   */
  peek<T extends Sample<ArrayBufferView> = Sample>(path: string): Borrowed<T> | null;
  peek(path: string): Borrowed<Sample> | Borrowed<Frame> | null {
    const cls = this.decl(path).class;
    this.ensureSub(path);
    const h = nowLocal();
    if (cls === 'bulk') {
      const rec = this.frames.get(path);
      return rec ? this.borrowFrame(rec, h) : null;
    }
    const r = this.readSlot(path);
    if (r) return new BorrowedValue<Sample>({ path, seq: r.seq, epoch: this.state.epoch, t_capture_ms: r.tc, age_ms: h - r.tc, latency_ms: h - r.tp, data: r.vals }, undefined, () => {});
    const s = this.latestSamples.get(path);
    return s ? new BorrowedValue<Sample>({ ...s, age_ms: h - s.t_capture_ms }, undefined, () => {}) : null;
  }

  private borrowFrame(rec: FrameRec, h: number): Borrowed<Frame> {
    rec.pins++;
    const token: PinToken = { rec, done: false };
    const b = new BorrowedValue<Frame>({ ...rec.meta, data: new Uint8Array(rec.buf, 0, rec.len), age_ms: h - rec.meta.t_capture_ms }, rec.bitmap, () => {
      if (token.done) return;
      token.done = true;
      this.pins.unregister(token);
      rec.pins--;
      this.recycle(rec);
    });
    this.pins.register(b, token, token);
    return b;
  }

  private recycle(rec: FrameRec) {
    if (rec.pins > 0 || !rec.superseded || rec.recycled) return;
    rec.recycled = true;
    rec.bitmap?.close();
    if (rec.buf.byteLength) this.post('bulk', { op: 'return', buf: rec.buf }, [rec.buf]);
  }

  // ------------------------------------------------------------------ leases and verbs

  /**
   * Acquire the lease on a commandable part; resolves when the host grants it (preempting lower
   * classes than this connection's `priority`). Rejects if refused (busy, not bound, not ready), on
   * timeout, or if the control socket drops before the grant.
   */
  lease(path: string, opts: LeaseOptions = {}): Promise<Lease> {
    const d = this.controls.find((c) => c.path === path);
    if (!d) {
      return Promise.reject(new Error(`no commandable part ${path}; declared: ${this.controls.map((c) => c.path).join(', ') || '(none)'}`));
    }
    if (this.leases.get(path)?.active) return Promise.reject(new Error(`already holding ${path}; release() it first`));
    if (this.pending.has(path)) return Promise.reject(new Error(`a lease request for ${path} is in flight`));
    const timeoutMs = opts.timeoutMs ?? 3000;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(path);
        this.post('control', { op: 'release', path });
        reject(new Error(`lease on ${path}: no answer within ${timeoutMs} ms`));
      }, timeoutMs);
      this.pending.set(path, { resolve, reject, timer });
      if (!this.post('control', { op: 'lease', path, ttl_ms: opts.ttlMs })) {
        clearTimeout(timer);
        this.pending.delete(path);
        reject(new Error('control worker is restarting'));
      }
    });
  }

  /** The active lease this session holds on `path`, if any. */
  held(path: string): Lease | undefined {
    const l = this.leases.get(path);
    return l?.active ? l : undefined;
  }

  private async verb(name: string, reason = '') {
    const r = await this.request('control', { op: 'verb', name, reason });
    if (!r.ok) throw new Error(`${name}: ${r.error}`);
  }

  /** DISARMED or SAFE → ARMED, in a fresh epoch (the way back after a fallback). */
  arm() {
    return this.verb('arm');
  }

  /** Back to DISARMED, revoking every lease. */
  disarm() {
    return this.verb('disarm');
  }

  /** Emergency stop: accepted from anyone, and latched until the robot's local clear. */
  estop(reason = 'estop from the browser') {
    return this.verb('estop', reason);
  }

  private leaseHost = {
    post: (m: unknown) => this.post('control', m),
    request: (m: Record<string, unknown>) => this.request('control', m),
    dropLease: (l: Lease) => {
      if (this.leases.get(l.path) === l) this.leases.delete(l.path);
    },
  };

  /** Robots come and go with their parts: report each change, and end the leases and streams of one that left. */
  private trackRobots() {
    const now = new Set([...this.streams.map((s) => s.robot), ...this.controls.map((c) => c.robot)]);
    for (const robot of this.robots) {
      if (now.has(robot)) continue;
      for (const path of [...this.leases.keys(), ...this.pending.keys()]) {
        if (path.startsWith(`${robot}/`)) this.endLease(path, { cause: 'robot_left', reason: `robot ${robot} left the session` });
      }
      for (const [path, set] of this.subs) {
        if (!path.startsWith(`${robot}/`)) continue;
        for (const q of [...set]) q.close();
        this.subs.delete(path);
        this.wanted.delete(path);
      }
      this.emit('robot', { robot, present: false });
    }
    for (const robot of now) if (!this.robots.has(robot)) this.emit('robot', { robot, present: true });
    this.robots = now;
  }

  /** The gateway ended the connection for good: fail a pending connect, or end the session. */
  private end(e: SessionEnded) {
    if (this.ended) return;
    this.ended = e;
    const why = e.code === 4401 ? 'the token was refused' : e.code === 4403 ? 'this page origin is not allowed' : e.code === 4410 ? 'the session ended' : 'the gateway closed';
    this.refused(new Error(`${this.url}: ${why} (${e.code}${e.reason ? `: ${e.reason}` : ''})`));
    for (const path of [...this.leases.keys(), ...this.pending.keys()]) this.endLease(path, { cause: 'session_ended', reason: why });
    for (const set of this.subs.values()) for (const q of [...set]) q.close();
    this.emit('ended', e);
  }

  private endLease(path: string, r: Omit<Revocation, 'path' | 'token'>) {
    const p = this.pending.get(path);
    if (p) {
      clearTimeout(p.timer);
      this.pending.delete(path);
      p.reject(new Error(`lease on ${path}: ${r.reason}`));
    }
    const l = this.leases.get(path);
    if (l) {
      this.leases.delete(path);
      l._end(r);
    }
  }

  // ------------------------------------------------------------------ worker messages

  private onWorker(cls: TrafficClass, m: any) {
    switch (m.op) {
      case 'reply': {
        const r = this.reqs.get(m.rid);
        if (r) {
          this.reqs.delete(m.rid);
          r.resolve(m);
        }
        break;
      }
      case 'sample': {
        const h = nowLocal();
        const s: Sample = {
          path: m.path, seq: m.seq, epoch: m.epoch, t_capture_ms: m.t_capture_ms, age_ms: h - m.t_capture_ms, latency_ms: h - m.t_publish_ms, data: m.data,
          ...(m.applied ? { applied: m.applied } : {}), ...(m.text != null ? { text: m.text } : {}),
        };
        this.mainLatency.push(s.latency_ms);
        this.latestSamples.set(m.path, s);
        let first = true;
        this.subs.get(m.path)?.forEach((q) => {
          q.push(first ? s : { ...s, data: s.data.slice() });
          first = false;
        });
        break;
      }
      case 'edge': {
        const control = String(m.path).replace(/\/_applied$/, '');
        this.edges.set(control, m.mode);
        const st = this.statusByPath.get(control);
        if (st) {
          const next: Status = { ...st, edge: m.mode };
          this.statusByPath.set(control, next);
          this.emit('status', next);
        }
        break;
      }
      case 'frame':
        this.onFrame(m);
        break;
      case 'clock':
        if (Number.isFinite(m.clock.offset_ms)) {
          this.clock = m.clock;
          this.forwardClock('state');
          this.forwardClock('bulk');
          if (m.clock.samples >= 2) this.gotClock();
          this.emit('clock', this.clock);
        }
        break;
      case 'decls':
        this.streams = m.decls.streams;
        this.controls = m.decls.controls;
        this.trackRobots();
        this.sessionId = m.decls.session ?? this.sessionId;
        this.state = m.state;
        this.emit('state', this.state);
        if (this.streams.length || this.controls.length) this.gotDecls();
        break;
      case 'state_bytes':
        this.stateBytes = m.bytes;
        this.forwardState('state');
        this.forwardState('bulk');
        break;
      case 'event':
        this.onControlEvent(m.ev);
        break;
      case 'lease_lost':
        // The control socket dropped: the token is abandoned locally and never used again.
        this.endLease(m.path, { cause: 'disconnected', reason: m.reason });
        break;
      case 'lease_state':
        for (const [path, l] of Object.entries(m.leases as Record<string, any>)) this.leases.get(path)?._update(l);
        this.emit('lease_state', { leases: m.leases, cmd: m.cmd });
        break;
      case 'socket': {
        if (m.state === 'attached') {
          this.id = m.id;
          if (m.session) this.sessionId = m.session;
          this.forwardClock('state');
          this.forwardClock('bulk');
          break;
        }
        const e = { cls, attempt: m.attempt ?? 0, ...(m.code != null ? { code: m.code } : {}), ...(m.reason ? { reason: m.reason } : {}), ...(m.fatal ? { fatal: true } : {}) };
        if (m.state === 'closed') this.emit('disconnected', e);
        if (m.state === 'closed' && m.fatal) this.end({ code: m.code, reason: m.reason || `closed with ${m.code}` });
        else if (m.reconnect) this.emit('reconnected', e);
        break;
      }
      case 'console':
        this.consoleLog.push({ cls: m.cls, lvl: m.lvl, msg: m.msg });
        if (this.consoleLog.length > 200) this.consoleLog.splice(0, this.consoleLog.length - 200);
        this.emit('console', { cls: m.cls, lvl: m.lvl, msg: m.msg });
        break;
    }
  }

  private onFrame(m: any) {
    const h = nowLocal();
    const prev = this.frames.get(m.path);
    const rec: FrameRec = {
      meta: { path: m.path, seq: m.seq, epoch: m.epoch, t_capture_ms: m.t_capture_ms, latency_ms: h - m.t_publish_ms, kind: m.kind, width: m.width, height: m.height, decode_ms: m.decode_ms },
      buf: m.buf,
      len: m.len,
      bitmap: m.bitmap ?? undefined,
      pins: 0,
      superseded: false,
      recycled: false,
    };
    this.frames.set(m.path, rec);
    if (prev) {
      prev.superseded = true;
      this.recycle(prev);
    }
    this.mainFrameAge.push(h - m.t_capture_ms);
    const subs = this.subs.get(m.path);
    if (subs?.size) {
      const owned: Frame = { ...rec.meta, age_ms: h - m.t_capture_ms, data: new Uint8Array(rec.buf, 0, rec.len).slice() };
      subs.forEach((q) => q.push(owned));
    }
    if (this.hasListeners('frame')) {
      const b = this.borrowFrame(rec, h);
      try {
        this.emit('frame', b);
      } finally {
        b.release();
      }
    }
  }

  private onControlEvent(ev: any) {
    switch (ev.type) {
      case 'status': {
        const edge = this.edges.get(ev.path);
        const st: Status = {
          path: ev.path, mode: ev.mode, epoch: ev.epoch, estop_latched: ev.estop_latched, holder: ev.holder, priority: ev.priority,
          token: ev.token, ours: ev.ours, bound: ev.bound, ...(edge ? { edge } : {}),
        };
        this.statusByPath.set(st.path, st);
        this.emit('status', st);
        break;
      }
      case 'lease': {
        const e: LeaseEvent = { event: ev.event, path: ev.path, holder: ev.holder ?? '', priority: ev.priority ?? '', token: ev.token ?? 0, ours: !!ev.ours, ...(ev.reason ? { reason: ev.reason } : {}) };
        if (e.event === 'grant') {
          const p = this.pending.get(e.path);
          if (p) {
            clearTimeout(p.timer);
            this.pending.delete(e.path);
            const l = new Lease(this.leaseHost, e.path, this.priority, e.token, e.holder);
            this.leases.set(e.path, l);
            p.resolve(l);
          }
        } else if (e.event === 'refused') {
          const p = this.pending.get(e.path);
          if (p) {
            clearTimeout(p.timer);
            this.pending.delete(e.path);
            p.reject(new Error(`lease on ${e.path} refused: ${e.reason ?? 'refused'}`));
          }
        } else if (e.event === 'revoke') {
          this.endLease(e.path, { cause: 'host', reason: e.reason ?? 'revoked' });
        }
        this.emit('lease', e);
        break;
      }
      case 'fence':
        this.emit('fence', { channel: ev.channel, token: ev.token, reason: ev.reason, ...(ev.local ? { local: true } : {}) });
        break;
    }
  }

  // ------------------------------------------------------------------ supervision

  private onWorkerDown(cls: TrafficClass, reason: string) {
    for (const [rid, r] of this.reqs) {
      if (r.cls === cls) {
        this.reqs.delete(rid);
        r.reject(new Error(`${cls} worker restarted: ${reason}`));
      }
    }
    if (cls === 'control') {
      // The lease clients lived in that wasm instance, and its socket was the participant: the host ends
      // the leases (the new Worker attaches a new participant).
      for (const path of new Set([...this.leases.keys(), ...this.pending.keys()])) {
        this.endLease(path, { cause: 'worker_restart', reason: `control worker restarted: ${reason}` });
      }
    }
    if (cls === 'state' && this.sab) {
      // A Worker killed mid-write leaves a slot's seqlock odd; reset every slot it owned.
      for (const slot of this.sabSlots.values()) Atomics.store(new Int32Array(this.sab, slot * SLOT_BYTES, 1), 0, 0);
    }
  }

  private onWorkerRestarted(info: WorkerRestart) {
    const { cls } = info;
    if (cls !== 'control') {
      this.forwardState(cls);
      this.forwardClock(cls);
      for (const [path, w] of this.wanted) {
        if (this.decl(path).class !== cls) continue;
        this.post(cls, { op: 'subscribe', path, deliver: w.deliver, sab_slot: cls === 'state' ? this.sabSlots.get(path) : undefined });
      }
    }
    this.emit('worker_restart', info);
  }

  // ------------------------------------------------------------------ lifecycle

  /** @internal */
  async _start() {
    const t0 = performance.now();
    this.module = await this.platform.loadModule();
    this.load.compile_ms = performance.now() - t0;
    this.load.cross_origin_isolated = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated;
    if (this.platform.canShareMemory && typeof SharedArrayBuffer === 'function') this.sab = new SharedArrayBuffer(SLOTS * SLOT_BYTES);
    this.load.shared_memory = !!this.sab;
    const ready = await Promise.all(CLASSES.map((c) => this.sup.start(c)));
    CLASSES.forEach((c, i) => (this.load.workers[c] = ready[i]));
    // What the control Worker relayed before the data Workers were up.
    for (const cls of ['state', 'bulk'] as const) {
      this.forwardState(cls);
      this.forwardClock(cls);
    }
    this.load.workers_ready_ms = performance.now() - t0;
    this.stallTimer = setInterval(() => this.sup.checkStalls(), 1000);
    const ms = this.opts.timeoutMs ?? 5000;
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        Promise.all([this.declsP, this.clockP]),
        this.refusedP,
        new Promise((_, rej) => {
          timer = setTimeout(() => rej(new Error(`no declarations and clock from ${this.url} within ${ms} ms (unreachable, no session, or token/origin rejected?)`)), ms);
        }),
      ]);
    } finally {
      clearTimeout(timer);
    }
    this.load.connect_ms = performance.now() - t0;
  }

  async stats(): Promise<any> {
    const [state, bulk] = await Promise.all([this.request('state', { op: 'stats' }), this.request('bulk', { op: 'stats' })]);
    return { state, bulk, main: { sample_latency: this.mainLatency.summary(), frame_age: this.mainFrameAge.summary(1e9) } };
  }

  /** Release held leases, stop every Worker and socket. */
  close() {
    if (this.closed) return;
    const held = [...this.leases.values()].filter((l) => l.active);
    for (const l of held) l.release();
    for (const path of [...this.pending.keys()]) this.endLease(path, { cause: 'closed', reason: 'session closed' });
    this.closed = true;
    clearInterval(this.stallTimer);
    for (const s of this.subs.values()) for (const q of s) q.close();
    // Give the release frames a moment to leave the control socket.
    if (held.length) setTimeout(() => this.sup.close(), 50);
    else this.sup.close();
  }

  [Symbol.dispose]() {
    this.close();
  }

  /** Test and demo hooks. Not part of the stable API. */
  readonly debug = {
    /** Stop (or resume) heartbeats while staying connected, as a hung tab would. */
    pauseHeartbeats: (paused: boolean) => {
      this.heartbeatsPaused = paused;
      this.post('control', { op: 'heartbeats', enabled: !paused });
    },
    /** Send one command with an explicit (stale) token, bypassing the local lease check; a `fence` event reports the verdict. */
    sendStaleCommand: (path: string, token: number, values?: ArrayLike<number>) => {
      const d = this.controls.find((c) => c.path === path);
      const n = d ? d.fields.reduce((a, f) => a + f.len, 0) : 0;
      this.post('control', { op: 'stale_command', path, token, vals: Float64Array.from(values ?? new Array(n).fill(0)) });
    },
    resetStats: () => {
      this.post('state', { op: 'reset_stats' });
      this.post('bulk', { op: 'reset_stats' });
      this.mainLatency.reset();
      this.mainFrameAge.reset();
    },
    /** Trip a real wasm trap (`Instant::now()` panics on wasm32-unknown-unknown) in a class Worker. */
    trap: (cls: TrafficClass) => this.post(cls, { op: 'trap' }),
    /** Drop a class socket as if the network blinked; it reconnects with backoff. */
    dropSocket: (cls: TrafficClass) => this.post(cls, { op: 'drop_socket' }),
  };
}

/** Shared by the browser and Node entry points. */
export async function connectWith(platform: Platform, url: string, opts: ConnectOptions = {}): Promise<Session> {
  if (opts.workers && opts.workers !== 'per-class') throw new Error('only workers: "per-class" is supported');
  const s = new Session(url, platform, opts);
  try {
    await s._start();
  } catch (e) {
    s.close();
    throw e;
  }
  return s;
}
