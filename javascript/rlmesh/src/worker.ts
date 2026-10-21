// One instance per traffic class (control | state | bulk), each in its own dedicated Worker with its own
// reconnecting WebSocket and its own wasm instance. Classes never share a Worker: sharing a thread between
// the 1 kHz state socket and the camera socket brings the head-of-line hiccup back.
//
// Runs as a browser module Worker or a Node worker_threads Worker (the `default` export condition).
// A wasm trap poisons this instance: every entry point is guarded, and a `WebAssembly.RuntimeError`
// stops the Worker's sockets and timers and posts `fatal`, so the supervisor on the main thread respawns
// it from the compiled Module.
//
// Every binary frame is a real v2 message behind a one-byte tag (wasm/src/frame.rs); the
// wasm encodes and decodes them with the session wire (rlmesh-session-core). Text frames are the gateway's JSON.

import { initSync, Streams, ControlSession, build_info, web_time_now_ms, std_instant_now_panics } from '../pkg/rlmesh.js';
import { ReconnectingSocket } from './socket.js';
import type { BackoffOptions } from './types.js';

type Msg = Record<string, any>;
interface Port {
  post(m: Msg, transfer?: Transferable[]): void;
  listen(fn: (m: Msg) => void): void;
}

async function getPort(): Promise<Port> {
  const g = globalThis as any;
  if (typeof g.WorkerGlobalScope !== 'undefined' && g instanceof g.WorkerGlobalScope) {
    return {
      post: (m, t = []) => g.postMessage(m, t),
      listen: (fn) => (g.onmessage = (e: MessageEvent) => fn(e.data)),
    };
  }
  const { parentPort } = await import('node:worker_threads');
  if (!parentPort) throw new Error('rlmesh worker: not running in a Worker');
  return {
    post: (m, t = []) => parentPort.postMessage(m, t as any),
    listen: (fn) => parentPort.on('message', fn),
  };
}

const port = await getPort();
const now = () => performance.timeOrigin + performance.now();
let cls = '?';
let poisoned = false;
let lastPanic = '';
const cleanups: (() => void)[] = [];
const post = (m: Msg, t: Transferable[] = []) => {
  if (!poisoned || m.op === 'fatal' || m.op === 'console') port.post(m, t);
};

// Surface wasm panic messages (console_error_panic_hook writes to console.error) and warnings.
for (const lvl of ['error', 'warn'] as const) {
  const orig = console[lvl].bind(console);
  console[lvl] = (...a: unknown[]) => {
    const msg = a.map(String).join(' ').slice(0, 2000);
    // console_error_panic_hook: "panicked at file:line:col:\n<message>\n\nStack: ..."
    if (lvl === 'error' && msg.includes('panicked')) lastPanic = msg.split('\n').slice(0, 2).join(' ').trim();
    post({ op: 'console', cls, lvl, msg });
    orig(...a);
  };
}

function fatal(e: unknown) {
  if (poisoned) return;
  const err = e as Error;
  const reason = `${err?.name ?? 'Error'}: ${err?.message ?? String(e)}${lastPanic ? ` (${lastPanic})` : ''}`;
  poisoned = true;
  for (const c of cleanups) {
    try {
      c();
    } catch {
      /* best effort */
    }
  }
  port.post({ op: 'fatal', cls, reason });
}

/** Run `fn`; a wasm trap poisons the instance, other errors are reported and survived. */
function guard<A extends unknown[]>(fn: (...a: A) => void): (...a: A) => void {
  return (...a: A) => {
    if (poisoned) return;
    try {
      fn(...a);
    } catch (e) {
      if (e instanceof WebAssembly.RuntimeError) fatal(e);
      else post({ op: 'console', cls, lvl: 'error', msg: String((e as Error)?.stack ?? e).slice(0, 2000) });
    }
  };
}

function every(ms: number, fn: () => void) {
  const h = setInterval(guard(fn), ms);
  cleanups.push(() => clearInterval(h));
}

/** A wasm call that may refuse: a `JsError` is returned as a message, a trap still poisons. */
function attempt<T>(fn: () => T): { ok: true; value: T } | { ok: false; error: string } {
  try {
    return { ok: true, value: fn() };
  } catch (e) {
    if (e instanceof WebAssembly.RuntimeError) throw e;
    return { ok: false, error: String((e as Error)?.message ?? e) };
  }
}

interface InitMsg {
  cls: string;
  url: string;
  protocols: string[];
  backoff?: BackoffOptions;
  sab?: SharedArrayBuffer;
  heartbeats?: boolean;
  priority?: string;
}

function openSocket(
  init: InitMsg,
  onOpen: (reconnect: boolean) => void,
  onMessage: (b: ArrayBuffer) => void,
  onText: (t: string) => void,
  onClose?: () => void,
) {
  const sock = new ReconnectingSocket(
    init.url,
    init.protocols,
    {
      onOpen: guard((reconnect) => {
        post({ op: 'socket', cls, state: 'open', reconnect });
        onOpen(reconnect);
      }),
      onMessage: guard(onMessage),
      onText: guard(onText),
      onClose: guard((i) => {
        post({ op: 'socket', cls, state: 'closed', ...i });
        onClose?.();
      }),
    },
    init.backoff ? { backoff: init.backoff } : {},
  );
  cleanups.push(() => sock.close());
  return sock;
}

function parse(t: string): Msg | null {
  try {
    return JSON.parse(t) as Msg;
  } catch {
    return null;
  }
}

// ------------------------------------------------------------------ state / bulk

interface Sub {
  idx: number;
  path: string;
  deliver: boolean;
  kind: string;
  // SharedArrayBuffer seqlock slot for `latest()` (state class, when memory can be shared)
  lock?: Int32Array;
  meta?: Float64Array;
  vals?: Float64Array;
  slot?: number;
  /** The last edge mode seen (applied streams). */
  edge?: string;
}

/** A subscription asked for before the declarations arrived. */
interface Wanted {
  path: string;
  deliver: boolean;
  sab_slot?: number;
}

function runData(init: InitMsg) {
  const streams = new Streams();
  const subs: Sub[] = [];
  const waiting = new Map<string, Wanted>();
  const pool: ArrayBuffer[] = [];
  let haveState = false;
  let decodeSum = 0, decodeN = 0, decodeMax = 0;
  let copySum = 0, copyN = 0;
  let imgSum = 0, imgN = 0;
  let frameBusy = false, framesSkipped = 0;
  const canBitmap = typeof (globalThis as any).createImageBitmap === 'function';

  const subscribeMsg = (s: Sub) => JSON.stringify({ type: 'subscribe', id: s.idx, path: s.path });

  const add = (w: Wanted) => {
    let s = subs.find((x) => x.path === w.path);
    if (!s) {
      const r = attempt(() => streams.add_stream(w.path, 2.0));
      if (!r.ok) {
        post({ op: 'console', cls, lvl: 'warn', msg: `subscribe ${w.path}: ${r.error}` });
        return;
      }
      const decl = r.value as Msg;
      s = { idx: decl.index, path: w.path, deliver: false, kind: decl.kind };
      if (init.sab && w.sab_slot != null) {
        const base = w.sab_slot * 256;
        s.slot = w.sab_slot;
        s.lock = new Int32Array(init.sab, base, 2);
        s.meta = new Float64Array(init.sab, base + 8, 4);
        s.vals = new Float64Array(init.sab, base + 40, 24);
      }
      subs.push(s);
      sock.send(subscribeMsg(s)); // dropped if not open; the open handler resubscribes
    }
    s.deliver = w.deliver;
  };

  const onFrame = async (s: Sub, t0: number) => {
    // Copy the payload out of wasm memory into a pooled buffer, decode the image here (off the main
    // thread), then transfer both.
    const len = streams.data_len();
    let buf = pool.pop();
    if (!buf || buf.byteLength < len) buf = new ArrayBuffer(Math.max(len, 256 * 1024));
    const u8 = new Uint8Array(buf);
    streams.data_into(u8);
    const meta = {
      op: 'frame', path: s.path, seq: streams.seq(), epoch: streams.epoch(), t_capture_ms: streams.capture_ms(),
      t_publish_ms: streams.publish_ms(), kind: s.kind, width: streams.width(), height: streams.height(), len,
    };
    if (frameBusy) {
      framesSkipped++;
      pool.push(buf);
      return;
    }
    let bitmap: ImageBitmap | undefined;
    let decode_ms = 0;
    if (canBitmap) {
      frameBusy = true;
      const ti = performance.now();
      try {
        if (meta.kind === 'rgb8') {
          const rgba = new Uint8ClampedArray(meta.width * meta.height * 4);
          for (let i = 0, j = 0; i < len; i += 3, j += 4) {
            rgba[j] = u8[i] as number; rgba[j + 1] = u8[i + 1] as number; rgba[j + 2] = u8[i + 2] as number; rgba[j + 3] = 255;
          }
          bitmap = await createImageBitmap(new ImageData(rgba, meta.width, meta.height));
        } else {
          bitmap = await createImageBitmap(new Blob([u8.subarray(0, len)], { type: 'image/jpeg' }));
        }
      } finally {
        frameBusy = false;
      }
      decode_ms = performance.now() - ti;
      imgSum += decode_ms;
      imgN++;
    }
    if (poisoned) return;
    post({ ...meta, bitmap, buf, decode_ms, worker_ms: now() - t0 }, bitmap ? [bitmap, buf] : [buf]);
  };

  const deliverOne = (s: Sub, t: number) => {
    if (s.kind === 'jpeg' || s.kind === 'rgb8') {
      onFrame(s, t).catch((e) => (e instanceof WebAssembly.RuntimeError ? fatal(e) : post({ op: 'console', cls, lvl: 'warn', msg: `frame: ${e}` })));
      return;
    }
    const applied = streams.applied_mode();
    if (applied && applied !== s.edge) {
      // The edge mode, on every change (status.edge), whether or not anyone subscribed.
      s.edge = applied;
      post({ op: 'edge', path: s.path, mode: applied, token: streams.applied_token() });
    }
    if (s.kind === 'f64' && s.lock && s.meta && s.vals && streams.data_len() > 0) {
      // seqlock: odd while writing
      Atomics.add(s.lock, 0, 1);
      const n = streams.data_f64_into(s.vals);
      s.meta[0] = streams.seq(); s.meta[1] = streams.capture_ms(); s.meta[2] = streams.publish_ms(); s.meta[3] = n;
      Atomics.add(s.lock, 0, 1);
    }
    if (!s.deliver) return;
    const base = {
      op: 'sample', path: s.path, seq: streams.seq(), epoch: streams.epoch(), t_capture_ms: streams.capture_ms(),
      t_publish_ms: streams.publish_ms(),
      ...(applied ? { applied: { mode: applied, present: streams.applied_present(), token: streams.applied_token() } } : {}),
    };
    if (s.kind === 'text') {
      const text = streams.text();
      post({ ...base, text, data: new TextEncoder().encode(text) });
      return;
    }
    const tc = performance.now();
    const out = streams.data_f64();
    copySum += performance.now() - tc; copyN++;
    post({ ...base, data: out }, [out.buffer]);
  };

  const onMessage = (data: ArrayBuffer) => {
    const t = now();
    const u8 = new Uint8Array(data);
    let n: number;
    try {
      n = streams.push(u8, t); // JS -> wasm copy, wire decode, the session's receive rules
    } catch (e) {
      if (e instanceof WebAssembly.RuntimeError) throw e;
      return; // a malformed frame is an Err from the core, not a panic: drop it
    }
    const d = now() - t;
    decodeSum += d; decodeN++; if (d > decodeMax) decodeMax = d;
    for (let k = 0; k < n; k++) {
      if (k > 0) streams.advance();
      const s = subs.find((x) => x.idx === streams.index());
      if (s) deliverOne(s, t);
    }
  };

  const onText = (text: string) => {
    const m = parse(text);
    if (m?.type === 'error') post({ op: 'console', cls, lvl: 'warn', msg: `gateway: ${m.message}` });
  };

  // (Re)subscribe everything on every open: a reconnect starts a fresh gateway connection.
  const sock = openSocket(init, () => subs.forEach((s) => sock.send(subscribeMsg(s))), onMessage, onText);

  const stats = () => ({
    op: 'stats', cls, have_state: haveState, connected: sock.isOpen, reconnect_attempts: sock.attempts,
    streams: subs.map((s) => streams.stats(s.idx)),
    decode_us_mean: decodeN ? (1000 * decodeSum) / decodeN : 0, decode_us_max: 1000 * decodeMax, decoded: decodeN,
    copy_us_mean: copyN ? (1000 * copySum) / copyN : 0,
    image_decode_ms_mean: imgN ? imgSum / imgN : 0, frames_skipped: framesSkipped,
  });
  every(500, () => post(stats()));

  return (m: Msg) => {
    switch (m.op) {
      case 'state': {
        const r = attempt(() => streams.set_state(new Uint8Array(m.bytes)));
        if (!r.ok) break;
        haveState = true;
        for (const w of waiting.values()) add(w);
        waiting.clear();
        break;
      }
      case 'subscribe':
        if (haveState) add(m as unknown as Wanted);
        else waiting.set(m.path, m as unknown as Wanted);
        break;
      case 'clock':
        streams.set_clock(m.participant ?? '', m.offset_ms, m.error_ms ?? 0);
        break;
      case 'return':
        if (pool.length < 4) pool.push(m.buf);
        break;
      case 'reset_stats':
        streams.reset_stats(); decodeSum = decodeN = decodeMax = 0; copySum = copyN = 0; imgSum = imgN = 0; framesSkipped = 0;
        break;
      case 'stats':
        post({ ...stats(), op: 'reply', rid: m.rid });
        break;
      case 'drop_socket':
        sock.reconnectNow();
        break;
    }
  };
}

// ------------------------------------------------------------------ control

interface Drive {
  vals: Float64Array;
  hz: number;
  timer: ReturnType<typeof setInterval>;
  rate: { t: number; n: number; hz: number };
}

/** The request id the wasm stamped into a `REQUEST` frame (tag, then a u32 LE). */
const requestId = (b: Uint8Array) => (b[1]! | (b[2]! << 8) | (b[3]! << 16) | (b[4]! << 24)) >>> 0;

function runControl(init: InitMsg) {
  const cs = new ControlSession();
  let heartbeats = init.heartbeats ?? true;
  let attached = false;
  let mode = 'disarmed';
  let declsRevision = -1;
  let lastStatePost = 0;
  let hbCount = 0;
  const drives = new Map<string, Drive>();
  const verbs = new Map<number, number>(); // wasm request id -> main-thread rid
  let sent = 0, refused = 0, lastRefusal = '', hbSent = 0, gated = 0;
  let lastClockPost = 0;
  let hbTimer: ReturnType<typeof setTimeout> | undefined;
  cleanups.push(() => drives.forEach((d) => clearInterval(d.timer)), () => clearTimeout(hbTimer));

  const send = (b: Uint8Array | string | undefined | null) => (b ? sock.send(b) : false);

  const stopDrive = (path: string) => {
    const d = drives.get(path);
    if (d) clearInterval(d.timer);
    drives.delete(path);
  };

  // Disconnect: the gateway drops this connection's participant, so every lease it held ends at the
  // host (the edge holds, then goes safe). Locally the clients are dropped so the old tokens can never be
  // used again, and the main thread reports the leases as revoked. Nothing is resumed on reconnect: the
  // new connection is a new participant.
  const dropLeases = (reason: string) => {
    for (const path of cs.lease_paths()) {
      const token = cs.forget(path);
      stopDrive(path);
      post({ op: 'lease_lost', path, token, reason });
    }
    attached = false;
  };

  const postStatuses = () => {
    const decls = cs.declarations() as Msg;
    for (const c of decls.controls ?? []) {
      const r = attempt(() => cs.status(c.path));
      if (r.ok) post({ op: 'event', ev: { type: 'status', ...(r.value as Msg) } });
    }
  };

  const onState = (ev: Msg, bytes: Uint8Array) => {
    mode = ev.mode ?? mode;
    for (const ch of ev.changes ?? []) {
      if (ch.change === 'revoked') {
        stopDrive(ch.path);
        post({ op: 'event', ev: { type: 'lease', event: 'revoke', ours: true, path: ch.path, token: ch.token, reason: ch.reason, holder: '', priority: '' } });
      }
    }
    const t = performance.now();
    if (ev.revision !== declsRevision) {
      declsRevision = ev.revision;
      post({ op: 'decls', decls: cs.declarations(), state: { session: ev.session, revision: ev.revision, epoch: ev.epoch, mode: ev.mode, estop_latched: ev.estop_latched } });
      postStatuses();
    }
    // The data Workers need the declarations and the publishers' clock mappings: forward the host's
    // bytes on every structural change, and at most every 250 ms otherwise.
    if (bytes.length > 1 && (ev.structural || t - lastStatePost > 250)) {
      lastStatePost = t;
      const copy = bytes.slice(1);
      post({ op: 'state_bytes', bytes: copy.buffer }, [copy.buffer]);
    }
  };

  const onOpen = () => {
    // The hello attaches this connection's participant; the answer names it.
    sock.send(JSON.stringify({ type: 'hello', priority: init.priority ?? 'model' }));
  };

  const onText = (text: string) => {
    const m = parse(text);
    if (!m) return;
    switch (m.type) {
      case 'hello': {
        cs.set_participant(m.participant);
        attached = true;
        hbCount = 0;
        post({ op: 'socket', cls, state: 'attached', id: m.participant, session: m.session });
        heartbeat();
        break;
      }
      case 'fence':
        post({ op: 'event', ev: { type: 'fence', channel: m.channel, token: m.token, reason: m.reason } });
        break;
      case 'error':
        post({ op: 'console', cls, lvl: 'warn', msg: `gateway: ${m.message}` });
        break;
    }
  };

  const onMessage = (data: ArrayBuffer) => {
    const u8 = new Uint8Array(data);
    let ev: Msg;
    try {
      ev = cs.on_frame(u8, now()) as Msg;
    } catch (e) {
      if (e instanceof WebAssembly.RuntimeError) throw e;
      return;
    }
    switch (ev.type) {
      case 'state':
        onState(ev, u8);
        break;
      case 'clock': {
        const t = performance.now();
        if (ev.ours && t - lastClockPost > 200) {
          lastClockPost = t;
          post({ op: 'clock', clock: cs.clock(now()), participant: cs.participant() });
        }
        break;
      }
      case 'lease':
        post({ op: 'event', ev });
        if (ev.state) onState(ev.state, new Uint8Array(0));
        break;
      case 'reply': {
        if (ev.state) mode = ev.state.mode ?? mode;
        const rid = verbs.get(ev.rid);
        if (rid != null) {
          verbs.delete(ev.rid);
          post({ op: 'reply', rid, ok: ev.ok, error: ev.error ?? '' });
        }
        break;
      }
    }
    if (ev.type === 'lease' || ev.type === 'reply') postStatuses();
  };

  const sock = openSocket(init, onOpen, onMessage, onText, () => dropLeases('control socket disconnected'));

  // Heartbeats live in the Worker so a busy main thread cannot starve them: a quick burst after the
  // hello (the clock converges in ~100 ms), then every 50 ms. They also carry the clock exchange.
  const heartbeat = guard(() => {
    clearTimeout(hbTimer);
    if (attached && sock.isOpen && heartbeats && send(cs.heartbeat(now()))) hbSent++;
    hbCount++;
    hbTimer = setTimeout(heartbeat, hbCount < 8 ? 10 : 50);
  });

  every(250, () => {
    const t = now();
    const leases: Record<string, unknown> = {};
    for (const path of cs.lease_paths()) {
      const d = drives.get(path);
      if (d) {
        const tt = performance.now();
        d.rate.hz = (1000 * d.rate.n) / (tt - d.rate.t); d.rate.t = tt; d.rate.n = 0;
      }
      leases[path] = { state: cs.lease_state(path, t), token: cs.lease_token(path), remaining_ms: cs.lease_remaining_ms(path, t), hz: d?.rate.hz ?? 0, driving: !!d };
    }
    post({ op: 'lease_state', leases, cmd: { sent, refused, gated, lastRefusal, hbSent, heartbeats } });
  });

  const command = (path: string, vals: Float64Array, periodMs: number): boolean => {
    const r = attempt(() => cs.command(path, vals, periodMs, now()));
    if (!r.ok) {
      refused++;
      lastRefusal = r.error;
      return false;
    }
    const ok = send(r.value);
    if (ok) sent++;
    return ok;
  };

  const verb = (m: Msg, frame: () => Uint8Array) => {
    const r = attempt(frame);
    if (!r.ok) {
      post({ op: 'reply', rid: m.rid, ok: false, error: r.error });
      return;
    }
    verbs.set(requestId(r.value), m.rid);
    if (!send(r.value)) {
      verbs.delete(requestId(r.value));
      post({ op: 'reply', rid: m.rid, ok: false, error: 'control socket not connected' });
    }
  };

  return (m: Msg) => {
    switch (m.op) {
      case 'lease': {
        const r = attempt(() => cs.request_lease(m.path, m.ttl_ms ?? Number.NaN, now()));
        if (!r.ok || !sock.isOpen || !attached) {
          post({ op: 'event', ev: { type: 'lease', event: 'refused', ours: true, path: m.path, holder: '', priority: '', token: 0, reason: r.ok ? 'control socket not connected' : r.error } });
          break;
        }
        send(r.value);
        break;
      }
      case 'release': {
        stopDrive(m.path);
        const r = attempt(() => cs.release(m.path));
        if (r.ok) send(r.value);
        break;
      }
      case 'verb':
        verb(m, () => cs.verb(m.name, m.reason ?? ''));
        break;
      case 'send': {
        const ok = command(m.path, m.vals, m.period_ms ?? 20);
        post({ op: 'reply', rid: m.rid, ok, error: ok ? '' : lastRefusal });
        break;
      }
      case 'drive': {
        stopDrive(m.path);
        if (!m.vals) break;
        const d: Drive = { vals: m.vals, hz: m.hz, rate: { t: performance.now(), n: 0, hz: 0 }, timer: 0 as unknown as ReturnType<typeof setInterval> };
        const periodMs = 1000 / m.hz;
        // Commands flow only while the session is ACTIVE and the lease is held: streaming into SAFE or
        // an expired lease only produces refusals at the edge.
        d.timer = setInterval(guard(() => {
          const state = cs.lease_state(m.path, now());
          if (!sock.isOpen || mode !== 'active' || (state !== 'held' && state !== 'expiring')) { gated++; return; }
          if (command(m.path, d.vals, periodMs)) d.rate.n++;
        }), periodMs);
        drives.set(m.path, d);
        break;
      }
      case 'heartbeats':
        heartbeats = !!m.enabled;
        break;
      case 'stale_command': {
        const r = attempt(() => cs.command_with_token(m.path, m.vals, m.token, now()) as unknown as [Uint8Array, string]);
        if (!r.ok) {
          post({ op: 'console', cls, lvl: 'warn', msg: `stale command: ${r.error}` });
          break;
        }
        const [frame, verdict] = r.value;
        post({ op: 'event', ev: { type: 'fence', channel: 0, token: m.token, reason: verdict || 'passes the fence', local: true } });
        send(frame);
        break;
      }
      case 'drop_socket':
        sock.reconnectNow();
        break;
    }
  };
}

// ------------------------------------------------------------------ bootstrap

let handler: ((m: Msg) => void) | null = null;
port.listen(guard((m: Msg) => {
  if (m.op === 'init') {
    cls = m.cls;
    const t0 = performance.now();
    let exports: any;
    try {
      exports = initSync({ module: m.module });
    } catch (err) {
      port.post({ op: 'ready', cls, ok: false, error: String(err) });
      return;
    }
    const info = {
      op: 'ready', cls, ok: true, instantiate_ms: performance.now() - t0,
      typeof_WebAssembly: typeof WebAssembly,
      module_exports: WebAssembly.Module.exports(m.module).map((x) => x.name),
      instance_exports: Object.keys(exports).length,
      memory_bytes: exports.memory?.buffer?.byteLength,
      build: build_info(),
      web_time_skew_ms: web_time_now_ms() - Date.now(),
      cross_origin_isolated: (globalThis as any).crossOriginIsolated,
    };
    port.post(info);
    handler = cls === 'control' ? runControl(m as InitMsg) : runData(m as InitMsg);
    return;
  }
  if (m.op === 'trap') {
    // Debug: a real Rust panic in this wasm instance (panic=abort -> `unreachable` trap).
    std_instant_now_panics();
    return;
  }
  handler?.(m);
}));
