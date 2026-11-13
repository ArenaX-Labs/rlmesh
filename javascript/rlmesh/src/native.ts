// Real-time sessions (protocol v2) in-process, over the native addon (Node >= 22, Electron, Bun):
// a host, robots and policies on Zenoh directly, the twin of Python's `rlmesh.realtime`.
//
//   import { Host, Robot, Policy, Rate } from 'rlmesh';
//   const host = await Host.open();  // Node 24: `await using host = ...` closes it for you
//   const robot = await Robot.connect('ur10e', UR10E, { session: host });
//   await robot.arm();
//   const policy = new Policy(HOLD, { session: host, name: 'hold' });
//   console.log(String(await policy.bind('ur10e')));               // the binding table
//   // robot loop: robot.publish('arm', { pos, vel, effort }); const due = robot.due('arm'); await rate.sleep();
//   await policy.run((obs) => ({ arm: { data: holdQ(obs), shape: [10, 6] } }), { duration: 5 });
//
// examples/node/hold.mjs runs this end to end.
//
// Every rule (normalisation, binding, conversions, leases, the edge, the run loop) runs in Rust;
// this module converts typed arrays at the boundary. Times are `bigint` ns on
// `process.hrtime.bigint()`'s clock. `publish` copies its values before returning.

import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import type { Priority } from './types.js';

/** A declared element type. */
export type DType =
  | 'bool' | 'uint8' | 'uint16' | 'uint32' | 'uint64'
  | 'int8' | 'int16' | 'int32' | 'int64'
  | 'float16' | 'float32' | 'float64';

export type TypedArray =
  | Uint8Array | Uint16Array | Uint32Array | BigUint64Array
  | Int8Array | Int16Array | Int32Array | BigInt64Array
  | Float32Array | Float64Array;

/** An array: its elements flat, row-major. `float16` arrives as a `Float16Array` where the runtime has one, else as raw `Uint16Array` bits. */
export interface Tensor {
  dtype: DType;
  shape: number[];
  data: TypedArray;
}

/** What a policy sends per action entry: `[chunk, ...]` elements, row-major. The dtype follows the typed array unless given. */
export interface ActionArray {
  data: TypedArray | ArrayLike<number>;
  shape: number[];
  dtype?: DType;
}

/** An input or action value: one array, several named ones, or text. */
export type InputValue = Tensor | Record<string, Tensor> | string;
export type Actions = Record<string, ActionArray | Record<string, ActionArray>>;
/** Called at the replan rate with one inference in flight; may be async. */
export type Predict = (obs: Observation) => Actions | Promise<Actions>;

/** A real-time session failed. */
export class SessionException extends Error {
  override name = 'SessionException';
}
/** The session host refused a verb; `status` names why (`LEASE_BUSY`, `NOT_READY`, ...). */
export class SessionRefused extends SessionException {
  override name = 'SessionRefused';
  status = '';
}
/** The binder refused a policy: the message is the refuse-and-explain text. */
export class BindingRefused extends SessionException {
  override name = 'BindingRefused';
  /** `[code, 'input' | 'action', entry]` per problem. */
  problems: [string, string, string][] = [];
}
/** A policy start was skipped. */
export class StartSkipped extends SessionException {
  override name = 'StartSkipped';
  /** `waiting_for`, `stale_input` or `window_gap`. */
  reason = '';
  waiting: string[] = [];
  stale: string[] = [];
  gaps: string[] = [];
}
/** The session is gone for a policy's run: its robot's heartbeats lapsed or the host is gone. */
export class SessionLost extends SessionException {
  override name = 'SessionLost';
  /** `robot_lost` or `host_lost`. */
  reason = '';
}
/** The host did not answer in time. */
export class SessionTimeout extends SessionException {
  override name = 'SessionTimeout';
}

// ---- the addon ------------------------------------------------------------------------------

interface NativeTensor { name?: string | null; dtype: string; shape: number[]; data: Uint8Array }
interface NativeEntry { entry: string; tensor?: NativeTensor | null; fields?: NativeTensor[] | null; text?: string | null }
interface NativeObservation {
  takeValues(): NativeEntry[];
  readonly timeNs: bigint;
  readonly ages: { entry: string; ageNs?: bigint | null }[];
  readonly stale: string[];
  readonly padded: { entry: string; slots: number }[];
  readonly blank: string[];
  readonly skewNs: bigint;
  readonly basedOn: { stream: number; seqFirst: number; seqLast: number }[];
  readonly seqs: { entry: string; seqFirst: number; seqLast: number }[];
}
interface NativeField { name: string; dtype: string; shape: number[] }
interface NativePart { path: string; kind: string; fields: NativeField[] }
interface NativeDue {
  path: string; mode: string; fields?: NativeTensor[] | null; safe?: string | null;
  handover?: { group: string; from: string; to: string } | null;
  leaseToken?: bigint | null; chunkSeq?: number | null; index?: number | null;
  receivedNs?: bigint | null; slotNs?: bigint | null;
  basedOn: { path: string; seqFirst: number; seqLast: number }[];
}
interface NativeEdgeStats {
  accepted: number; rejected: number; rejectedBy: { reason: string; count: number }[]; clamped: number; handovers: number; safeReports: number;
  modes: { mode: string; ticks: number; entries: number }[];
}
interface NativeStats {
  alignedTo?: string | null; phaseOffsetMs: number;
  starts: number; completions: number; skipped: { reason: string; count: number }[];
  chunksSent: number; shadowPredictions: number; sendErrors: number;
  latencyP50Ms?: number | null; latencyP99Ms?: number | null; latencyMaxMs?: number | null;
  drifting: boolean; parked: string[];
}
interface NativeHost { readonly address: string; close(): Promise<void> }
interface NativeRobot {
  readonly name: string; readonly mode: string; readonly modeReason: string; readonly leaving?: string | null;
  parts(): NativePart[];
  publish(path: string, fields: (string | Uint8Array)[], captureNs?: bigint | null): number | null;
  publishJpeg(path: string, jpeg: Uint8Array, captureNs?: bigint | null): void;
  due(path: string): NativeDue | null;
  handoverDone(path: string): void;
  arm(): Promise<void>; disarm(): Promise<void>; keep(label: string): Promise<void>;
  estop(reason: string): void; clearEstop(): void;
  reportSafe(reason: string): void;
  nowNs(): bigint;
  edgeStats(path: string): NativeEdgeStats;
  health(wait: boolean, timeoutS: number): Promise<NativeHealth>;
  close(): Promise<void>;
}
interface NativeHealth {
  epoch: number;
  hostNs: number;
  links: {
    participant: string; heartbeats: number; rttMs?: number | null; jitterMs: number; heartbeatGapMaxMs: number;
    lastHeartbeatAgeMs: number; offsetMs?: number | null; offsetErrorMs?: number | null; health: string;
    lastChunkAccepted: number; lastChunkApplied: number; queueDepth: number;
    rejectedBy: { reason: string; count: number }[];
    clockRefused: { part: string; participant: string; errorMs: number; boundMs: number }[];
    published: { part: string; published: number; coalesced: number }[];
    run?: {
      starts: number; completions: number; skipped: { reason: string; count: number }[]; chunksSent: number;
      pendingMs?: number | null; latencyP50Ms?: number | null; latencyP99Ms?: number | null; drifting: boolean;
    } | null;
    lastSeqSeen: { part: string; seq: number }[];
  }[];
}
interface NativePolicy {
  readonly name: string; readonly table: string; readonly tableJson: string; readonly fingerprint: string;
  publishedParts(): NativePart[];
  publish(path: string, fields: (string | Uint8Array)[], captureNs?: bigint | null): number | null;
  readonly warnings: string[]; readonly replanHz: number;
  observe(): NativeObservation;
  send(values: NativeEntry[], observation: NativeObservation): { entry: string; seq?: number | null }[];
  acquire(): Promise<{ path: string; token: bigint }[]>;
  release(): Promise<void>;
  holds(): boolean;
  run(
    predict: (obs: NativeObservation) => void,
    durationS?: number | null,
    mode?: RunMode | null,
    onStep?: ((step: NativeStep) => void) | null,
    resume?: Resume | null,
  ): Promise<NativeStats>;
  reply(values: NativeEntry[] | null, error: string | null): void;
  stop(): void;
  resume(): string[];
  health(wait: boolean, timeoutS: number): Promise<NativeHealth>;
  stats(): NativeStats;
  events(): Record<string, unknown>[];
  close(): Promise<void>;
}
interface NativeStep {
  readonly shadow: boolean;
  readonly timeNs: bigint;
  takeActions(): { entry: string; part?: string | null; seq?: number | null; actions?: NativeEntry | null }[];
}
interface NativeRate { sleep(): Promise<number>; readonly missed: number }
interface NativeRecorder { readonly path: string; close(): Promise<RecordStats | null> }
interface NativeScene {
  readonly run: string;
  address(): Promise<string | null>;
  status(): Promise<RunState>;
  waitUntil(stage: string, timeoutS: number): Promise<RunState>;
  arm(): Promise<RunState>;
  stop(timeoutS: number): Promise<RunState>;
  close(timeoutS: number): Promise<void>;
}
interface Addon {
  openHost(address: string, leaseGraceMs: number): Promise<NativeHost>;
  connectRobot(address: string, name: string, robotJson: string, timeoutS: number): Promise<NativeRobot>;
  // The lobby and scene runs: only in an addon built with the `fleet` feature (unreleased).
  robotFromLobby?(home: string | null, name: string, robotJson: string, timeoutS: number | null): Promise<NativeRobot>;
  runScene?(home: string | null, sceneToml: string, manualArm: boolean, timeoutS: number): Promise<NativeScene>;
  bindPolicy(address: string, name: string, policyJson: string, robot: string | null, options: object): Promise<NativePolicy>;
  NativeRate: new (hz: number) => NativeRate;
  startRecorder(address: string, path: string, name: string, timeoutS: number): Promise<NativeRecorder>;
  readRecording(path: string): Promise<RecordingSummary>;
}

let addon: Addon | undefined;

/**
 * The addon for this platform: `RLMESH_NATIVE` if set, else `rlmesh.{platform}-{arch}.node` next to
 * this module (a local build), else the `@rlmesh/{platform}-{arch}` package npm installs as an
 * optional dependency. Loaded on first use, so `connect` (the gateway client) works without it.
 */
function native(): Addon {
  if (addon) return addon;
  const require = createRequire(import.meta.url);
  const target = `${process.platform}-${process.arch}`;
  const local = new URL(`./rlmesh.${target}.node`, import.meta.url).pathname;
  const file = process.env.RLMESH_NATIVE ?? (existsSync(local) ? local : `@rlmesh/${target}`);
  try {
    addon = require(file) as Addon;
  } catch (e) {
    throw new SessionException(
      `the rlmesh native addon is not available for ${target} (${file}): ${(e as Error).message}. ` +
        `Reinstall rlmesh without --no-optional, build it with \`mise run build:js:native\`, or join through a gateway with connect().`,
    );
  }
  return addon;
}

const KIND_PREFIX = 'rlmesh:';

/** A native error as this module's class: its first line names the kind (lib.rs `error`). */
function rethrow(e: unknown): never {
  if (!(e instanceof Error) || !e.message.startsWith(KIND_PREFIX)) throw e;
  const newline = e.message.indexOf('\n');
  const kind = JSON.parse(e.message.slice(KIND_PREFIX.length, newline)) as Record<string, any>;
  const message = e.message.slice(newline + 1);
  let out: Error;
  switch (kind.kind) {
    case 'SessionRefused':
      out = Object.assign(new SessionRefused(message), { status: kind.status });
      break;
    case 'BindingRefused':
      out = Object.assign(new BindingRefused(message), { problems: kind.problems });
      break;
    case 'StartSkipped':
      out = Object.assign(new StartSkipped(message), {
        reason: kind.reason, waiting: kind.waiting, stale: kind.stale, gaps: kind.gaps,
      });
      break;
    case 'SessionLost':
      out = Object.assign(new SessionLost(message), { reason: kind.reason });
      break;
    case 'TimeoutError':
      out = new SessionTimeout(message);
      break;
    case 'ValueError':
      out = new RangeError(message);
      break;
    default:
      out = new SessionException(message);
  }
  throw out;
}

function call<T>(f: () => T): T {
  try {
    return f();
  } catch (e) {
    return rethrow(e);
  }
}

async function wait<T>(p: () => Promise<T>): Promise<T> {
  try {
    return await p();
  } catch (e) {
    return rethrow(e);
  }
}

// ---- arrays ---------------------------------------------------------------------------------

type ArrayCtor = { new (n: number): TypedArray; new (b: ArrayBufferLike, off: number, len: number): TypedArray; BYTES_PER_ELEMENT: number };
const Float16 = (globalThis as unknown as { Float16Array?: ArrayCtor }).Float16Array;
const CTORS: Record<DType, ArrayCtor> = {
  bool: Uint8Array, uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array, uint64: BigUint64Array as unknown as ArrayCtor,
  int8: Int8Array, int16: Int16Array, int32: Int32Array, int64: BigInt64Array as unknown as ArrayCtor,
  float16: Float16 ?? Uint16Array, float32: Float32Array, float64: Float64Array,
};
/** `f`loat, `i`nt, `u`nsigned or `b`ool, as NumPy's same-kind casting groups them. */
const KIND: Record<DType, 'b' | 'u' | 'i' | 'f'> = {
  bool: 'b', uint8: 'u', uint16: 'u', uint32: 'u', uint64: 'u',
  int8: 'i', int16: 'i', int32: 'i', int64: 'i', float16: 'f', float32: 'f', float64: 'f',
};

function dtypeOf(a: TypedArray): DType {
  if (a instanceof Float64Array) return 'float64';
  if (a instanceof Float32Array) return 'float32';
  if (Float16 && a instanceof Float16) return 'float16';
  if (a instanceof Int8Array) return 'int8';
  if (a instanceof Int16Array) return 'int16';
  if (a instanceof Int32Array) return 'int32';
  if (a instanceof BigInt64Array) return 'int64';
  if (a instanceof Uint16Array) return 'uint16';
  if (a instanceof Uint32Array) return 'uint32';
  if (a instanceof BigUint64Array) return 'uint64';
  return 'uint8';
}

function isTyped(v: unknown): v is TypedArray {
  return ArrayBuffer.isView(v) && !(v instanceof DataView);
}

/** The bytes of a typed array, without a copy. */
function bytes(a: TypedArray): Uint8Array {
  return new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
}

/** Native bytes as a typed array of `dtype`: a view when aligned, else one copy. */
function view(t: NativeTensor): Tensor {
  const dtype = t.dtype as DType;
  const Ctor = CTORS[dtype] ?? Uint8Array;
  const size = Ctor.BYTES_PER_ELEMENT;
  let buf = t.data;
  if (buf.byteOffset % size !== 0) buf = buf.slice();
  return { dtype, shape: t.shape, data: new Ctor(buf.buffer, buf.byteOffset, buf.byteLength / size) };
}

/** `values` as `dtype`, cast like NumPy's `same_kind` (a float never becomes an int). */
function cast(what: string, values: unknown, dtype: DType, elements: number): TypedArray {
  const Ctor = CTORS[dtype];
  let out: TypedArray;
  if (isTyped(values)) {
    const from = dtypeOf(values);
    if (from !== dtype) {
      const ok = KIND[dtype] === 'f' || KIND[from] === KIND[dtype] || (KIND[from] === 'u' && KIND[dtype] === 'i') || KIND[from] === 'b';
      if (!ok) throw new TypeError(`${what}: cannot publish ${from} as ${dtype}`);
    }
    out = from === dtype ? values : convert(Ctor, Array.from(values as ArrayLike<number | bigint>, Number), dtype);
  } else {
    const list = typeof values === 'number' || typeof values === 'boolean' ? [Number(values)] : Array.from(values as ArrayLike<number>, Number);
    if (KIND[dtype] !== 'f' && list.some((x) => !Number.isInteger(x))) {
      throw new TypeError(`${what}: cannot publish float64 as ${dtype}`);
    }
    out = convert(Ctor, list, dtype);
  }
  if (out.length !== elements) throw new RangeError(`${what}: ${out.length} elements, the part declares ${elements}`);
  return out;
}

function convert(Ctor: ArrayCtor, list: number[], dtype: DType): TypedArray {
  const out = new Ctor(list.length);
  const big = dtype === 'int64' || dtype === 'uint64';
  for (let i = 0; i < list.length; i++) (out as any)[i] = big ? BigInt(Math.trunc(list[i]!)) : list[i];
  return out;
}

function action(entry: string, a: ActionArray): NativeTensor {
  if (!a || !Array.isArray(a.shape)) throw new TypeError(`${entry}: an action is { data, shape }`);
  const dtype = a.dtype ?? (isTyped(a.data) ? dtypeOf(a.data) : 'float64');
  const elements = a.shape.reduce((n, d) => n * d, 1);
  const data = isTyped(a.data) && dtypeOf(a.data) === dtype ? a.data : convert(CTORS[dtype], Array.from(a.data as ArrayLike<number>, Number), dtype);
  if (data.length !== elements) throw new RangeError(`${entry}: ${data.length} elements for shape [${a.shape.join(', ')}]`);
  return { dtype, shape: a.shape, data: bytes(data) };
}

function actions(out: Actions): NativeEntry[] {
  if (!out || typeof out !== 'object') throw new TypeError('predict must return an object keyed by action name');
  return Object.entries(out).map(([entry, value]) =>
    'shape' in value && 'data' in value
      ? { entry, tensor: action(entry, value as ActionArray) }
      : { entry, fields: Object.entries(value).map(([name, a]) => ({ ...action(`${entry}.${name}`, a as ActionArray), name })) },
  );
}

function input(e: NativeEntry): InputValue {
  if (e.tensor) return view(e.tensor);
  if (e.fields) return Object.fromEntries(e.fields.map((f) => [f.name ?? '', view(f)]));
  return e.text ?? '';
}

// ---- the API --------------------------------------------------------------------------------

/** A session host in this process, listening on a local TCP port. Participants connect to its `address`. */
export class Host implements AsyncDisposable {
  private constructor(private native: NativeHost) {}

  /** The session `address` names (`[host/][session]`; `":0"`, the default: hosted here on an OS-assigned port). */
  static async open(opts: { address?: string; leaseGraceMs?: number } = {}): Promise<Host> {
    return new Host(await wait(() => native().openHost(opts.address ?? ':0', opts.leaseGraceMs ?? 250)));
  }

  /** `{ip}:{port}[/{session}]#{key}`: what participants connect to. The key is a credential (its holder attaches as device `local`). */
  get address(): string {
    return this.native.address;
  }

  /** Stop the host. */
  close(): Promise<void> {
    return wait(() => this.native.close());
  }

  [Symbol.asyncDispose](): Promise<void> {
    return this.close();
  }
}

function address(session: Host | string): string {
  return typeof session === 'string' ? session : session.address;
}

/** What the robot does on this control tick (returned by `Robot.due`). */
export interface Due {
  path: string;
  /** `driving`, `hold`, `safe`, `estop` or `idle`. */
  mode: string;
  /** The command to dispatch (one array, or one per field), validated and clamped; null when nothing is. */
  value: Tensor | Record<string, Tensor> | null;
  /** The declared safe behaviour to run now. */
  safe: string | null;
  /** `[group, fromPath, toPath]` on the tick a command-group handover starts. */
  handover: [string, string, string] | null;
  leaseToken: bigint | null;
  chunkSeq: number | null;
  index: number | null;
  /** When the edge received the chunk, and when this action's slot starts (the robot's clock). */
  receivedNs: bigint | null;
  slotNs: bigint | null;
  /** The input samples the chunk was computed from: `[path, seqFirst, seqLast]`. */
  basedOn: [string, number, number][];
}

/** An edge's counters: accepted, rejected and clamped commands, and `[ticks, entries]` per mode. */
export interface EdgeStats {
  accepted: number;
  rejected: number;
  /** The rejections by reason (`too_old`, `no_clock_mapping`, ...), those that occurred. */
  rejectedBy: Record<string, number>;
  clamped: number;
  handovers: number;
  safeReports: number;
  modes: Record<string, [number, number]>;
}

/**
 * The session's live link health, as the host last published it (once a second): `epoch`,
 * `hostNs` (the host's clock when it published) and one link per attached participant, keyed by
 * its name. Durations are in ms, as in `RunStats` and `EdgeStats`.
 */
export interface SessionHealth {
  epoch: number;
  hostNs: number;
  links: Record<string, LinkHealth>;
}

/** One participant's link as the host measures it. */
export interface LinkHealth {
  heartbeats: number;
  /** The latest heartbeat round trip (null before one). */
  rttMs: number | null;
  jitterMs: number;
  heartbeatGapMaxMs: number;
  lastHeartbeatAgeMs: number;
  /** Its clock mapping, host minus participant, and the mapping's error bound (null until mapped). */
  offsetMs: number | null;
  offsetErrorMs: number | null;
  /** What it reports of itself: `ok`, `degraded`, `unhealthy` or `unknown`. */
  health: string;
  /** Robots: the newest chunk seq accepted and applied (0: none), and the actions queued. */
  lastChunkAccepted: number;
  lastChunkApplied: number;
  queueDepth: number;
  /** Robots: chunks refused since it attached, in all and by reason (as `EdgeStats` names them). */
  rejected: number;
  rejectedBy: Record<string, number>;
  /** Robots: each part refusing chunks for a clock error. */
  clockRefused: { part: string; participant: string; errorMs: number; boundMs: number }[];
  /** Per part it publishes (`robot:path`): samples published and coalesced. */
  published: Record<string, { published: number; coalesced: number }>;
  /** A bound policy's counters from its latest report. */
  run: {
    starts: number;
    completions: number;
    skipped: Record<string, number>;
    chunksSent: number;
    pendingMs: number | null;
    latencyP50Ms: number | null;
    latencyP99Ms: number | null;
    drifting: boolean;
  } | null;
  /** The newest seq it received per part it reads (`robot:path`). */
  lastSeqSeen: Record<string, number>;
}

/** How `health()` reads. */
export interface HealthOptions {
  /** Wait for the host's next publication instead of returning the newest. */
  wait?: boolean;
  /** Seconds to wait (default: the object's timeout). */
  timeout?: number;
}

function sessionHealth(h: NativeHealth): SessionHealth {
  return {
    epoch: h.epoch,
    hostNs: h.hostNs,
    links: Object.fromEntries(
      h.links.map((l) => [
        l.participant,
        {
          heartbeats: l.heartbeats,
          rttMs: l.rttMs ?? null,
          jitterMs: l.jitterMs,
          heartbeatGapMaxMs: l.heartbeatGapMaxMs,
          lastHeartbeatAgeMs: l.lastHeartbeatAgeMs,
          offsetMs: l.offsetMs ?? null,
          offsetErrorMs: l.offsetErrorMs ?? null,
          health: l.health,
          lastChunkAccepted: l.lastChunkAccepted,
          lastChunkApplied: l.lastChunkApplied,
          queueDepth: l.queueDepth,
          rejected: l.rejectedBy.reduce((n, r) => n + r.count, 0),
          rejectedBy: Object.fromEntries(l.rejectedBy.map((r) => [r.reason, r.count])),
          clockRefused: l.clockRefused,
          published: Object.fromEntries(l.published.map((p) => [p.part, { published: p.published, coalesced: p.coalesced }])),
          run: l.run
            ? {
                starts: l.run.starts,
                completions: l.run.completions,
                skipped: Object.fromEntries(l.run.skipped.map((k) => [k.reason, k.count])),
                chunksSent: l.run.chunksSent,
                pendingMs: l.run.pendingMs ?? null,
                latencyP50Ms: l.run.latencyP50Ms ?? null,
                latencyP99Ms: l.run.latencyP99Ms ?? null,
                drifting: l.run.drifting,
              }
            : null,
          lastSeqSeen: Object.fromEntries(l.lastSeqSeen.map((n) => [n.part, n.seq])),
        },
      ]),
    ),
  };
}

type PublishValue = TypedArray | ArrayLike<number> | number | boolean | string;

/** One sample of `owner`'s part `path` as its fields' buffers, in declaration order. */
function sample(
  owner: string,
  parts: Map<string, NativeField[]>,
  path: string,
  value: PublishValue | Record<string, PublishValue>,
): (string | Uint8Array)[] {
  const declared = parts.get(path);
  if (!declared) throw new RangeError(`${owner} has no part '${path}' (its parts: ${[...parts.keys()].join(', ')})`);
  let values: Record<string, PublishValue>;
  if (value !== null && typeof value === 'object' && !isTyped(value) && !Array.isArray(value)) {
    values = value as Record<string, PublishValue>;
  } else {
    if (declared.length !== 1) {
      throw new TypeError(`${path} has fields ${declared.map((f) => f.name).join(', ')}: publish('${path}', { ${declared[0]!.name}: ... })`);
    }
    values = { [declared[0]!.name]: value as PublishValue };
  }
  const unknown = Object.keys(values).filter((k) => !declared.some((f) => f.name === k));
  const missing = declared.filter((f) => !(f.name in values)).map((f) => f.name);
  if (unknown.length || missing.length) {
    throw new TypeError(
      `${path}: fields ${declared.map((f) => f.name).join(', ')} (missing ${missing.join(', ') || 'none'}, unknown ${unknown.join(', ') || 'none'})`,
    );
  }
  return declared.map((field) => {
    const item = values[field.name];
    if (field.dtype === 'utf8') {
      if (typeof item !== 'string') throw new TypeError(`${path} is text: publish a string`);
      return item;
    }
    if (typeof item === 'string') throw new TypeError(`${path}.${field.name}: publish numbers, not a string`);
    const elements = field.shape.reduce((n, d) => n * d, 1);
    return bytes(cast(`${path}.${field.name}`, item, field.dtype as DType, elements));
  });
}

/**
 * One robot: its map declared, its parts published, its commands served. The robot's control loop
 * publishes its sensors and asks `due` what to dispatch on each control tick. `onHandover(group,
 * from, to)` runs when a command group's lease moves between members (switch the device's control
 * mode there); the group dispatches nothing until it returns.
 */
export class Robot implements AsyncDisposable {
  private fields = new Map<string, NativeField[]>();
  private kinds = new Map<string, string>();

  private constructor(
    private native: NativeRobot,
    private onHandover?: (group: string, from: string, to: string) => void,
    private readonly timeout: number = 10,
  ) {
    for (const part of native.parts()) {
      this.kinds.set(part.path, part.kind);
      this.fields.set(part.path, part.fields);
    }
  }

  /** Attach as `name`, declare `parts` (the robot map) and start its publishers and edges. */
  static async connect(
    name: string,
    parts: Record<string, unknown>,
    opts: { session: Host | string; timeout?: number; onHandover?: (group: string, from: string, to: string) => void },
  ): Promise<Robot> {
    const robot = await wait(() => native().connectRobot(address(opts.session), name, JSON.stringify(parts), opts.timeout ?? 10));
    return new Robot(robot, opts.onHandover, opts.timeout ?? 10);
  }

  /**
   * Wait in the home rlmeshd's lobby (`home`: `'rig-pi.local'`; default this machine's) until a
   * scene's run assigns this robot; it then joins the run and declares `parts`. A refused join (its
   * pin, another map) is reported to the daemon and the wait goes on. Run until `leaving` is set,
   * close, and wait again.
   */
  static async waitInLobby(
    name: string,
    parts: Record<string, unknown>,
    opts: { home?: string; timeout?: number; onHandover?: (group: string, from: string, to: string) => void } = {},
  ): Promise<Robot> {
    const robot = await wait(() => fleet('robotFromLobby')(opts.home ?? null, name, JSON.stringify(parts), opts.timeout ?? null));
    return new Robot(robot, opts.onHandover, opts.timeout ?? 10);
  }

  /** The robot's participant id. */
  get name(): string {
    return this.native.name;
  }

  /**
   * Why this robot should leave its scene's run: `scene_ended` or `scene_failed` when the run
   * closes, `robot_lost` when the host lost it (wait in the lobby again: its run re-sends its join),
   * `superseded` when another process took its name over (stop); `null` while it is in the run, and
   * outside a scene.
   */
  get leaving(): string | null {
    return call(() => this.native.leaving) ?? null;
  }

  /** Each part's kind, by path, in authored order. */
  get parts(): Record<string, string> {
    return Object.fromEntries(this.kinds);
  }

  /** The session mode (`disarmed`, `armed`, `active`, `safe`, `estop`). */
  get mode(): string {
    return call(() => this.native.mode);
  }

  /** Why the mode is what it is (`robot_lost`, `estop_local`, ...; `unspecified` before any change). */
  get modeReason(): string {
    return call(() => this.native.modeReason);
  }

  /**
   * Publish one sample of `path`; never blocks on the network. A single-field part takes the value
   * itself (`publish('wrist/cam', frame)`, `publish('task', 'pick')`); a multi-field part takes an
   * object with one key per field (`publish('arm', { pos, vel, effort })`). Values are cast to the
   * declared dtype when the cast is same-kind and must have the declared element count. Returns the
   * sample's seq, or null when it was coalesced or handed to the image encoder.
   */
  publish(path: string, value: PublishValue | Record<string, PublishValue>, opts: { captureNs?: bigint } = {}): number | null {
    const buffers = sample(this.name, this.fields, path, value);
    return call(() => this.native.publish(path, buffers, opts.captureNs ?? null));
  }

  /**
   * Publish one camera frame already encoded as JPEG, sent as it is (a camera's MJPEG output: no
   * decode, no re-encode). `path` is a camera declared `encoding: 'jpeg'`; the frame's header must
   * give its declared width and height. Never blocks on the network.
   */
  publishJpeg(path: string, jpeg: Uint8Array, opts: { captureNs?: bigint } = {}): void {
    call(() => this.native.publishJpeg(path, jpeg, opts.captureNs ?? null));
  }

  /**
   * Tick the edge serving `path` (and its command group): what to dispatch now. Call it once per
   * control tick per command group, at the part's control rate. Null when there is nothing to do.
   */
  due(path: string): Due | null {
    const d = call(() => this.native.due(path));
    if (!d) return null;
    const fields = d.fields?.map((f) => [f.name ?? '', view(f)] as const) ?? null;
    const due: Due = {
      path: d.path,
      mode: d.mode,
      value: fields === null ? null : fields.length === 1 ? fields[0]![1] : Object.fromEntries(fields),
      safe: d.safe ?? null,
      handover: d.handover ? [d.handover.group, d.handover.from, d.handover.to] : null,
      leaseToken: d.leaseToken ?? null,
      chunkSeq: d.chunkSeq ?? null,
      index: d.index ?? null,
      receivedNs: d.receivedNs ?? null,
      slotNs: d.slotNs ?? null,
      basedOn: d.basedOn.map((b) => [b.path, b.seqFirst, b.seqLast]),
    };
    if (due.handover) {
      this.onHandover?.(...due.handover);
      call(() => this.native.handoverDone(path));
    }
    return due;
  }

  /** DISARMED or SAFE to ARMED, in a fresh epoch. */
  arm(): Promise<void> {
    return wait(() => this.native.arm());
  }

  /** Back to DISARMED. */
  disarm(): Promise<void> {
    return wait(() => this.native.disarm());
  }

  /** Ask the host to keep its recording of the last minutes as an incident named `label` (at most every 5 s). */
  keep(label = 'keep'): Promise<void> {
    return wait(() => this.native.keep(label));
  }

  /**
   * Emergency stop, latched locally with no network (call it from anywhere, a button handler
   * too): the next `due()` purges every queue and returns `estop`, and the host is told in the
   * background. Latched until `clearEstop()`, then disarm and arm.
   */
  estop(reason = 'estop'): void {
    call(() => this.native.estop(reason));
  }

  /** Clear the local e-stop latch and tell the host (after any unsent e-stop); then disarm and arm. */
  clearEstop(): void {
    call(() => this.native.clearEstop());
  }

  /** Tell the host the robot fell back on its own (a device watchdog fired). */
  reportSafe(reason: string): void {
    call(() => this.native.reportSafe(reason));
  }

  /** This robot's clock: `process.hrtime.bigint()`. */
  nowNs(): bigint {
    return call(() => this.native.nowNs());
  }

  /**
   * The session's live link health (`SessionHealth`). The first call subscribes and waits for the
   * host's first publication; later calls resolve to the newest at once, and `{ wait: true }` to
   * the next one.
   */
  health(opts: HealthOptions = {}): Promise<SessionHealth> {
    return wait(async () => sessionHealth(await this.native.health(opts.wait ?? false, opts.timeout ?? this.timeout)));
  }

  /** The counters of the edge serving `path`. */
  edgeStats(path: string): EdgeStats {
    const s = call(() => this.native.edgeStats(path));
    return {
      accepted: s.accepted,
      rejected: s.rejected,
      rejectedBy: Object.fromEntries(s.rejectedBy.map((r) => [r.reason, r.count])),
      clamped: s.clamped,
      handovers: s.handovers,
      safeReports: s.safeReports,
      modes: Object.fromEntries(s.modes.map((m) => [m.mode, [m.ticks, m.entries]])),
    };
  }

  /** Detach from the session. */
  close(): Promise<void> {
    return wait(() => this.native.close());
  }

  [Symbol.asyncDispose](): Promise<void> {
    return this.close();
  }
}

/** A snapshot's metadata (`obs.meta`): ages, staleness and provenance. */
export interface Meta {
  /** Each input's conservative age at the snapshot, by entry name (null: blank). */
  agesNs: Record<string, bigint | null>;
  /** Each input's seq range delivered, `[seqFirst, seqLast]`, by entry name (absent: blank). Entry names hold across a rebind; stream ids do not. */
  seqs: Record<string, [number, number]>;
  /** Inputs delivered stale under `allow: hold`. */
  stale: Set<string>;
  /** Window slots padded by `allow: warmup`, by entry. */
  padded: Record<string, number>;
  /** Inputs bound to a constant (`allow: blank`). */
  blank: Set<string>;
  /** The newest capture minus the oldest, across inputs. */
  skewNs: bigint;
  /** Per stream, `[streamId, seqFirst, seqLast]`: the chunk's `based_on`, by the ids of the binding in force (`policy.binding`). */
  basedOn: [number, number, number][];
  /** The session time the snapshot was taken at. */
  timeNs: bigint;
}

/** What `predict` receives: each input by entry name, plus `meta`. Arrays stay valid while you keep them. */
export class Observation {
  readonly inputs: Record<string, InputValue>;
  readonly meta: Meta;

  /** @internal */
  constructor(readonly native: NativeObservation) {
    this.inputs = Object.fromEntries(native.takeValues().map((e) => [e.entry, input(e)]));
    this.meta = {
      agesNs: Object.fromEntries(native.ages.map((a) => [a.entry, a.ageNs ?? null])),
      seqs: Object.fromEntries(native.seqs.map((s) => [s.entry, [s.seqFirst, s.seqLast]])),
      stale: new Set(native.stale),
      padded: Object.fromEntries(native.padded.map((p) => [p.entry, p.slots])),
      blank: new Set(native.blank),
      skewNs: native.skewNs,
      basedOn: native.basedOn.map((b) => [b.stream, b.seqFirst, b.seqLast]),
      timeNs: native.timeNs,
    };
  }
}

/**
 * What `Policy.run` does with a start it cannot command (no lease held: the session not armed,
 * preempted, its robot held). `'command'` (the default) skips it (`no_lease`). `'shadow'` still
 * calls `predict` and converts and checks its actions, then drops them: nothing reaches a robot.
 * A shadow run commands from the first start whose inputs were read with a lease held.
 */
export type RunMode = 'command' | 'shadow';

/**
 * What `Policy.run` does once a higher class took a part from it (an operator preempting it) and
 * let go. `'manual'` (the default): the part stays parked, and the robot's part goes to its safe
 * behaviour, until `policy.resume()`; a robot never moves again because someone released a
 * controller or their process died. `'auto'`: it is asked for again as soon as it is free.
 */
export type Resume = 'manual' | 'auto';

/** A completed start, as `onStep` receives it. */
export interface Step {
  /** A shadow start: its actions were checked and not sent. */
  shadow: boolean;
  /** The session time of the start's snapshot. */
  timeNs: bigint;
  /** Each action entry's chunk as the part's edge receives it: one `[chunk, ...]` array per command field (null: dropped). */
  actions: Record<string, Record<string, Tensor> | null>;
  /** The part each action entry commands, `robot:path` (null: dropped). */
  parts: Record<string, string | null>;
  /** Each entry's chunk seq when it was sent (null: not sent). */
  seqs: Record<string, number | null>;
}

function step(native: NativeStep): Step {
  const entries = native.takeActions();
  return {
    shadow: native.shadow,
    timeNs: native.timeNs,
    actions: Object.fromEntries(entries.map((e) => [e.entry, e.actions ? (input(e.actions) as Record<string, Tensor>) : null])),
    parts: Object.fromEntries(entries.map((e) => [e.entry, e.part ?? null])),
    seqs: Object.fromEntries(entries.map((e) => [e.entry, e.seq ?? null])),
  };
}

/** A binding table: what each entry is bound to and how it is delivered. */
export class BindingTable {
  /** The human table, as the cockpit shows it. */
  readonly table: string;
  /** The binding fingerprint (16 hex digits). */
  readonly fingerprint: string;
  /** The machine-readable table the fingerprint is taken over. */
  readonly json: Record<string, any>;
  /** Warnings the binding was accepted with. */
  readonly warnings: string[];
  /** The replan rate R. */
  readonly replanHz: number;

  /** @internal */
  constructor(native: NativePolicy) {
    this.table = native.table;
    this.fingerprint = native.fingerprint;
    this.json = JSON.parse(native.tableJson);
    this.warnings = native.warnings;
    this.replanHz = native.replanHz;
  }

  toString(): string {
    return this.table;
  }
}

/** Starts, completions, skips by reason, chunks sent and latency. */
export interface RunStats {
  /** The input whose publication the starts are phased just after (wall clock). */
  alignedTo: string | null;
  /** How far after it, in ms. */
  phaseOffsetMs: number;
  starts: number;
  completions: number;
  skipped: Record<string, number>;
  chunksSent: number;
  /** Starts a shadow run predicted without a lease (not `no_lease` skips). */
  shadowPredictions: number;
  sendErrors: number;
  latencyP50Ms: number | null;
  latencyP99Ms: number | null;
  latencyMaxMs: number | null;
  drifting: boolean;
  /** The parts a higher class took, `robot:path`, kept from the policy until `resume()`. */
  parked: string[];
}

function runStats(s: NativeStats): RunStats {
  return {
    alignedTo: s.alignedTo ?? null,
    phaseOffsetMs: s.phaseOffsetMs,
    starts: s.starts,
    completions: s.completions,
    skipped: Object.fromEntries(s.skipped.map((k) => [k.reason, k.count])),
    chunksSent: s.chunksSent,
    shadowPredictions: s.shadowPredictions,
    sendErrors: s.sendErrors,
    latencyP50Ms: s.latencyP50Ms ?? null,
    latencyP99Ms: s.latencyP99Ms ?? null,
    latencyMaxMs: s.latencyMaxMs ?? null,
    drifting: s.drifting,
    parked: s.parked,
  };
}

/** How one `Policy.run` runs. */
export interface RunOptions {
  /** Stop after this many seconds. */
  duration?: number;
  /** What a start that cannot command does (default `'command'`). */
  mode?: RunMode;
  /** What a part a higher class took does once it is free (default `'manual'`). */
  resume?: Resume;
  /** Called after each completed start with its actions, marked shadow or sent. */
  onStep?: (step: Step) => void;
}

export interface BindOptions {
  /** Scopes relative selectors. */
  under?: string;
  /** Overrides an entry's path. */
  at?: Record<string, string>;
  /** `{ entry: 'positional' }` binds joints by index. */
  names?: Record<string, 'positional'>;
  /** Conversions allowed for every entry. */
  allow?: string[];
  /** Fills `{name}` segments. */
  captures?: Record<string, string>;
}

/**
 * A policy map and, once bound, the controller that runs it. `priority` is the lease class it
 * attaches with: `model` (default), `operator` (preempts models: teleop), `safety` or `scripted`.
 */
export class Policy implements AsyncDisposable {
  private native: NativePolicy | null = null;
  private readonly policyJson: string;
  // A true private field: the address carries the session's key, which an inspected Policy must not show.
  readonly #address: string;
  readonly name: string;
  private readonly priority: Priority;
  private readonly timeout: number;
  private readonly publishesJson: string | undefined;
  private fields = new Map<string, NativeField[]>();

  /**
   * `publishes` declares parts of the policy's own to `publish` annotations on, in the robot map's
   * form, none commandable: `{ target: { kind: 'x/target', hz: 30, fields: { xy: ['float32',
   * [2]] } } }`. They appear under the policy's name (`tracker/target`): recorded, shown in the
   * cockpit, readable by other policies, and never bound to this one.
   */
  constructor(
    policy: Record<string, unknown>,
    opts: {
      session: Host | string;
      name?: string;
      priority?: Priority;
      timeout?: number;
      publishes?: Record<string, unknown>;
    },
  ) {
    this.policyJson = JSON.stringify(policy);
    this.publishesJson = opts.publishes ? JSON.stringify(opts.publishes) : undefined;
    this.#address = address(opts.session);
    this.name = opts.name ?? 'policy';
    this.priority = opts.priority ?? 'model';
    this.timeout = opts.timeout ?? 10;
  }

  /**
   * Attach, bind to `robot`'s parts and declare the binding. Rejects with `BindingRefused`.
   * With no robot it binds the session's view of every robot, each under its name, so one policy
   * drives several: its actions name their robot (`eye-1:eyes`).
   */
  async bind(robot?: string | null, opts: BindOptions = {}): Promise<BindingTable> {
    if (this.native) throw new SessionException(`policy ${this.name} is already bound`);
    const positional = Object.entries(opts.names ?? {}).map(([entry, mode]) => {
      if (mode !== 'positional') {
        throw new RangeError(`names: { ${entry}: '${mode}' }: only 'positional' is an override; joint name lists belong in the policy map`);
      }
      return entry;
    });
    this.native = await wait(() =>
      native().bindPolicy(this.#address, this.name, this.policyJson, robot ?? null, {
        priority: this.priority,
        under: opts.under,
        at: opts.at,
        positional,
        allow: opts.allow,
        captures: opts.captures,
        timeoutS: this.timeout,
        publishesJson: this.publishesJson,
      }),
    );
    for (const part of this.native.publishedParts()) this.fields.set(part.path, part.fields);
    return new BindingTable(this.native);
  }

  /**
   * Publish one sample of one of the policy's own parts (`publishes`), after `bind`. Takes values
   * the way `Robot.publish` does, and never blocks on the network. Returns the sample's seq, or
   * null when it was coalesced.
   */
  publish(path: string, value: PublishValue | Record<string, PublishValue>, opts: { captureNs?: bigint } = {}): number | null {
    const native = this.nat();
    if (this.fields.size === 0) throw new RangeError(`policy ${this.name} publishes nothing: pass publishes`);
    const buffers = sample(this.name, this.fields, path, value);
    return call(() => native.publish(path, buffers, opts.captureNs ?? null));
  }

  private nat(): NativePolicy {
    if (!this.native) throw new SessionException(`policy ${this.name} is not bound: call bind() first`);
    return this.native;
  }

  /** The binding table in force (after `bind`): after a `rebound` event, the new one. */
  get binding(): BindingTable {
    return new BindingTable(this.nat());
  }

  /** A policy start now; throws `StartSkipped` when one is skipped. */
  observe(): Observation {
    return new Observation(call(() => this.nat().observe()));
  }

  /**
   * Send one timed chunk per action entry, computed from `observation`. Returns each entry's chunk
   * seq, or null when this policy does not hold the part's lease.
   */
  send(out: Actions, observation: Observation): Record<string, number | null> {
    const sent = call(() => this.nat().send(actions(out), observation.native));
    return Object.fromEntries(sent.map((s) => [s.entry, s.seq ?? null]));
  }

  /** Take the binding's leases now (preempting a lower class), then keep them. Resolves to the token per part, keyed `robot:path`. */
  async acquire(): Promise<Record<string, bigint>> {
    const leases = await wait(() => this.nat().acquire());
    return Object.fromEntries(leases.map((l) => [l.path, l.token]));
  }

  /** Give the leases back; they are not re-requested until `acquire()` or `run()`. */
  release(): Promise<void> {
    return wait(() => this.nat().release());
  }

  /** Whether this policy holds every lease its binding commands. */
  get holds(): boolean {
    return call(() => this.nat().holds());
  }

  /**
   * Run the policy loop until `duration` seconds or `stop()`; resolves to the run's counters.
   * The loop runs on a native thread; `predict(obs)` runs on this one, at most at the replan rate
   * with one inference in flight (a start that falls while it runs is skipped, never queued). If
   * `predict` throws or rejects, the policy stops sending, releases its leases (the robot holds,
   * then goes safe) and the run rejects with that error.
   *
   * With `mode: 'shadow'` a start that cannot command still calls `predict` (its own published
   * parts keep flowing) and checks its actions, sending nothing until the policy holds a lease
   * (`shadowing` and `commanding` events). `onStep(step)` is called after each completed start with
   * its actions as the edges receive them, marked `shadow`; if it throws, the run stops and
   * rejects with that error.
   */
  async run(predict: Predict, opts: RunOptions = {}): Promise<RunStats> {
    const nat = this.nat();
    const mode = opts.mode ?? 'command';
    if (mode !== 'command' && mode !== 'shadow') throw new RangeError(`mode ${String(mode)}: 'command' or 'shadow'`);
    const resume = opts.resume ?? 'manual';
    if (resume !== 'manual' && resume !== 'auto') throw new RangeError(`resume ${String(resume)}: 'manual' or 'auto'`);
    let failure: { error: unknown } | null = null;
    const onStep = opts.onStep;
    const stepped = onStep
      ? (raw: NativeStep) => {
          if (failure) return;
          try {
            onStep(step(raw));
          } catch (error) {
            failure = { error };
            try {
              nat.stop();
            } catch {
              // The run already ended.
            }
          }
        }
      : null;
    const start = (raw: NativeObservation) => {
      void (async () => {
        try {
          const out = actions(await predict(new Observation(raw)));
          nat.reply(out, null);
        } catch (error) {
          failure ??= { error };
          try {
            nat.reply(null, String((error as Error)?.message ?? error));
          } catch {
            // The run already ended.
          }
        }
      })();
    };
    try {
      const stats = runStats(await nat.run(start, opts.duration ?? null, mode, stepped, resume));
      if (failure) throw (failure as { error: unknown }).error;
      return stats;
    } catch (e) {
      if (failure) throw (failure as { error: unknown }).error;
      return rethrow(e);
    }
  }

  /** Ask a `run()` in progress to return. */
  stop(): void {
    call(() => this.nat().stop());
  }

  /**
   * Hand the parts a higher class took back to this policy (a `run()` in progress asks for them
   * again, once their holder lets go); returns them, `robot:path`.
   */
  resume(): string[] {
    return call(() => this.nat().resume());
  }

  /**
   * The session's live link health (`SessionHealth`). The first call subscribes and waits for the
   * host's first publication; later calls resolve to the newest at once, and `{ wait: true }` to
   * the next one. A policy that only reads reads it too.
   */
  health(opts: HealthOptions = {}): Promise<SessionHealth> {
    return wait(async () => sessionHealth(await this.nat().health(opts.wait ?? false, opts.timeout ?? this.timeout)));
  }

  /** Starts, completions, skips by reason, chunks sent and latency. */
  stats(): RunStats {
    return runStats(call(() => this.nat().stats()));
  }

  /** Take the events reported so far: `rate_drift`, `inference_pending`, `robot_held`, `clock_refused`, `waiting_to_arm`, `shadowing`, `commanding`, `input_lost`, `rebound`, `parked` (with `part` and `by`). */
  events(): Record<string, unknown>[] {
    return call(() => this.nat().events()).map((e) => Object.fromEntries(Object.entries(e).filter(([, v]) => v != null)));
  }

  /** Release the leases and detach. */
  async close(): Promise<void> {
    if (this.native) await wait(() => this.native!.close());
  }

  [Symbol.asyncDispose](): Promise<void> {
    return this.close();
  }
}

/**
 * A fixed-rate loop timer on `process.hrtime.bigint()` (a robot's control loop): `sleep()` resolves
 * when the next period starts, waiting off the JS thread; a call more than a period late skips the
 * missed periods instead of bursting.
 */
export class Rate {
  private native: NativeRate;

  constructor(hz: number) {
    this.native = call(() => new (native().NativeRate)(hz));
  }

  /** Wait for the next period; resolves to how late this call was, in ns. */
  sleep(): Promise<number> {
    return wait(() => this.native.sleep());
  }

  /** Periods skipped so far. */
  get missed(): number {
    return this.native.missed;
  }
}

/** A finished recording's counters. */
export interface RecordStats {
  path: string;
  messages: number;
  channels: number;
  bytes: number;
  logTimeClamped: number;
}

/** A recording read back (`readRecording`). */
export interface RecordingSummary {
  messages: number;
  /** Metadata record names (`rlmesh.robot.{name}`, ...). */
  metadata: string[];
  /** Per topic: its schema, message count, log-time span and ordering, undecodable messages. */
  topics: {
    topic: string;
    schema: string;
    channels: number;
    messages: number;
    firstLogNs: bigint;
    lastLogNs: bigint;
    logTimeMonotonic: boolean;
    publishTimeMonotonic: boolean;
    undecodable: number;
  }[];
  /** Every lease change, in order. */
  leases: { logNs: bigint; part: string; holder: string | null; priority: string | null; token: bigint | null }[];
}

/**
 * An MCAP recording of the session, attached as an observer. Records every robot's streams
 * (`/{robot}/{path}`, with the edge's `/{robot}/{path}/_applied`), every action chunk sent to an
 * edge (`/{robot}/{path}/_chunks`), and the authority and binding changes (`/_session/authority`,
 * `/_session/bindings`), each envelope as it travelled. `close()` finishes the file.
 */
export class SessionRecorder implements AsyncDisposable {
  /** The finished file's counters, once closed. */
  stats: RecordStats | null = null;

  private constructor(private native: NativeRecorder) {}

  static async start(path: string, opts: { session: Host | string; name?: string; timeout?: number }): Promise<SessionRecorder> {
    return new SessionRecorder(await wait(() => native().startRecorder(address(opts.session), path, opts.name ?? 'recorder', opts.timeout ?? 10)));
  }

  /** The MCAP file. */
  get path(): string {
    return this.native.path;
  }

  /** Finish the file; resolves to its counters (also kept in `stats`). */
  async close(): Promise<RecordStats | null> {
    const stats = await wait(() => this.native.close());
    if (stats) this.stats = stats;
    return this.stats;
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.close();
  }
}

/** A refusal in a run's state. */
export interface Refusal {
  status: string;
  message: string;
}

/** A scene run's state: its stage, each member's, and why it failed. */
export interface RunState {
  run: string;
  version: number;
  stage: string;
  members: { name: string; stage: string; error: Refusal | null }[];
  error: Refusal | null;
}

/** A native run state, its absent errors `null`. */
function runState(s: RunState): RunState {
  return { ...s, members: s.members.map((m) => ({ ...m, error: m.error ?? null })), error: s.error ?? null };
}

/**
 * A scene's run on the home rlmeshd, started by `runScene`. Its members join `address()` (this
 * script's own policy among them). `close()` stops the run unless it ended.
 */
export class Scene implements AsyncDisposable {
  constructor(private native: NativeScene) {}

  /** The run's id (`r-7k2m`), as `rlmesh scene stop` takes it. */
  get run(): string {
    return this.native.run;
  }

  /** Where the run's members join, once its host is up. */
  address(): Promise<string | null> {
    return wait(async () => (await this.native.address()) ?? null);
  }

  /** The run's state now. */
  status(): Promise<RunState> {
    return wait(async () => runState(await this.native.status()));
  }

  /** Resolve once the run is at `stage` (`'running'`); a run that ends first rejects with its error. */
  waitUntil(stage: string, opts: { timeout?: number } = {}): Promise<RunState> {
    return wait(async () => runState(await this.native.waitUntil(stage, opts.timeout ?? 30)));
  }

  /** Arm a run that waits for it (`arm: 'manual'`, or after a lost member rejoined). */
  arm(): Promise<RunState> {
    return wait(async () => runState(await this.native.arm()));
  }

  /** Stop the run and wait for it to end; this process's policies' runs return. */
  stop(opts: { timeout?: number } = {}): Promise<RunState> {
    return wait(async () => runState(await this.native.stop(opts.timeout ?? 10)));
  }

  /** Stop the run unless it ended, and leave the lobby. */
  close(): Promise<void> {
    return wait(() => this.native.close(10));
  }

  [Symbol.asyncDispose](): Promise<void> {
    return this.close();
  }
}

/**
 * Run the scene file at `path` on its robots' home rlmeshd (`home`; default this machine's). The
 * daemon checks it, claims its robots, assigns them from its lobby and arms once every member
 * declared and bound (`arm: 'manual'`: on `Scene.arm`). A refused scene rejects with
 * `SessionRefused` and its code (`INVALID_REQUEST`, `NOT_AUTHORIZED`, `ROBOT_BUSY`).
 */
export async function runScene(
  path: string,
  opts: { home?: string; arm?: 'auto' | 'manual'; timeout?: number } = {},
): Promise<Scene> {
  const arm = opts.arm ?? 'auto';
  if (arm !== 'auto' && arm !== 'manual') throw new RangeError(`arm ${String(arm)}: auto or manual`);
  const text = await readFile(path, 'utf8');
  return new Scene(await wait(() => fleet('runScene')(opts.home ?? null, text, arm === 'manual', opts.timeout ?? 10)));
}

/** A lobby or scene call of the addon, which only a `fleet` build has (unreleased). */
function fleet<K extends 'robotFromLobby' | 'runScene'>(name: K): NonNullable<Addon[K]> {
  const addon = native();
  const f = addon[name];
  if (f === undefined) throw new Error(`${name}: this rlmesh build has no lobby or scenes (the unreleased fleet feature)`);
  return f.bind(addon) as NonNullable<Addon[K]>;
}

/** Read a recording back: messages per topic, time ordering, the lease timeline. */
export function readRecording(path: string): Promise<RecordingSummary> {
  return wait(async () => {
    const s = await native().readRecording(path);
    return {
      ...s,
      leases: s.leases.map((l) => ({ ...l, holder: l.holder ?? null, priority: l.priority ?? null, token: l.token ?? null })),
    };
  });
}

/** Bind `policy` to `robot` (`null`: every robot's view) and drive it with `predict` until `duration`. Shorthand for `Policy` + `bind` + `run`. */
export async function drive(
  policy: Record<string, unknown>,
  robot: string | null,
  predict: Predict,
  opts: BindOptions & RunOptions & {
    session: Host | string;
    name?: string;
    priority?: Priority;
    timeout?: number;
    publishes?: Record<string, unknown>;
  },
): Promise<RunStats> {
  await using p = new Policy(policy, opts);
  await p.bind(robot, opts);
  return p.run(predict, opts);
}
