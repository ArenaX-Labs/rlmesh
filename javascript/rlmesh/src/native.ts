// Real-time sessions (protocol v2) in-process, over the native addon (Node >= 22, Electron, Bun):
// a host, robots and policies on Zenoh directly, the twin of Python's `rlmesh.v2`.
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

/** What a policy sends per action entry: `[horizon, ...]` elements, row-major. The dtype follows the typed array unless given. */
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
/** The binder refused a policy: the message is the refuse-and-explain text (§6.7). */
export class BindingRefused extends SessionException {
  override name = 'BindingRefused';
  /** `[code, 'input' | 'action', entry]` per problem. */
  problems: [string, string, string][] = [];
}
/** A policy start was skipped (§8.8). */
export class StartSkipped extends SessionException {
  override name = 'StartSkipped';
  /** `waiting_for`, `stale_input` or `window_gap`. */
  reason = '';
  waiting: string[] = [];
  stale: string[] = [];
  gaps: string[] = [];
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
  chunksSent: number; sendErrors: number;
  latencyP50Ms?: number | null; latencyP99Ms?: number | null; latencyMaxMs?: number | null;
  drifting: boolean;
}
interface NativeHost { readonly address: string; close(): Promise<void> }
interface NativeRobot {
  readonly name: string; readonly mode: string;
  parts(): NativePart[];
  publish(path: string, fields: (string | Uint8Array)[], captureNs?: bigint | null): number | null;
  due(path: string): NativeDue | null;
  handoverDone(path: string): void;
  arm(): Promise<void>; disarm(): Promise<void>; estop(reason: string): Promise<void>; clear(): Promise<void>;
  reportSafe(reason: string): void;
  nowNs(): bigint;
  edgeStats(path: string): NativeEdgeStats;
  close(): Promise<void>;
}
interface NativePolicy {
  readonly name: string; readonly table: string; readonly tableJson: string; readonly fingerprint: string;
  readonly warnings: string[]; readonly replanHz: number;
  observe(): NativeObservation;
  send(values: NativeEntry[], observation: NativeObservation): { entry: string; seq?: number | null }[];
  acquire(): Promise<{ path: string; token: bigint }[]>;
  release(): Promise<void>;
  holds(): boolean;
  run(predict: (obs: NativeObservation) => void, durationS?: number | null): Promise<NativeStats>;
  reply(values: NativeEntry[] | null, error: string | null): void;
  stop(): void;
  stats(): NativeStats;
  events(): Record<string, unknown>[];
  close(): Promise<void>;
}
interface NativeRate { sleep(): Promise<number>; readonly missed: number }
interface NativeRecorder { readonly path: string; close(): Promise<RecordStats | null> }
interface Addon {
  openHost(address: string, leaseGraceMs: number): Promise<NativeHost>;
  connectRobot(address: string, name: string, robotJson: string, timeoutS: number): Promise<NativeRobot>;
  bindPolicy(address: string, name: string, policyJson: string, robot: string, options: object): Promise<NativePolicy>;
  NativeRate: new (hz: number) => NativeRate;
  startRecorder(address: string, path: string, name: string, timeoutS: number): Promise<NativeRecorder>;
  readRecording(path: string): Promise<RecordingSummary>;
}

let addon: Addon | undefined;

/**
 * The addon for this platform: `RLMESH_NATIVE` if set, else `rlmesh.{platform}-{arch}.node` next to
 * this module. Loaded on first use, so `connect` (the gateway client) works without it.
 */
function native(): Addon {
  if (addon) return addon;
  const require = createRequire(import.meta.url);
  const file = process.env.RLMESH_NATIVE ?? new URL(`./rlmesh.${process.platform}-${process.arch}.node`, import.meta.url).pathname;
  try {
    addon = require(file) as Addon;
  } catch (e) {
    throw new SessionException(
      `the rlmesh native addon is not available for ${process.platform}-${process.arch} (${file}): ` +
        `${(e as Error).message}. Build it with \`mise run build:js:native\`, or join through a gateway with connect().`,
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

/** A session host in this process, listening on local TCP ports. Participants connect to its `address`. */
export class Host implements AsyncDisposable {
  private constructor(private native: NativeHost) {}

  /** The session `address` names (`[host/][session]`; `":0"`, the default: hosted here on an OS-assigned port). */
  static async open(opts: { address?: string; leaseGraceMs?: number } = {}): Promise<Host> {
    return new Host(await wait(() => native().openHost(opts.address ?? ':0', opts.leaseGraceMs ?? 250)));
  }

  /** `{ip}:{port}[/{session}]`: what participants connect to. */
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

type PublishValue = TypedArray | ArrayLike<number> | number | boolean | string;

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
  ) {
    for (const part of native.parts()) {
      this.kinds.set(part.path, part.kind);
      this.fields.set(part.path, part.fields);
    }
  }

  /** Attach as `name`, declare `parts` (the robot map, §3) and start its publishers and edges. */
  static async connect(
    name: string,
    parts: Record<string, unknown>,
    opts: { session: Host | string; timeout?: number; onHandover?: (group: string, from: string, to: string) => void },
  ): Promise<Robot> {
    const robot = await wait(() => native().connectRobot(address(opts.session), name, JSON.stringify(parts), opts.timeout ?? 10));
    return new Robot(robot, opts.onHandover);
  }

  /** The robot's participant id. */
  get name(): string {
    return this.native.name;
  }

  /** Each part's kind, by path, in authored order. */
  get parts(): Record<string, string> {
    return Object.fromEntries(this.kinds);
  }

  /** The session mode (`disarmed`, `armed`, `active`, `safe`, `estop`). */
  get mode(): string {
    return call(() => this.native.mode);
  }

  /**
   * Publish one sample of `path`; never blocks on the network. A single-field part takes the value
   * itself (`publish('wrist/cam', frame)`, `publish('task', 'pick')`); a multi-field part takes an
   * object with one key per field (`publish('arm', { pos, vel, effort })`). Values are cast to the
   * declared dtype when the cast is same-kind and must have the declared element count. Returns the
   * sample's seq, or null when it was coalesced or handed to the image encoder.
   */
  publish(path: string, value: PublishValue | Record<string, PublishValue>, opts: { captureNs?: bigint } = {}): number | null {
    const declared = this.fields.get(path);
    if (!declared) throw new RangeError(`${this.name} has no part '${path}' (its parts: ${[...this.fields.keys()].join(', ')})`);
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
    const buffers = declared.map((field) => {
      const item = values[field.name];
      if (field.dtype === 'utf8') {
        if (typeof item !== 'string') throw new TypeError(`${path} is text: publish a string`);
        return item;
      }
      if (typeof item === 'string') throw new TypeError(`${path}.${field.name}: publish numbers, not a string`);
      const elements = field.shape.reduce((n, d) => n * d, 1);
      return bytes(cast(`${path}.${field.name}`, item, field.dtype as DType, elements));
    });
    return call(() => this.native.publish(path, buffers, opts.captureNs ?? null));
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

  /** Emergency stop: latched until `clear`, then disarm and arm. */
  estop(reason = 'estop'): Promise<void> {
    return wait(() => this.native.estop(reason));
  }

  /** Release a latched e-stop (the robot's local clear). */
  clear(): Promise<void> {
    return wait(() => this.native.clear());
  }

  /** Tell the host the robot fell back on its own (a device watchdog fired). */
  reportSafe(reason: string): void {
    call(() => this.native.reportSafe(reason));
  }

  /** This robot's clock: `process.hrtime.bigint()`. */
  nowNs(): bigint {
    return call(() => this.native.nowNs());
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
  /** Each input's conservative age at the snapshot (null: blank). */
  agesNs: Record<string, bigint | null>;
  /** Inputs delivered stale under `allow: hold`. */
  stale: Set<string>;
  /** Window slots padded by `allow: warmup`, by entry. */
  padded: Record<string, number>;
  /** Inputs bound to a constant (`allow: blank`). */
  blank: Set<string>;
  /** The newest capture minus the oldest, across inputs. */
  skewNs: bigint;
  /** Per stream, `[streamId, seqFirst, seqLast]`: the chunk's `based_on`. */
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
      stale: new Set(native.stale),
      padded: Object.fromEntries(native.padded.map((p) => [p.entry, p.slots])),
      blank: new Set(native.blank),
      skewNs: native.skewNs,
      basedOn: native.basedOn.map((b) => [b.stream, b.seqFirst, b.seqLast]),
      timeNs: native.timeNs,
    };
  }
}

/** A binding table (§6.8): what each entry is bound to and how it is delivered. */
export class Binding {
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

/** Starts, completions, skips by reason, chunks sent and latency (§8.10). */
export interface RunStats {
  /** The input whose publication the starts are phased just after (wall clock). */
  alignedTo: string | null;
  /** How far after it, in ms. */
  phaseOffsetMs: number;
  starts: number;
  completions: number;
  skipped: Record<string, number>;
  chunksSent: number;
  sendErrors: number;
  latencyP50Ms: number | null;
  latencyP99Ms: number | null;
  latencyMaxMs: number | null;
  drifting: boolean;
}

function runStats(s: NativeStats): RunStats {
  return {
    alignedTo: s.alignedTo ?? null,
    phaseOffsetMs: s.phaseOffsetMs,
    starts: s.starts,
    completions: s.completions,
    skipped: Object.fromEntries(s.skipped.map((k) => [k.reason, k.count])),
    chunksSent: s.chunksSent,
    sendErrors: s.sendErrors,
    latencyP50Ms: s.latencyP50Ms ?? null,
    latencyP99Ms: s.latencyP99Ms ?? null,
    latencyMaxMs: s.latencyMaxMs ?? null,
    drifting: s.drifting,
  };
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
  /** Fills `{name}` segments (§6.4). */
  captures?: Record<string, string>;
}

/**
 * A policy map (§6) and, once bound, the controller that runs it. `priority` is the lease class it
 * attaches with: `model` (default), `operator` (preempts models: teleop), `safety` or `scripted`.
 */
export class Policy implements AsyncDisposable {
  private native: NativePolicy | null = null;
  private bound: Binding | null = null;
  private readonly policyJson: string;
  private readonly address: string;
  readonly name: string;
  private readonly priority: Priority;
  private readonly timeout: number;

  constructor(
    policy: Record<string, unknown>,
    opts: { session: Host | string; name?: string; priority?: Priority; timeout?: number },
  ) {
    this.policyJson = JSON.stringify(policy);
    this.address = address(opts.session);
    this.name = opts.name ?? 'policy';
    this.priority = opts.priority ?? 'model';
    this.timeout = opts.timeout ?? 10;
  }

  /** Attach, bind to `robot`'s parts and declare the binding. Rejects with `BindingRefused` (§6.7). */
  async bind(robot: string, opts: BindOptions = {}): Promise<Binding> {
    if (this.native) throw new SessionException(`policy ${this.name} is already bound`);
    const positional = Object.entries(opts.names ?? {}).map(([entry, mode]) => {
      if (mode !== 'positional') {
        throw new RangeError(`names: { ${entry}: '${mode}' }: only 'positional' is an override; joint name lists belong in the policy map`);
      }
      return entry;
    });
    this.native = await wait(() =>
      native().bindPolicy(this.address, this.name, this.policyJson, robot, {
        priority: this.priority,
        under: opts.under,
        at: opts.at,
        positional,
        allow: opts.allow,
        captures: opts.captures,
        timeoutS: this.timeout,
      }),
    );
    this.bound = new Binding(this.native);
    return this.bound;
  }

  private nat(): NativePolicy {
    if (!this.native) throw new SessionException(`policy ${this.name} is not bound: call bind() first`);
    return this.native;
  }

  /** The binding table (after `bind`). */
  get binding(): Binding {
    this.nat();
    return this.bound!;
  }

  /** A policy start now; throws `StartSkipped` when one is skipped (§8.8). */
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

  /** Take the binding's leases now (preempting a lower class), then keep them. Resolves to the token per part. */
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
   * Run the policy loop (§8.9) until `duration` seconds or `stop()`; resolves to the run's counters.
   * The loop runs on a native thread; `predict(obs)` runs on this one, at most at the replan rate
   * with one inference in flight (a start that falls while it runs is skipped, never queued). If
   * `predict` throws or rejects, the policy stops sending, releases its leases (the robot holds,
   * then goes safe) and the run rejects with that error.
   */
  async run(predict: Predict, opts: { duration?: number } = {}): Promise<RunStats> {
    const nat = this.nat();
    let failure: { error: unknown } | null = null;
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
      return runStats(await nat.run(start, opts.duration ?? null));
    } catch (e) {
      if (failure) throw (failure as { error: unknown }).error;
      return rethrow(e);
    }
  }

  /** Ask a `run()` in progress to return. */
  stop(): void {
    call(() => this.nat().stop());
  }

  /** Starts, completions, skips by reason, chunks sent and latency (§8.10). */
  stats(): RunStats {
    return runStats(call(() => this.nat().stats()));
  }

  /** Take the events reported so far: `rate_drift`, `inference_pending`. */
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
 * An MCAP recording of the session (§13), attached as an observer. Records every robot's streams
 * (`/{robot}/{path}`, with the edge's `/{robot}/{path}/_applied`), every action chunk sent to an
 * edge (`/{robot}/{path}/_chunks`), and the authority and binding changes (`/_session/authority`,
 * `/_session/bindings`), each envelope as it travelled. `close()` finishes the file.
 */
export class Recorder implements AsyncDisposable {
  /** The finished file's counters, once closed. */
  stats: RecordStats | null = null;

  private constructor(private native: NativeRecorder) {}

  static async start(path: string, opts: { session: Host | string; name?: string; timeout?: number }): Promise<Recorder> {
    return new Recorder(await wait(() => native().startRecorder(address(opts.session), path, opts.name ?? 'recorder', opts.timeout ?? 10)));
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

/** Bind `policy` to `robot` and run `predict` until `duration` (§6.4 options). Shorthand for `Policy` + `bind` + `run`. */
export async function run(
  policy: Record<string, unknown>,
  robot: string,
  predict: Predict,
  opts: BindOptions & { session: Host | string; name?: string; priority?: Priority; duration?: number; timeout?: number },
): Promise<RunStats> {
  await using p = new Policy(policy, opts);
  await p.bind(robot, opts);
  return p.run(predict, opts);
}
