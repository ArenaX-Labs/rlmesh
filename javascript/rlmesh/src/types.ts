// Public types of the `rlmesh` package. Names at this boundary carry no wire prefixes: modes,
// priorities and kinds are lowercase strings.

/** The session mode, owned by the host (DISARMED → ARMED → ACTIVE → SAFE; ESTOP latches). */
export type Mode = 'disarmed' | 'armed' | 'active' | 'safe' | 'estop';
/** What a robot's edge did on a control tick, per command group (from the part's applied stream). */
export type EdgeMode = 'idle' | 'driving' | 'hold' | 'safe' | 'estop';
/** Lease class, lowest first: a higher class preempts a lower one. */
export type Priority = 'scripted' | 'model' | 'operator' | 'safety';
/** Traffic class: one Worker and one WebSocket each. */
export type TrafficClass = 'control' | 'state' | 'bulk';
/** How a stream's values reach JS: numbers (as f64), a JPEG or raw RGB frame, or text. */
export type Kind = 'f64' | 'jpeg' | 'rgb8' | 'text';

/** One declared field. Numeric values are delivered flat, field by field: this field is `data[offset ..< offset + len]`. */
export interface FieldDecl {
  name: string;
  dtype: string;
  shape: number[];
  unit: string;
  offset: number;
  len: number;
}

/** A stream the session declares: `{robot}/{part}`, or a commandable part's applied stream `{robot}/{part}/_applied`. */
export interface StreamDecl {
  path: string;
  robot: string;
  part: string;
  applied: boolean;
  channel: number;
  class: 'state' | 'bulk';
  kind: Kind;
  rate_hz: number;
  /** The declared freshness bound (NaN: none). */
  max_age_ms: number;
  fields: FieldDecl[];
  /** Frames only. */
  width: number;
  height: number;
}

/** A commandable part: what `lease()` arbitrates. `send`/`drive` values are its command fields' elements, in order. */
export interface ControlDecl {
  path: string;
  robot: string;
  part: string;
  channel: number;
  class: 'control';
  kind: string;
  /** The command mode (`position`, `velocity`, ...). */
  mode: string;
  /** One member of a command group is leased at a time. */
  group: string | null;
  fields: FieldDecl[];
  rate_hz: number;
  /** How long the edge holds the last command before it goes safe. */
  hold_max_ms: number;
  /** This connection's binding covers it. */
  bound: boolean;
}

/** This participant's clock against session time: `session = local + offset_ms`, within `error_ms`. */
export interface Clock {
  offset_ms: number;
  rtt_ms: number;
  error_ms: number;
  samples: number;
  /** The host's published mapping of this participant (from the same heartbeats). */
  host_offset_ms?: number;
  host_error_ms?: number;
}

/** The session as the host's latest snapshot shows it. */
export interface SessionState {
  session: string;
  revision: number;
  epoch: number;
  mode: Mode;
  /** Why the mode is what it is (`robot_lost`, `estop_local`, ...; `unspecified` before any change). */
  mode_reason: string;
  estop_latched: boolean;
}

/** The host's view of one commandable part, plus its edge mode when its applied stream is subscribed. */
export interface Status {
  path: string;
  mode: Mode;
  mode_reason: string;
  epoch: number;
  estop_latched: boolean;
  /** The lease holder's participant id (`''`: none). */
  holder: string;
  priority: Priority | '';
  token: number;
  /** This connection holds it. */
  ours: boolean;
  bound: boolean;
  edge?: EdgeMode;
}

/** An applied sample's metadata: what the edge dispatched on that tick. */
export interface AppliedMeta {
  mode: EdgeMode;
  present: boolean;
  token: number;
}

/** A sample you own: `data` is yours to keep. Times are in this participant's clock, ms since the UNIX epoch. */
export interface Sample<T extends ArrayBufferView = Float64Array> {
  path: string;
  seq: number;
  epoch: number;
  t_capture_ms: number;
  /** Capture → now. */
  age_ms: number;
  /** Publish → delivered here. */
  latency_ms: number;
  data: T;
  applied?: AppliedMeta;
  /** Text streams. */
  text?: string;
}

export interface Frame extends Sample<Uint8Array> {
  kind: Kind;
  width: number;
  height: number;
  /** Image decode time in the bulk Worker (ms). */
  decode_ms: number;
}

/**
 * A borrowed value from `peek()` or a `frame` event: `data` (and `bitmap`) point at pooled buffers the
 * session reuses. Valid until released: call `release()` or use `using f = session.peek(path)`. Reading
 * `data`/`bitmap` after release throws. A `frame` event's value is released when the listener returns.
 */
export interface Borrowed<T extends Sample<ArrayBufferView>> extends Disposable {
  readonly value: T;
  /** Decoded image (bulk frames in browsers; undefined where `createImageBitmap` is unavailable). */
  readonly bitmap: ImageBitmap | undefined;
  readonly released: boolean;
  release(): void;
}

export type LeaseState = 'requested' | 'held' | 'expiring' | 'expired' | 'revoked' | 'released';

/** Why a lease ended without `release()`. */
export interface Revocation {
  path: string;
  token: number;
  reason: string;
  /**
   * `host`: the host took it (preempted, lapsed, a new epoch, a rebinding). `disconnected`: the control
   * socket dropped, so the token is abandoned; it is never resumed, take a new lease. `worker_restart`: the
   * control Worker was respawned after a trap. `closed`: the session was closed.
   */
  cause: 'host' | 'disconnected' | 'worker_restart' | 'closed' | 'robot_left' | 'session_ended';
}

export interface LeaseEvent {
  event: 'grant' | 'refused' | 'revoke';
  path: string;
  holder: string;
  priority: Priority | '';
  token: number;
  reason?: string;
  /** This session's lease. */
  ours: boolean;
}

/** The gateway saw a chunk the host's snapshot fences (it still forwards it; the edge refuses it). */
export interface FenceReport {
  channel: number;
  token: number;
  reason: string;
  /** Set when the check ran locally (`debug.sendStaleCommand`) before sending. */
  local?: boolean;
}

export interface WorkerRestart {
  cls: TrafficClass;
  reason: string;
  /** Restarts of this class so far. */
  restarts: number;
  /** Detection → new Worker ready, ms. */
  downtime_ms: number;
}

export interface SocketEvent {
  cls: TrafficClass;
  /** Reconnect attempts since the last successful open. */
  attempt: number;
  code?: number;
  reason?: string;
  /** The gateway refused or ended the connection for good (`code` says why): no reconnect follows. */
  fatal?: boolean;
}

/** A robot joined (`present`) or left the session: its parts and streams come and go with it. */
export interface RobotEvent {
  robot: string;
  present: boolean;
}

/**
 * The connection ended for good: `4410` the session ended (its host is gone), `4401` the token was
 * refused (unknown, or revoked on the robot), `4403` the page origin or what the token's device may do
 * (its cap) was. Leases end and subscriptions close; connect again to continue.
 */
export interface SessionEnded {
  code: number;
  reason: string;
}

export interface ConsoleLine {
  cls: string;
  lvl: string;
  msg: string;
}

export interface LeaseStateEvent {
  leases: Record<string, { state: string; token: number; remaining_ms: number; hz: number; driving: boolean }>;
  cmd: { sent: number; refused: number; gated: number; lastRefusal: string; hbSent: number; heartbeats: boolean };
}

export interface SessionEvents {
  state: SessionState;
  status: Status;
  lease: LeaseEvent;
  clock: Clock;
  frame: Borrowed<Frame>;
  fence: FenceReport;
  worker_restart: WorkerRestart;
  disconnected: SocketEvent;
  reconnected: SocketEvent;
  console: ConsoleLine;
  lease_state: LeaseStateEvent;
  robot: RobotEvent;
  ended: SessionEnded;
}

export interface BackoffOptions {
  /** First retry delay (ms). Default 100. */
  initialMs?: number;
  /** Cap (ms). Default 5000. */
  maxMs?: number;
  /** Growth per attempt. Default 2. */
  factor?: number;
  /** 0..1: the share of each delay that is randomized (full jitter at 1). Default 0.5. */
  jitter?: number;
}

export interface ConnectOptions {
  /** Bearer token, sent as the `rlmesh.token.<token>` WebSocket subprotocol next to `rlmesh.v2`; every gateway socket needs one. In a browser, `connect()` with no URL takes the page's (`#t=…`). */
  token?: string;
  /** The class this connection's participant attaches at, and its leases are arbitrated at. Default `model`; `operator` (taking over from a model) needs a gateway started with `--max-priority operator`, and a token whose device may; `observer` takes no leases (what a token paired at `--max observer` may do). */
  priority?: Priority | 'observer';
  /** The declarations and a first clock estimate must arrive within this (ms). Default 5000. */
  timeoutMs?: number;
  /** Per-socket reconnect backoff. */
  backoff?: BackoffOptions;
  /** A class Worker that posts nothing for this long is restarted (ms). Default 10000. */
  workerStallMs?: number;
  /** Override where the class Worker script and the wasm are loaded from. */
  workerUrl?: string | URL;
  wasmUrl?: string | URL;
  /** Only `per-class` (one Worker per traffic class) exists. */
  workers?: 'per-class';
}

export interface LeaseOptions {
  /** The grant must arrive within this (ms). Default 3000. */
  timeoutMs?: number;
  /** The lease ttl: how long the host waits for a heartbeat before it lapses. Default: the session's lease grace. */
  ttlMs?: number;
}

export interface SubscribeOptions {
  /** Bounded queue: when full, the oldest sample is dropped and `dropped` counts it. Default 1024. */
  maxQueue?: number;
}
