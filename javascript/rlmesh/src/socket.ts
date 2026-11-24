import type { BackoffOptions } from './types.js';

export const SUBPROTOCOL = 'rlmesh.v2';
export const TOKEN_PREFIX = 'rlmesh.token.';

export function protocolsFor(token?: string): string[] {
  return token ? [SUBPROTOCOL, `${TOKEN_PREFIX}${token}`] : [SUBPROTOCOL];
}

const DEFAULTS: Required<BackoffOptions> = { initialMs: 100, maxMs: 5000, factor: 2, jitter: 0.5 };

export function backoffDelay(attempt: number, opts: BackoffOptions = {}, rand: () => number = Math.random): number {
  const o = { ...DEFAULTS, ...opts };
  const base = Math.min(o.maxMs, o.initialMs * o.factor ** Math.max(0, attempt));
  const j = Math.min(1, Math.max(0, o.jitter));
  return base * (1 - j) + base * j * rand();
}

export interface WebSocketLike {
  binaryType: string;
  readonly readyState: number;
  onopen: ((ev: unknown) => void) | null;
  onclose: ((ev: { code?: number; reason?: string }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  send(data: ArrayBufferView | ArrayBuffer | string): void;
  close(code?: number, reason?: string): void;
}

export type WebSocketCtor = new (url: string, protocols?: string | string[]) => WebSocketLike;

export interface SocketHandlers {
  onOpen(reconnect: boolean): void;
  onMessage(data: ArrayBuffer): void;
  onText?(text: string): void;
  onClose(info: { code?: number; reason?: string; attempt: number; delayMs: number; wasOpen: boolean; fatal?: boolean }): void;
}

export interface SocketOptions {
  backoff?: BackoffOptions;
  /** An open shorter than this does not reset the backoff (a server that accepts then drops). */
  stableMs?: number;
  WebSocket?: WebSocketCtor;
  setTimeout?: (fn: () => void, ms: number) => unknown;
  clearTimeout?: (h: unknown) => void;
  now?: () => number;
  random?: () => number;
}

/** Close codes the gateway ends a connection with for good: a bad token, a bad origin, the session ended. */
export const FATAL_CLOSE = new Set([4401, 4403, 4410]);

export class ReconnectingSocket {
  private ws: WebSocketLike | null = null;
  private attempt = 0;
  private opened = 0;
  private everOpened = false;
  private timer: unknown = null;
  private stopped = false;
  private readonly WS: WebSocketCtor;
  private readonly st: (fn: () => void, ms: number) => unknown;
  private readonly ct: (h: unknown) => void;
  private readonly now: () => number;

  constructor(
    readonly url: string,
    readonly protocols: string[],
    private h: SocketHandlers,
    private opts: SocketOptions = {},
  ) {
    const WS = opts.WebSocket ?? (globalThis as unknown as { WebSocket?: WebSocketCtor }).WebSocket;
    if (!WS) throw new Error('no WebSocket implementation (Node >= 22 or a browser)');
    this.WS = WS;
    this.st = opts.setTimeout ?? ((fn, ms) => setTimeout(fn, ms));
    this.ct = opts.clearTimeout ?? ((h) => clearTimeout(h as ReturnType<typeof setTimeout>));
    this.now = opts.now ?? (() => performance.now());
    this.connect();
  }

  get isOpen() {
    return this.ws?.readyState === 1;
  }

  get attempts() {
    return this.attempt;
  }

  send(b: ArrayBufferView | ArrayBuffer | string): boolean {
    if (this.ws?.readyState !== 1) return false;
    this.ws.send(b);
    return true;
  }

  close() {
    this.stopped = true;
    if (this.timer != null) this.ct(this.timer);
    this.timer = null;
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      ws.onclose = ws.onerror = ws.onmessage = ws.onopen = null;
      try {
        ws.close(1000, 'closed');
      } catch {
        /* already closed */
      }
    }
  }

  reconnectNow() {
    this.ws?.close(4000, 'reconnect');
  }

  private connect() {
    if (this.stopped) return;
    let ws: WebSocketLike;
    try {
      ws = new this.WS(this.url, this.protocols);
    } catch (e) {
      this.scheduleRetry({ code: 0, reason: String(e) }, false);
      return;
    }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    let wasOpen = false;
    ws.onopen = () => {
      wasOpen = true;
      this.opened = this.now();
      const reconnect = this.everOpened;
      this.everOpened = true;
      this.h.onOpen(reconnect);
    };
    ws.onmessage = (e) => {
      if (e.data instanceof ArrayBuffer) this.h.onMessage(e.data);
      else if (typeof e.data === 'string') this.h.onText?.(e.data);
    };
    ws.onerror = () => {
      /* onclose follows */
    };
    ws.onclose = (e) => {
      if (this.ws !== ws) return;
      this.ws = null;
      if (wasOpen && this.now() - this.opened >= (this.opts.stableMs ?? 2000)) this.attempt = 0;
      if (e.code != null && FATAL_CLOSE.has(e.code)) {
        this.stopped = true;
        this.h.onClose({ attempt: this.attempt, delayMs: 0, wasOpen, code: e.code, reason: e.reason ?? '', fatal: true });
        return;
      }
      this.scheduleRetry(e, wasOpen);
    };
  }

  private scheduleRetry(e: { code?: number; reason?: string }, wasOpen: boolean) {
    if (this.stopped) return;
    const delayMs = backoffDelay(this.attempt, this.opts.backoff, this.opts.random);
    const info = { attempt: this.attempt, delayMs, wasOpen, ...(e.code != null ? { code: e.code } : {}), ...(e.reason ? { reason: e.reason } : {}) };
    this.attempt++;
    this.h.onClose(info);
    this.timer = this.st(() => {
      this.timer = null;
      this.connect();
    }, delayMs);
  }
}
