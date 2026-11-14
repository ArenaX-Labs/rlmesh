import type { LeaseState, Priority, Revocation } from './types.js';

/** What a Lease needs from its session (kept narrow so the two modules do not import each other). */
export interface LeaseHost {
  post(m: unknown): boolean;
  request(m: Record<string, unknown>): Promise<any>;
  dropLease(l: Lease): void;
}

/**
 * Command authority over one commandable part, granted by the session host. Commands go out from the
 * control Worker as timed action chunks stamped with the lease's token, binding and epoch; the robot's
 * edge fences them. Heartbeats are automatic while the connection is up, and renew the lease at the host.
 *
 * A lease ends by `release()`, or involuntarily (see `revoked`): the host took it (preempted, lapsed, a
 * new epoch), the control socket dropped, or the control Worker restarted. An involuntarily ended lease
 * is never resumed; take a new one.
 */
export class Lease implements Disposable {
  state: LeaseState = 'held';
  /** Time left before the local deadline (a conservative bound of the host's expiry); refreshed every 250 ms. */
  remaining_ms = Number.NaN;
  /** Resolves when the lease ends without `release()`. Never rejects. */
  readonly revoked: Promise<Revocation>;
  private resolveRevoked!: (r: Revocation) => void;

  /** @internal */
  constructor(
    private host: LeaseHost,
    readonly path: string,
    readonly priority: Priority,
    public token: number,
    readonly holder: string,
  ) {
    this.revoked = new Promise((r) => (this.resolveRevoked = r));
  }

  /** Held and not yet ended (it may still expire locally if heartbeats stop). */
  get active() {
    return this.state === 'held' || this.state === 'expiring';
  }

  /**
   * One command (the part's command fields' elements, in order), valid for `periodMs` (default 20). The
   * edge holds it, then goes safe after the part's `hold_max_ms`: keep commanding (`drive`) while you hold
   * the lease. Resolves false if refused locally (not held, expired, wrong value count).
   */
  async send(values: ArrayLike<number>, opts: { periodMs?: number } = {}): Promise<boolean> {
    if (!this.active) return false;
    const r = await this.host.request({ op: 'send', path: this.path, vals: Float64Array.from(values), period_ms: opts.periodMs ?? 20 });
    return !!r.ok;
  }

  /**
   * Repeat `values` at `hz` from the control Worker until changed; `drive(null)` stops. The loop pauses
   * (counted as `gated`) while the session is not ACTIVE, the lease is not held, or the socket is down.
   */
  drive(values: ArrayLike<number> | null, opts: { hz?: number } = {}) {
    if (values && !this.active) throw new Error(`lease on ${this.path} is ${this.state}`);
    const hz = Math.min(1000, Math.max(1, opts.hz ?? 100));
    this.host.post({ op: 'drive', path: this.path, vals: values ? Float64Array.from(values) : null, hz });
  }

  /** Hand control back to the host. Idempotent. */
  release() {
    if (!this.active && this.state !== 'expired') return;
    this.state = 'released';
    this.host.post({ op: 'release', path: this.path });
    this.host.dropLease(this);
  }

  [Symbol.dispose]() {
    this.release();
  }

  /** @internal */
  _update(l: { state: string; token: number; remaining_ms: number }) {
    if (!this.active) return;
    if (l.state === 'expiring' || l.state === 'held' || l.state === 'expired') this.state = l.state;
    this.remaining_ms = l.remaining_ms;
  }

  /** @internal */
  _end(r: Omit<Revocation, 'path' | 'token'>) {
    if (this.state === 'revoked' || this.state === 'released') return;
    this.state = 'revoked';
    this.resolveRevoked({ path: this.path, token: this.token, ...r });
  }
}
