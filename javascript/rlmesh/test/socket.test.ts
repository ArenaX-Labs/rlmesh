import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ReconnectingSocket, backoffDelay, protocolsFor, type WebSocketLike } from '../src/socket.js';

test('backoff grows exponentially to the cap, jitter stays within bounds', () => {
  const o = { initialMs: 100, maxMs: 1000, factor: 2, jitter: 0 };
  assert.deepEqual([0, 1, 2, 3, 4, 10].map((a) => backoffDelay(a, o)), [100, 200, 400, 800, 1000, 1000]);
  for (let a = 0; a < 8; a++) {
    const lo = backoffDelay(a, { ...o, jitter: 0.5 }, () => 0);
    const hi = backoffDelay(a, { ...o, jitter: 0.5 }, () => 0.999999);
    const base = backoffDelay(a, o);
    assert.ok(lo >= base / 2 - 1e-9 && hi <= base && lo <= hi);
  }
});

test('token rides as a subprotocol next to rlmesh.v2', () => {
  assert.deepEqual(protocolsFor(), ['rlmesh.v2']);
  assert.deepEqual(protocolsFor('abc'), ['rlmesh.v2', 'rlmesh.token.abc']);
});

/** A fake WebSocket the test drives by hand, plus manual timers and clock. */
function harness(failFirst: number) {
  const sockets: FakeWS[] = [];
  let created = 0;
  class FakeWS implements WebSocketLike {
    binaryType = '';
    readyState = 0;
    onopen: ((e: unknown) => void) | null = null;
    onclose: ((e: { code?: number; reason?: string }) => void) | null = null;
    onerror: ((e: unknown) => void) | null = null;
    onmessage: ((e: { data: unknown }) => void) | null = null;
    sent: unknown[] = [];
    constructor(
      readonly url: string,
      readonly protocols?: string | string[],
    ) {
      sockets.push(this);
      const n = created++;
      queueMicrotask(() => (n < failFirst ? this.serverClose(1006) : this.serverOpen()));
    }
    serverOpen() {
      this.readyState = 1;
      this.onopen?.({});
    }
    serverClose(code = 1006) {
      this.readyState = 3;
      this.onerror?.({});
      this.onclose?.({ code });
    }
    send(d: unknown) {
      this.sent.push(d);
    }
    close() {
      this.serverClose(1000);
    }
  }
  const timers: { fn: () => void; ms: number }[] = [];
  let t = 0;
  return {
    sockets,
    timers,
    clock: { advance: (ms: number) => (t += ms) },
    opts: {
      WebSocket: FakeWS,
      setTimeout: (fn: () => void, ms: number) => timers.push({ fn, ms }),
      clearTimeout: () => {},
      now: () => t,
      random: () => 0.5,
      backoff: { initialMs: 100, maxMs: 400, factor: 2, jitter: 0 },
      stableMs: 1000,
    },
  };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

test('reconnects with backoff, reports open/reconnect/close, resets after a stable open', async () => {
  const h = harness(2);
  const log: string[] = [];
  const sock = new ReconnectingSocket('ws://x/state', ['rlmesh.v2'], {
    onOpen: (re) => log.push(re ? 'reopen' : 'open'),
    onMessage: () => log.push('msg'),
    onClose: (i) => log.push(`close a${i.attempt} ${i.delayMs}ms`),
  }, h.opts);
  await tick();
  assert.deepEqual(log, ['close a0 100ms']);
  h.timers.shift()!.fn();
  await tick();
  assert.deepEqual(log.slice(1), ['close a1 200ms']);
  h.timers.shift()!.fn();
  await tick();
  assert.equal(log.at(-1), 'open');
  assert.ok(sock.isOpen);
  assert.equal(h.sockets.at(-1)!.binaryType, 'arraybuffer');
  assert.deepEqual(h.sockets.at(-1)!.protocols, ['rlmesh.v2']);
  assert.ok(sock.send(new Uint8Array(1)));
  // An open shorter than stableMs keeps the backoff growing (accept-then-drop servers).
  h.sockets.at(-1)!.serverClose();
  assert.equal(log.at(-1), 'close a2 400ms');
  assert.equal(sock.send(new Uint8Array(1)), false, 'send while down is dropped');
  h.timers.shift()!.fn();
  await tick();
  assert.equal(log.at(-1), 'reopen');
  // A stable open resets it.
  h.clock.advance(5000);
  h.sockets.at(-1)!.serverClose();
  assert.equal(log.at(-1), 'close a0 100ms');
  sock.close();
  h.timers.shift()!.fn(); // a pending retry after close() does nothing
  await tick();
  assert.equal(log.at(-1), 'close a0 100ms');
});

test('a fatal close (bad token, bad origin, session ended) is reported once and never retried', async () => {
  const h = harness(0);
  const closes: { code?: number; fatal?: boolean }[] = [];
  const sock = new ReconnectingSocket('ws://x/control', ['rlmesh.v2'], {
    onOpen: () => {},
    onMessage: () => {},
    onClose: (i) => closes.push({ code: i.code, fatal: i.fatal }),
  }, h.opts);
  await tick();
  h.sockets.at(-1)!.serverClose(4401);
  assert.deepEqual(closes, [{ code: 4401, fatal: true }]);
  assert.equal(h.timers.length, 0, 'no retry scheduled');
  assert.equal(sock.isOpen, false);
});
