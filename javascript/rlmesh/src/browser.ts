// `rlmesh` for browsers (the `browser` export condition).
//
//   import * as rlmesh from 'rlmesh';
//   const session = await rlmesh.connect();                      // the gateway serving this page (#t=token)
//   for await (const s of session.subscribe('arm7/arm')) s.data; // Float64Array, yours
//   using f = session.peek('arm7/scene/cam');                    // borrowed: pooled buffer + ImageBitmap
//   const lease = await session.lease('arm7/arm');               // preempts lower classes than `operator`
//   lease.drive([0.5, 0, 0, 0, 0, 0, 0], { hz: 100 });            // the loop runs in the control Worker
//   session.on('worker_restart', (e) => …); lease.revoked.then((r) => …);

import { connectWith, type Platform, type Session } from './session.js';
import type { WorkerLike } from './supervisor.js';
import type { ConnectOptions } from './types.js';

export * from './index.js';

/** Compile the wasm once on the main thread; every class Worker instantiates the same Module. */
async function compile(url: URL): Promise<WebAssembly.Module> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  if (res.headers.get('content-type')?.startsWith('application/wasm')) return WebAssembly.compileStreaming(res);
  return WebAssembly.compile(await res.arrayBuffer());
}

function webWorker(url: URL, name: string): WorkerLike {
  const w = new Worker(url, { type: 'module', name });
  const a: WorkerLike = {
    onmessage: null,
    onerror: null,
    postMessage: (m, t = []) => w.postMessage(m, t),
    terminate: () => w.terminate(),
  };
  w.onmessage = (e) => a.onmessage?.(e);
  w.onerror = (e) => a.onerror?.({ message: e.message });
  return a;
}

export function browserPlatform(opts: ConnectOptions = {}): Platform {
  const workerUrl = new URL(opts.workerUrl ?? new URL('./worker.js', import.meta.url));
  const wasmUrl = new URL(opts.wasmUrl ?? new URL('./rlmesh_bg.wasm', import.meta.url));
  return {
    createWorker: (cls) => webWorker(workerUrl, `rlmesh-${cls}`),
    loadModule: () => compile(wasmUrl),
    canShareMemory: !!(globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated,
  };
}

const TOKEN_KEY = 'rlmesh.token';

/**
 * The token of the gateway serving this page: `#t=…` in the page URL, which never reaches a server, is
 * moved to `sessionStorage` (and out of the URL and its history entry); else the one kept there.
 */
function pageToken(): string | undefined {
  const t = new URLSearchParams(location.hash.slice(1)).get('t');
  if (t) {
    sessionStorage.setItem(TOKEN_KEY, t);
    history.replaceState(history.state, '', location.pathname + location.search);
  }
  return sessionStorage.getItem(TOKEN_KEY) ?? undefined;
}

/**
 * Join a session through its gateway. `url` is the gateway's base `ws://`/`wss://` URL; it defaults to
 * the page's own origin, with the token from the page URL's fragment (`#t=…`) unless `opts.token` is
 * given. Resolves once the declarations and a first clock estimate have arrived.
 */
export function connect(url?: string, opts: ConnectOptions = {}): Promise<Session> {
  const base = url ?? `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`;
  const token = opts.token ?? (url === undefined ? pageToken() : undefined);
  return connectWith(browserPlatform(opts), base, token ? { ...opts, token } : opts);
}
