// `rlmesh` for Node >= 22 (the `default` export condition): the same session, class Workers on
// worker_threads, Node's global WebSocket, the wasm read from disk. Frames carry no ImageBitmap.

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';
import { connectWith, type Platform, type Session } from './session.js';
import type { WorkerLike } from './supervisor.js';
import type { ConnectOptions, TrafficClass } from './types.js';

export * from './index.js';

/** A worker_threads Worker behind the browser Worker shape the supervisor uses. */
class NodeWorker implements WorkerLike {
  onmessage: ((e: { data: any }) => void) | null = null;
  onerror: ((e: { message?: string }) => void) | null = null;
  private w: Worker;
  private terminated = false;
  constructor(url: URL, name: string) {
    this.w = new Worker(url, { name });
    this.w.unref();
    this.w.on('message', (data) => this.onmessage?.({ data }));
    this.w.on('error', (e) => this.onerror?.({ message: String(e?.message ?? e) }));
    this.w.on('exit', (code) => {
      if (!this.terminated) this.onerror?.({ message: `worker exited with code ${code}` });
    });
  }
  postMessage(m: unknown, transfer: Transferable[] = []) {
    this.w.postMessage(m, transfer as any);
  }
  terminate() {
    this.terminated = true;
    void this.w.terminate();
  }
}

export function nodePlatform(opts: ConnectOptions = {}): Platform {
  const workerUrl = new URL(opts.workerUrl ?? new URL('./worker.js', import.meta.url));
  const wasmUrl = new URL(opts.wasmUrl ?? new URL('./rlmesh_bg.wasm', import.meta.url));
  return {
    createWorker: (cls: TrafficClass) => new NodeWorker(workerUrl, `rlmesh-${cls}`),
    loadModule: async () => WebAssembly.compile(await readFile(fileURLToPath(wasmUrl))),
    canShareMemory: true,
  };
}

/** Join a session through a gateway's base `ws://`/`wss://` URL. */
export function connect(url: string, opts: ConnectOptions = {}): Promise<Session> {
  if (!url) return Promise.reject(new Error('connect(url): a gateway URL is required outside the browser'));
  return connectWith(nodePlatform(opts), url, opts);
}
