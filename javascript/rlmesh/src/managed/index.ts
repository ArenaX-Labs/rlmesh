// `rlmesh/managed`: the managed platform's HTTP API as Hey API SDK functions. Pure TypeScript over
// `fetch`: no wasm, no native addon. Node resolves node.ts instead.
import { configure } from './client.js';
import { client } from './gen/client.gen.js';

configure(client);

export * from './gen/index.js';
export { client };
export type { Client } from './gen/client/index.js';
export { DEFAULT_BASE_URL, ManagedError, createClient, type ClientConfig, type EvaluationStreamFrame, type EvaluationStreamSnapshot } from './client.js';
