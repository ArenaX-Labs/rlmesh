// `rlmesh/managed`: the managed platform's public HTTP API as Hey API SDK functions, in browsers
// and Node. Pure TypeScript over `fetch`: it loads no wasm and no native addon. Node resolves
// node.ts instead, which adds RLMESH_API_KEY / RLMESH_PLATFORM_URL and the signed-in CLI.
import { configure } from './client.js';
import { client } from './gen/client.gen.js';

configure(client);

export * from './gen/index.js';
export { client };
export type { Client } from './gen/client/index.js';
export { DEFAULT_BASE_URL, ManagedError, createClient, type ClientConfig, type EvaluationStreamFrame, type EvaluationStreamSnapshot } from './client.js';
