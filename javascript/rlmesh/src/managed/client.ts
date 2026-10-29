// The hand-written part of rlmesh/managed: credentials and the error class on a Hey API client
// (gen/, generated). The SDK functions take `{ client, path, query, body }` and resolve to
// `{ data, error, response }`; `error` is a `ManagedError` for every non-2xx answer.

import { createClient as createFetchClient, createConfig, type Client } from './gen/client/index.js';
import type { ErrorDetail } from './gen/types.gen.js';

export const DEFAULT_BASE_URL = 'https://api.rlmesh.dev';

export interface ClientConfig {
  /** The platform's API base URL; `https://api.rlmesh.dev` by default. */
  baseUrl?: string;
  /** An organization or personal API key, sent as the bearer token. */
  apiKey?: string;
  /** A token source called before each request, for browser sessions; used when `apiKey` is unset. */
  getAccessToken?: () => string | undefined | Promise<string | undefined>;
  /** The `fetch` to use; the global one by default. */
  fetch?: typeof fetch;
}

/**
 * A non-2xx answer from the platform: what the SDK functions return as `error` (or throw, with
 * `throwOnError: true`). It keeps the body's `error` object, so it still reads as the contract's
 * `ErrorResponse` (`error.error.code`), and adds the HTTP status and the request id.
 */
export class ManagedError extends Error {
  readonly status: number;
  /** The platform's `error.code`, the machine-readable reason, when the body has the platform's shape. */
  readonly code?: string;
  /** The body's `error` object: `code`, `message` and the per-code `details`. */
  readonly error?: ErrorDetail;
  /** The response body, parsed as JSON when it was JSON. */
  readonly body: unknown;
  readonly requestId?: string;

  constructor(status: number, body: unknown, requestId?: string) {
    const detail = (body as { error?: ErrorDetail } | null)?.error;
    const error = detail && typeof detail === 'object' && typeof detail.message === 'string' ? detail : undefined;
    super(error?.message || `HTTP ${status}: ${typeof body === 'string' ? body.slice(0, 300) : JSON.stringify(body)}`);
    this.name = 'ManagedError';
    this.status = status;
    this.code = error?.code;
    this.error = error;
    this.body = body;
    this.requestId = requestId;
  }
}

/** Credentials from somewhere else (Node: the signed-in CLI); `refresh` asks for a new token. */
export type Fallback = (refresh: boolean) => Promise<{ token: string; baseUrl?: string }>;

/**
 * Configure a Hey API client for the platform: base URL, bearer token for the platform's own
 * `bearerAuth` operations (cluster-join tokens are passed by the caller), and `ManagedError`s.
 * With a `fallback`, requests carrying its token go to its base URL when none was configured,
 * and a 401 refetches the token and retries once.
 */
export function configure(target: Client, config: ClientConfig = {}, fallback?: Fallback): Client {
  const { apiKey, getAccessToken } = config;
  const baseUrl = config.baseUrl?.replace(/\/+$/, '');
  let fallbackToken: string | undefined;
  const fromFallback = async (refresh: boolean) => {
    const signedIn = await fallback!(refresh);
    fallbackToken = signedIn.token;
    return signedIn;
  };
  const carriesFallback = (request: Request) => fallbackToken !== undefined && request.headers.get('Authorization') === `Bearer ${fallbackToken}`;

  target.setConfig({
    baseUrl: baseUrl ?? DEFAULT_BASE_URL,
    ...(config.fetch ? { fetch: config.fetch } : {}),
    auth: async (auth) => {
      if (auth.key !== undefined && auth.key !== 'bearerAuth') return undefined;
      if (apiKey) return apiKey;
      if (getAccessToken) return getAccessToken();
      return fallback ? (await fromFallback(false)).token : undefined;
    },
  });
  target.interceptors.error.use((error, response) =>
    response && !response.ok && !(error instanceof ManagedError) ? new ManagedError(response.status, error, response.headers.get('x-request-id') ?? undefined) : error,
  );
  if (!fallback) return target;

  target.interceptors.request.use(async (request) => {
    if (baseUrl || !carriesFallback(request) || !request.url.startsWith(`${DEFAULT_BASE_URL}/`)) return request;
    const signedIn = (await fromFallback(false)).baseUrl?.replace(/\/+$/, '');
    return signedIn ? new Request(signedIn + request.url.slice(DEFAULT_BASE_URL.length), request) : request;
  });
  target.interceptors.response.use(async (response, request, options) => {
    if (response.status !== 401 || !carriesFallback(request)) return response;
    const { token } = await fromFallback(true);
    const headers = new Headers(request.headers);
    headers.set('Authorization', `Bearer ${token}`);
    const body = options.serializedBody ?? (options.body as BodyInit | undefined);
    return (options.fetch ?? globalThis.fetch)(new Request(request, { headers, body }));
  });
  return target;
}

/** A Hey API client for the platform; pass it as `client` to the SDK functions. */
export function createClient(config?: ClientConfig): Client {
  return configure(createFetchClient(createConfig()), config);
}

/**
 * A frame of `streamEvaluationEvents` (the contract types them as strings): the first is a
 * `snapshot` event (`EvaluationStreamSnapshot`), then status, episode, progress, artifact, log and
 * telemetry deltas, and `resync` when the server's broadcast lagged (re-read the evaluation).
 */
export type EvaluationStreamFrame =
  | { type: 'status'; runId: string; observedAt: string; payload?: unknown }
  | { type: 'episode'; taskId?: string; runId: string; observedAt: string; episode: unknown }
  | { type: 'progress'; taskId?: string; episodeArtifactId?: string; runId: string; observedAt: string; progress: unknown }
  | { type: 'artifact'; taskId?: string; episodeArtifactId?: string; runId: string; observedAt: string; kind?: string; uri?: string }
  | { type: 'log'; taskId?: string; level: string; message: string; observedAt: string }
  | {
      type: 'telemetry';
      /** Present only for the reader organization's dedicated capacity. */
      clusterId?: string;
      /** Shared platform capacity; no cell identity is exposed. */
      managed?: boolean;
      runId: string;
      workloadId?: string;
      kind: string;
      observedAt: string;
      metrics: Record<string, unknown>;
    }
  | { type: 'resync' };

export interface EvaluationStreamSnapshot {
  phase: string;
  routes: Extract<EvaluationStreamFrame, { type: 'telemetry' }>[];
}
