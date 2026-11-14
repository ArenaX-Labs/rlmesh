// `rlmesh/managed` for Node (the `node` export condition): index.ts, plus credentials the way
// Python's client finds them. An unset `apiKey` (and no `getAccessToken`) falls back to
// RLMESH_API_KEY, then to the signed-in CLI (`rlmesh token --json`, after `rlmesh login`); an unset
// `baseUrl` to RLMESH_PLATFORM_URL, then to the CLI profile's platform.

import { execFile } from 'node:child_process';
import { configure, type ClientConfig } from './client.js';
import { createClient as createFetchClient, createConfig, type Client } from './gen/client/index.js';
import { client } from './gen/client.gen.js';

export * from './index.js';

export interface NodeClientConfig extends ClientConfig {
  /** The CLI profile to take credentials from (`rlmesh token --profile`); its default profile otherwise. */
  profile?: string;
}

/** A Hey API client for the platform, with Node's credential fallbacks; pass it as `client` to the SDK functions. */
export function createClient(config: NodeClientConfig = {}): Client {
  return configureNode(createFetchClient(createConfig()), config);
}

function configureNode(target: Client, config: NodeClientConfig): Client {
  const env = process.env;
  const apiKey = config.apiKey ?? (config.getAccessToken ? undefined : env.RLMESH_API_KEY || undefined);
  const baseUrl = config.baseUrl ?? (env.RLMESH_PLATFORM_URL || undefined);
  const fallback = apiKey || config.getAccessToken ? undefined : signedIn(config.profile);
  return configure(target, { ...config, apiKey, baseUrl }, fallback);
}

// The default client the SDK functions use when given none. index.ts configured it already; this
// adds the environment and the CLI.
configureNode(client, {});

interface CliToken {
  platform?: string;
  token: string;
  expiresAt?: string | null;
}

/** The CLI's token, reused until a minute before it expires or until `refresh` (after a 401). */
function signedIn(profile: string | undefined) {
  let pending: Promise<CliToken> | undefined;
  let refreshAt = 0;
  return async (refresh: boolean) => {
    if (!pending || refresh || Date.now() >= refreshAt) {
      const run = cliToken(profile);
      pending = run;
      refreshAt = Infinity;
      run.then(
        (t) => (refreshAt = t.expiresAt ? Date.parse(t.expiresAt) - 60_000 : Infinity),
        () => pending === run && (pending = undefined),
      );
    }
    const { platform, token } = await pending;
    return { baseUrl: platform, token };
  };
}

function cliToken(profile: string | undefined): Promise<CliToken> {
  const args = ['token', '--json', ...(profile ? ['--profile', profile] : [])];
  return new Promise((resolve, reject) => {
    execFile('rlmesh', args, { timeout: 60_000 }, (error, stdout, stderr) => {
      if ((error as NodeJS.ErrnoException | null)?.code === 'ENOENT') {
        return reject(new Error('no credentials: pass apiKey, set RLMESH_API_KEY, or install the rlmesh CLI and run `rlmesh login`'));
      }
      if (error) return reject(new Error(String(stderr).trim() || 'rlmesh token failed; run `rlmesh login`'));
      try {
        resolve(JSON.parse(stdout) as CliToken);
      } catch {
        reject(new Error('rlmesh token --json printed no JSON; update the rlmesh CLI'));
      }
    });
  });
}
