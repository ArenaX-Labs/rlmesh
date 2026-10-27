import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { claimClusterJoin, createClient as createBrowserClient, getMe, listEvaluations, ManagedError, previewEvaluation } from '../src/managed/index.js';
import { createClient, createEvaluation } from '../src/managed/node.js';

type Call = { url: string; auth: string | null; body: string };

/** A fetch that records each call and answers with `reply` (JSON `{}` by default). */
function fakeFetch(calls: Call[], reply: (call: Call) => Response = () => Response.json({})): typeof fetch {
  return (async (request: Request) => {
    const call = { url: request.url, auth: request.headers.get('authorization'), body: await request.text() };
    calls.push(call);
    return reply(call);
  }) as typeof fetch;
}

test('the bearer token comes from apiKey, else getAccessToken, and never replaces a cluster-join token', async () => {
  const calls: Call[] = [];
  let asked = 0;
  const getAccessToken = async () => (asked++, 'session');
  await getMe({ client: createBrowserClient({ apiKey: 'key', getAccessToken, fetch: fakeFetch(calls), baseUrl: 'https://p.example/' }) });
  const browser = createBrowserClient({ getAccessToken, fetch: fakeFetch(calls) });
  await listEvaluations({ client: browser, query: { status: 'running', tag: ['team:a', 'run:1'] } });
  await claimClusterJoin({ client: browser, headers: { Authorization: 'Bearer join-token' }, body: {} as never });
  assert.deepEqual(
    calls.map((c) => [c.url, c.auth]),
    [
      ['https://p.example/v1/me', 'Bearer key'],
      ['https://api.rlmesh.dev/v1/evaluations?status=running&tag=team%3Aa&tag=run%3A1', 'Bearer session'],
      ['https://api.rlmesh.dev/v1/cluster-join', 'Bearer join-token'],
    ],
  );
  assert.equal(asked, 1);
});

test('in Node, credentials fall back to RLMESH_API_KEY, then to the signed-in CLI, refetched once on a 401', async () => {
  const saved = { ...process.env };
  const bin = mkdtempSync(join(tmpdir(), 'rlmesh-cli-'));
  try {
    const calls: Call[] = [];
    process.env.RLMESH_API_KEY = 'env-key';
    process.env.RLMESH_PLATFORM_URL = 'https://env.example';
    await getMe({ client: createClient({ fetch: fakeFetch(calls) }) });

    delete process.env.RLMESH_API_KEY;
    delete process.env.RLMESH_PLATFORM_URL;
    // Each run prints the next token: cli-1, cli-2, ...
    writeFileSync(
      join(bin, 'rlmesh'),
      `#!/bin/sh\necho "$@" >> "${bin}/argv"\nn=$(wc -l < "${bin}/argv")\necho "{\\"platform\\":\\"https://cli.example/\\",\\"token\\":\\"cli-$n\\",\\"expiresAt\\":null}"\n`,
    );
    chmodSync(join(bin, 'rlmesh'), 0o755);
    process.env.PATH = `${bin}:${process.env.PATH}`;
    const cli = createClient({ profile: 'staging', fetch: fakeFetch(calls, (c) => (c.auth === 'Bearer cli-1' ? Response.json({}, { status: 401 }) : Response.json({ id: 'e1' }))) });
    const { data } = await createEvaluation({ client: cli, body: { name: 'x' } as never });
    await getMe({ client: cli });

    assert.deepEqual(data, { id: 'e1' });
    assert.deepEqual(
      calls.map((c) => [c.url, c.auth, c.body]),
      [
        ['https://env.example/v1/me', 'Bearer env-key', ''],
        ['https://cli.example/v1/evaluations', 'Bearer cli-1', '{"name":"x"}'],
        ['https://cli.example/v1/evaluations', 'Bearer cli-2', '{"name":"x"}'],
        ['https://cli.example/v1/me', 'Bearer cli-2', ''],
      ],
    );
    assert.equal(readFileSync(join(bin, 'argv'), 'utf8'), 'token --json --profile staging\n'.repeat(2));
  } finally {
    process.env = saved;
    rmSync(bin, { recursive: true, force: true });
  }
});

test('a non-2xx answer is a ManagedError with the status, the platform error and the request id, returned or thrown', async () => {
  const refusal = { error: { code: 'unknown_scenario', message: 'no such scenario', details: { scenario: 'x' } } };
  const client = createBrowserClient({ apiKey: 'k', fetch: fakeFetch([], () => Response.json(refusal, { status: 400, headers: { 'x-request-id': 'req-1' } })) });
  const { error, response } = await previewEvaluation({ client, body: {} as never });
  assert.ok(error instanceof ManagedError);
  assert.deepEqual([error.status, error.code, error.message, error.error?.details, error.requestId, response?.status], [400, 'unknown_scenario', 'no such scenario', { scenario: 'x' }, 'req-1', 400]);
  await assert.rejects(previewEvaluation({ client, body: {} as never, throwOnError: true }), (e: unknown) => e instanceof ManagedError && e.code === 'unknown_scenario');

  const html = createBrowserClient({ apiKey: 'k', fetch: fakeFetch([], () => new Response('<h1>bad gateway</h1>', { status: 502 })) });
  const bad = (await getMe({ client: html })).error;
  assert.ok(bad instanceof ManagedError && bad.code === undefined && bad.message === 'HTTP 502: <h1>bad gateway</h1>');
});
