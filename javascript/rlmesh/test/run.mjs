// Unit tests: bundle test/*.test.ts with esbuild into .test-dist/, then `node --test`.
// --e2e: the recovery tests in test/e2e/*.test.mjs against dist/node.js and real robot + gateway binaries.
import * as esbuild from 'esbuild';
import { spawnSync } from 'node:child_process';
import { readdirSync, rmSync } from 'node:fs';

process.chdir(new URL('..', import.meta.url).pathname);
let files;
if (process.argv.includes('--e2e')) {
  files = readdirSync('test/e2e').filter((f) => f.endsWith('.test.mjs')).map((f) => `test/e2e/${f}`);
} else {
  rmSync('.test-dist', { recursive: true, force: true });
  const entryPoints = readdirSync('test').filter((f) => f.endsWith('.test.ts')).map((f) => `test/${f}`);
  await esbuild.build({ entryPoints, outdir: '.test-dist', bundle: true, format: 'esm', platform: 'node', target: 'es2022', sourcemap: 'inline', logLevel: 'warning', outExtension: { '.js': '.mjs' } });
  files = entryPoints.map((f) => f.replace(/^test\//, '.test-dist/').replace(/\.ts$/, '.mjs'));
}
const r = spawnSync(process.execPath, ['--test', '--test-reporter=spec', ...files], { stdio: 'inherit' });
process.exit(r.status ?? 1);
