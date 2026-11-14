// Unit tests: bundle test/*.test.ts with esbuild into .test-dist/, then `node --test`.
// --e2e: the recovery tests in test/e2e/*.test.mjs against dist/node.js and real robot + gateway binaries.
// --native: test/native/*.test.mjs, sessions in-process over dist/node.js and the native addon.
import * as esbuild from 'esbuild';
import { spawnSync } from 'node:child_process';
import { readdirSync, rmSync } from 'node:fs';

process.chdir(new URL('..', import.meta.url).pathname);
let files;
const suite = ['--e2e', '--native'].find((flag) => process.argv.includes(flag))?.slice(2);
if (suite) {
  files = readdirSync(`test/${suite}`).filter((f) => f.endsWith('.test.mjs')).map((f) => `test/${suite}/${f}`);
} else {
  rmSync('.test-dist', { recursive: true, force: true });
  const entryPoints = readdirSync('test').filter((f) => f.endsWith('.test.ts')).map((f) => `test/${f}`);
  await esbuild.build({ entryPoints, outdir: '.test-dist', bundle: true, format: 'esm', platform: 'node', target: 'es2022', sourcemap: 'inline', logLevel: 'warning', outExtension: { '.js': '.mjs' } });
  files = entryPoints.map((f) => f.replace(/^test\//, '.test-dist/').replace(/\.ts$/, '.mjs'));
}
const r = spawnSync(process.execPath, ['--test', '--test-reporter=spec', ...files], { stdio: 'inherit' });
process.exit(r.status ?? 1);
