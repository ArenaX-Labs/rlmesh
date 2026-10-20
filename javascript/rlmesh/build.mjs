// Build the `rlmesh` package into dist/ (and, with --demo, the browser example into examples/web/dist/).
//
//   dist/browser.js   `browser` export condition (Workers via `new Worker`, wasm via fetch)
//   dist/node.js      `default` export condition (worker_threads, wasm from disk)
//   dist/worker.js    the class Worker, shared by both (bundles the wasm-bindgen glue)
//   dist/rlmesh_bg.wasm, dist/*.d.ts
//
// Needs pkg/ (wasm-bindgen + wasm-opt output): `mise run build:js` builds it first.
import * as esbuild from 'esbuild';
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, rmSync, statSync } from 'node:fs';

process.chdir(new URL('.', import.meta.url).pathname);
if (!existsSync('pkg/rlmesh_bg.wasm')) {
  console.error('pkg/rlmesh_bg.wasm missing: run `mise run build:js`');
  process.exit(1);
}
const demo = process.argv.includes('--demo');
const common = { bundle: true, format: 'esm', target: 'es2022', sourcemap: true, logLevel: 'warning' };
const worker = { ...common, entryPoints: ['src/worker.ts'], platform: 'neutral', external: ['node:*'], mainFields: ['module', 'main'] };

rmSync('dist', { recursive: true, force: true });
mkdirSync('dist');
await esbuild.build({ ...common, entryPoints: ['src/browser.ts'], outfile: 'dist/browser.js', platform: 'browser' });
await esbuild.build({ ...common, entryPoints: ['src/node.ts'], outfile: 'dist/node.js', platform: 'node' });
await esbuild.build({ ...worker, outfile: 'dist/worker.js' });
cpSync('pkg/rlmesh_bg.wasm', 'dist/rlmesh_bg.wasm');
execFileSync('npx', ['tsc', '-p', 'tsconfig.build.json'], { stdio: 'inherit' });
for (const f of ['browser.js', 'node.js', 'worker.js', 'rlmesh_bg.wasm', 'browser.d.ts', 'node.d.ts']) console.log(`dist/${f}`, statSync(`dist/${f}`).size);

if (demo) {
  const web = '../../examples/web';
  rmSync(`${web}/dist`, { recursive: true, force: true });
  mkdirSync(`${web}/dist`);
  await esbuild.build({ ...common, minify: true, entryPoints: [`${web}/main.ts`], outfile: `${web}/dist/main.js`, platform: 'browser' });
  await esbuild.build({ ...worker, minify: true, outfile: `${web}/dist/worker.js` });
  cpSync(`${web}/index.html`, `${web}/dist/index.html`);
  cpSync('pkg/rlmesh_bg.wasm', `${web}/dist/rlmesh_bg.wasm`);
  cpSync('pkg/size.json', `${web}/dist/size.json`);
  for (const f of ['main.js', 'worker.js', 'rlmesh_bg.wasm']) console.log(`${web}/dist/${f}`, statSync(`${web}/dist/${f}`).size);
}
