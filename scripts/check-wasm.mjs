// The browser binding's .wasm (`mise run check:js:wasm`, after `build:js`):
// - imports: it may only import wasm-bindgen glue ("./<name>_bg.js" / "wbg"). An `env` or `wasi_*`
//   import means something pulled in libc or WASI, which a browser cannot satisfy;
// - size: gzip -9 of the optimized .wasm must stay under the budget (--max-gzip N bytes).
import { readFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';

const args = process.argv.slice(2);
const flag = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const path = args.find((a) => a.endsWith('.wasm')) ?? new URL('../javascript/rlmesh/pkg/rlmesh_bg.wasm', import.meta.url).pathname;
const maxGzip = Number(flag('--max-gzip', 64 * 1024));
const bytes = readFileSync(path);
const mod = new WebAssembly.Module(bytes);
const imports = WebAssembly.Module.imports(mod);
const modules = [...new Set(imports.map((i) => i.module))];
const bad = modules.filter((m) => m !== 'wbg' && !/^\.\/[\w-]+_bg\.js$/.test(m));
const gzip = gzipSync(bytes, { level: 9 }).length;
const names = imports.map((i) => i.name.replace(/_[0-9a-f]{16}$/, ''));
console.log(JSON.stringify({ path, bytes: bytes.length, gzip, max_gzip: maxGzip, import_modules: modules, imports: imports.length, exports: WebAssembly.Module.exports(mod).length, sample: names.slice(0, 8) }));
let fail = false;
if (bad.length) { console.error(`unexpected wasm import modules: ${bad.join(', ')}`); fail = true; }
if (gzip > maxGzip) { console.error(`wasm gzip size ${gzip} B exceeds the budget of ${maxGzip} B`); fail = true; }
process.exit(fail ? 1 : 0);
