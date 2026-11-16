// Check the packed npm release (`mise run release:js:smoke`, after release:js:package), as the
// release rehearsal does on Linux and macOS. Publishes nothing.
//
// Contents, for every tarball in target/npm-pkg: one version throughout, the Cargo workspace's;
// rlmesh pins each @rlmesh/{platform}-{arch} at it; a .node only in the platform packages and no
// dist/fleet.js anywhere; every file rlmesh's exports name is packed; each addon is a shared
// library for its own target (its ELF, Mach-O or PE32+ header) and, with --max-glibc, needs no
// newer glibc.
//
// Install (unless --no-install), for this machine's platform: a fresh project gets the rlmesh
// tarball and this platform's addon tarball, and no registry, so nothing comes from npm and the
// other platforms' optional dependencies are skipped. Then it imports rlmesh, loads the addon
// through it (its version check), opens a Host, and runs examples/node/hold.mjs with its import
// pointed at the installed package, on a scratch RLMESH_STATE_DIR and a free RLMESH_PORT.
//
//   node scripts/npm-smoke.mjs [--require-all] [--max-glibc 2.28] [--no-install]
import { spawnSync } from "node:child_process";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

// The addon each platform package must carry: its binary format and machine.
const PLATFORMS = {
  "linux-x64": { os: "linux", cpu: "x64", binary: "elf", machine: 0x3e, label: "ELF x86-64" },
  "linux-arm64": { os: "linux", cpu: "arm64", binary: "elf", machine: 0xb7, label: "ELF aarch64" },
  "darwin-x64": { os: "darwin", cpu: "x64", binary: "macho", machine: 0x01000007, label: "Mach-O x86_64" },
  "darwin-arm64": { os: "darwin", cpu: "arm64", binary: "macho", machine: 0x0100000c, label: "Mach-O arm64" },
  "win32-x64": { os: "win32", cpu: "x64", binary: "pe", machine: 0x8664, label: "PE32+ x86-64 DLL" },
};
const LICENSES = ["LICENSE-APACHE", "LICENSE-MIT"];

const root = new URL("..", import.meta.url).pathname;
process.chdir(root);
const args = process.argv.slice(2);
const requireAll = args.includes("--require-all");
const install = !args.includes("--no-install");
const maxGlibcArg = args.includes("--max-glibc") ? args[args.indexOf("--max-glibc") + 1] : undefined;
const maxGlibc = maxGlibcArg?.split(".").map(Number);
if (maxGlibcArg !== undefined && (maxGlibc.length !== 2 || maxGlibc.some(Number.isNaN))) {
  console.error(`--max-glibc takes a version such as 2.28, not ${maxGlibcArg}`);
  process.exit(2);
}

const version = readFileSync("Cargo.toml", "utf8").match(/^version = "(.+)"$/m)[1];
const pkgDir = "target/npm-pkg";
const errors = [];
const fail = (message) => errors.push(message);
const ok = (message) => console.log(`ok  ${message}`);

function run(command, argv, opts = {}) {
  const r = spawnSync(command, argv, { encoding: "utf8", ...opts });
  if (r.error) throw r.error;
  if (r.status !== 0) {
    process.stdout.write(r.stdout ?? "");
    process.stderr.write(r.stderr ?? "");
    const why = r.signal ? `killed by ${r.signal}` : `exit ${r.status}`;
    throw new Error(`${command} ${argv.join(" ")}: ${why}`);
  }
  return r.stdout;
}

/** Every file under `dir`, relative to it, with forward slashes. */
function walk(dir, prefix = "") {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(join(dir, e.name), `${prefix}${e.name}/`) : [`${prefix}${e.name}`],
  );
}

/** The release-relative strings in an exports map: "./dist/node.js", ... */
function exportTargets(exports) {
  if (typeof exports === "string") return [exports];
  return Object.values(exports ?? {}).flatMap(exportTargets);
}

/** What is wrong with `buf` as `platform`'s addon, or null. */
function checkBinary(buf, platform) {
  const want = PLATFORMS[platform];
  if (want.binary === "elf") {
    if (buf.readUInt32BE(0) !== 0x7f454c46) return "not an ELF file";
    if (buf[4] !== 2 || buf[5] !== 1) return "not a 64-bit little-endian ELF";
    if (buf.readUInt16LE(16) !== 3) return `ELF type ${buf.readUInt16LE(16)}, not a shared object (3)`;
    const machine = buf.readUInt16LE(18);
    if (machine !== want.machine) return `ELF machine 0x${machine.toString(16)}, not ${want.label}`;
    if (maxGlibc) {
      let newest = [0, 0];
      for (const m of buf.toString("latin1").matchAll(/GLIBC_(\d+)\.(\d+)/g)) {
        const v = [Number(m[1]), Number(m[2])];
        if (v[0] > newest[0] || (v[0] === newest[0] && v[1] > newest[1])) newest = v;
      }
      if (newest[0] > maxGlibc[0] || (newest[0] === maxGlibc[0] && newest[1] > maxGlibc[1]))
        return `needs GLIBC_${newest.join(".")}, newer than ${maxGlibcArg}`;
    }
  } else if (want.binary === "macho") {
    if (buf.readUInt32LE(0) !== 0xfeedfacf) return "not a 64-bit Mach-O file";
    const cpu = buf.readUInt32LE(4);
    if (cpu !== want.machine) return `Mach-O CPU type 0x${cpu.toString(16)}, not ${want.label}`;
    const type = buf.readUInt32LE(12);
    if (type !== 6 && type !== 8) return `Mach-O file type ${type}, not a dylib (6) or bundle (8)`;
  } else {
    if (buf.toString("latin1", 0, 2) !== "MZ") return "not a PE file (no MZ header)";
    const pe = buf.readUInt32LE(0x3c);
    if (buf.toString("latin1", pe, pe + 4) !== "PE\0\0") return "not a PE file (no PE signature)";
    const machine = buf.readUInt16LE(pe + 4);
    if (machine !== want.machine) return `PE machine 0x${machine.toString(16)}, not ${want.label}`;
    if (!(buf.readUInt16LE(pe + 22) & 0x2000)) return "a PE image, but not a DLL";
    if (buf.readUInt16LE(pe + 24) !== 0x20b) return "not PE32+";
    // Node for Windows does not ship the MSVC runtime: release:js:native links it statically.
    if (/vcruntime\d+\.dll/i.test(buf.toString("latin1"))) return "imports the MSVC runtime DLL";
  }
  return null;
}

// ---- contents -------------------------------------------------------------------------------

if (!existsSync(pkgDir)) {
  console.error(`${pkgDir} is missing: run \`mise run release:js:package\` first`);
  process.exit(1);
}
const scratch = mkdtempSync(join(tmpdir(), "rlmesh-npm-smoke-"));
const tarballs = readdirSync(pkgDir).filter((f) => f.endsWith(".tgz"));
const mainTgz = `rlmesh-${version}.tgz`;
const platformTgz = (p) => `rlmesh-${p}-${version}.tgz`;
const present = Object.keys(PLATFORMS).filter((p) => tarballs.includes(platformTgz(p)));
const missing = Object.keys(PLATFORMS).filter((p) => !present.includes(p));
const expected = new Set([mainTgz, ...present.map(platformTgz)]);
for (const f of tarballs) if (!expected.has(f)) fail(`${f}: not rlmesh or a platform package at ${version}`);
if (!tarballs.includes(mainTgz)) fail(`${mainTgz} is missing`);
if (requireAll && missing.length) fail(`no platform package for ${missing.join(", ")}`);
if (!present.length) fail("no platform packages");

/** Unpack one tarball into the scratch directory: its package.json and its file list. */
function unpack(tgz) {
  const dir = join(scratch, "unpacked", tgz);
  mkdirSync(dir, { recursive: true });
  run("tar", ["-xzf", join(pkgDir, tgz), "-C", dir]);
  const files = walk(join(dir, "package"));
  const pkg = JSON.parse(readFileSync(join(dir, "package", "package.json"), "utf8"));
  return { dir: join(dir, "package"), files, pkg };
}

for (const tgz of [mainTgz, ...present.map(platformTgz)].filter((f) => tarballs.includes(f))) {
  const { dir, files, pkg } = unpack(tgz);
  const before = errors.length;
  if (pkg.version !== version) fail(`${tgz}: version ${pkg.version}, not ${version}`);
  if (pkg.private) fail(`${tgz}: marked private`);
  for (const f of files.filter((f) => /(^|\/)fleet\.js(\.map)?$/.test(f))) fail(`${tgz}: packs ${f}`);
  for (const l of LICENSES) if (!files.includes(l)) fail(`${tgz}: no ${l}`);

  if (tgz === mainTgz) {
    if (pkg.name !== "rlmesh") fail(`${tgz}: named ${pkg.name}`);
    for (const f of files.filter((f) => f.endsWith(".node"))) fail(`${tgz}: packs an addon, ${f}`);
    if (pkg.scripts || pkg.devDependencies) fail(`${tgz}: keeps scripts or devDependencies`);
    if (pkg.dependencies && Object.keys(pkg.dependencies).length) fail(`${tgz}: has dependencies`);
    const optional = pkg.optionalDependencies ?? {};
    const wantOptional = Object.keys(PLATFORMS).map((p) => `@rlmesh/${p}`);
    if (JSON.stringify(Object.keys(optional).sort()) !== JSON.stringify(wantOptional.sort()))
      fail(`${tgz}: optionalDependencies ${Object.keys(optional).join(", ")}`);
    for (const [name, v] of Object.entries(optional)) if (v !== version) fail(`${tgz}: ${name} pinned at ${v}`);
    for (const target of exportTargets(pkg.exports)) {
      if (!files.includes(target.replace(/^\.\//, ""))) fail(`${tgz}: exports ${target}, which is not packed`);
    }
    for (const f of ["README.md", "dist/rlmesh_bg.wasm", "dist/worker.js"])
      if (!files.includes(f)) fail(`${tgz}: no ${f}`);
  } else {
    const platform = tgz.slice("rlmesh-".length, -`-${version}.tgz`.length);
    const want = PLATFORMS[platform];
    if (pkg.name !== `@rlmesh/${platform}`) fail(`${tgz}: named ${pkg.name}`);
    if (pkg.main !== "rlmesh.node") fail(`${tgz}: main is ${pkg.main}`);
    if (String(pkg.os) !== want.os || String(pkg.cpu) !== want.cpu)
      fail(`${tgz}: os ${pkg.os} cpu ${pkg.cpu}, not ${want.os} ${want.cpu}`);
    if (want.os === "linux" && String(pkg.libc) !== "glibc") fail(`${tgz}: libc ${pkg.libc}, not glibc`);
    const extra = files.filter((f) => !["package.json", "rlmesh.node", ...LICENSES].includes(f));
    if (extra.length) fail(`${tgz}: also packs ${extra.join(", ")}`);
    if (files.includes("rlmesh.node")) {
      const why = checkBinary(readFileSync(join(dir, "rlmesh.node")), platform);
      if (why) fail(`${tgz}: rlmesh.node: ${why}`);
    } else fail(`${tgz}: no rlmesh.node`);
  }
  if (errors.length === before) {
    const kind = tgz === mainTgz ? "" : `, ${PLATFORMS[pkg.name.slice("@rlmesh/".length)].label}`;
    ok(`${tgz}: ${pkg.name}@${pkg.version}, ${files.length} files${kind}`);
  }
}
if (missing.length && !requireAll) console.log(`(no platform package for ${missing.join(", ")})`);

// ---- install --------------------------------------------------------------------------------

async function freePort() {
  const server = createServer();
  await new Promise((resolve, reject) => server.once("error", reject).listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return port;
}

// Run inside the installed project: everything it imports resolves from its node_modules.
const CHECK = `import { createRequire } from 'node:module';
import { Host } from 'rlmesh';

const require = createRequire(import.meta.url);
const target = \`\${process.platform}-\${process.arch}\`;
const { version } = require('rlmesh/package.json');
const entry = import.meta.resolve('rlmesh');
if (!entry.endsWith('/node_modules/rlmesh/dist/node.js')) throw new Error(\`rlmesh resolved to \${entry}\`);
const addonFile = require.resolve(\`@rlmesh/\${target}\`);
const addonVersion = require(addonFile).version();
if (addonVersion !== version) throw new Error(\`addon \${addonVersion}, package \${version}\`);
// Host.open loads the addon through rlmesh itself: its lookup and its version check.
const host = await Host.open();
const address = host.address.replace(/#.*/, '#<key>');
await host.close();
await import('rlmesh/managed');
console.log(\`rlmesh \${version} (\${entry}), addon \${addonFile}, a Host opened at \${address}\`);
`;

async function smokeInstall() {
  const target = `${process.platform}-${process.arch}`;
  if (!PLATFORMS[target]) throw new Error(`no npm platform package for ${target}`);
  if (!present.includes(target)) throw new Error(`no ${platformTgz(target)} to install on ${target}`);
  const project = join(scratch, "project");
  mkdirSync(project);
  writeFileSync(
    join(project, "package.json"),
    `${JSON.stringify({ name: "rlmesh-npm-smoke", private: true, type: "module" }, null, 2)}\n`,
  );
  // A registry nothing listens on, and a cache of its own: npm can only use the two tarballs, and
  // the other platforms' optional dependencies fail to resolve and are skipped.
  run(
    "npm",
    [
      "install",
      "--registry=http://127.0.0.1:9/",
      "--fetch-retries=0",
      `--cache=${join(scratch, "npm-cache")}`,
      "--no-audit",
      "--no-fund",
      "--loglevel=error",
      join(root, pkgDir, mainTgz),
      join(root, pkgDir, platformTgz(target)),
    ],
    { cwd: project, timeout: 300_000 },
  );
  const scoped = readdirSync(join(project, "node_modules", "@rlmesh"));
  if (scoped.join() !== target) throw new Error(`installed @rlmesh/${scoped.join(", @rlmesh/")}, not just ${target}`);
  if (walk(join(project, "node_modules", "rlmesh")).some((f) => f.endsWith(".node")))
    throw new Error("the installed rlmesh carries an addon of its own");
  ok(`installed rlmesh and @rlmesh/${target} from the tarballs, with no registry`);

  // examples/node/hold.mjs, importing the installed package and reading its fixture from here.
  const repoImport = "'../../javascript/rlmesh/dist/node.js'";
  const fixture = "'../../crates/rlmesh-session-core/tests/fixtures/robots/ur10e.json'";
  let hold = readFileSync("examples/node/hold.mjs", "utf8");
  for (const [from, to] of [
    [repoImport, "'rlmesh'"],
    [fixture, "'./ur10e.json'"],
  ]) {
    if (hold.split(from).length !== 2) throw new Error(`examples/node/hold.mjs no longer has ${from} once`);
    hold = hold.replace(from, to);
  }
  writeFileSync(join(project, "hold.mjs"), hold);
  copyFileSync("crates/rlmesh-session-core/tests/fixtures/robots/ur10e.json", join(project, "ur10e.json"));
  writeFileSync(join(project, "check.mjs"), CHECK);

  const state = join(scratch, "state");
  mkdirSync(state);
  const env = { ...process.env, RLMESH_STATE_DIR: state, RLMESH_PORT: String(await freePort()) };
  delete env.RLMESH_NATIVE; // the addon must come from the installed package
  const node = (script, timeout) => run(process.execPath, [script], { cwd: project, env, timeout });

  console.log(`ok  ${node("check.mjs", 60_000).trim()}`);
  const out = node("hold.mjs", 120_000);
  const done = out.match(/(\d+) starts completed, (\d+) chunks sent/);
  if (!done || Number(done[1]) === 0 || Number(done[2]) === 0) {
    process.stdout.write(out);
    throw new Error("hold.mjs ran, but its policy completed no starts or sent no chunks");
  }
  ok(`hold.mjs against the installed package: ${done[0]}`);
}

if (!errors.length && install) {
  try {
    await smokeInstall();
  } catch (e) {
    fail(e.message);
  }
}

if (errors.length) {
  for (const e of errors) console.error(`FAIL ${e}`);
  console.error(`(scratch kept in ${scratch})`);
  process.exit(1);
}
rmSync(scratch, { recursive: true, force: true });
console.log(
  `npm smoke passed: ${version}, ${present.length} platform package(s)${install ? `, installed on ${process.platform}-${process.arch}` : ""}`,
);
