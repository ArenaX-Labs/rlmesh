// Pack the npm release (`mise run release:js:package`, after `build:js` and release:js:native):
//
//   target/npm-pkg/rlmesh-{version}.tgz                  the package: dist/ without any .node
//   target/npm-pkg/rlmesh-{platform}-{arch}-{version}.tgz one @rlmesh/{platform}-{arch} per addon
//
// The version comes from the Cargo workspace, so every language ships the same one. `rlmesh` names
// each platform package as an optional dependency at that exact version; npm installs only the one
// whose os/cpu/libc match, and native.ts loads it. --require-all refuses a partial set (publish).
import { execFileSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";

const PLATFORMS = {
  "linux-x64": { os: "linux", cpu: "x64", libc: "glibc" },
  "linux-arm64": { os: "linux", cpu: "arm64", libc: "glibc" },
  "darwin-x64": { os: "darwin", cpu: "x64" },
  "darwin-arm64": { os: "darwin", cpu: "arm64" },
  "win32-x64": { os: "win32", cpu: "x64" },
};

const root = new URL("..", import.meta.url).pathname;
process.chdir(root);
const requireAll = process.argv.includes("--require-all");
const version = readFileSync("Cargo.toml", "utf8").match(/^version = "(.+)"$/m)[1];
const source = JSON.parse(readFileSync("javascript/rlmesh/package.json", "utf8"));
const out = "target/npm-pkg";
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

const common = {
  version,
  license: source.license,
  homepage: "https://rlmesh.dev",
  repository: {
    type: "git",
    url: "git+https://github.com/ArenaX-Labs/rlmesh.git",
    directory: "javascript/rlmesh",
  },
};
const licenses = ["LICENSE-APACHE", "LICENSE-MIT"];

function pack(dir) {
  for (const f of licenses) cpSync(f, `${dir}/${f}`);
  execFileSync("npm", ["pack", "--pack-destination", `../`], {
    cwd: dir,
    stdio: ["ignore", "ignore", "inherit"],
  });
  rmSync(dir, { recursive: true });
}

const present = Object.keys(PLATFORMS).filter((p) =>
  existsSync(`target/npm-native/rlmesh.${p}.node`),
);
const missing = Object.keys(PLATFORMS).filter((p) => !present.includes(p));
if (requireAll && missing.length) {
  console.error(`missing native addons in target/npm-native: ${missing.join(", ")}`);
  process.exit(1);
}
for (const platform of present) {
  const { os, cpu, libc } = PLATFORMS[platform];
  const dir = `${out}/${platform}`;
  mkdirSync(dir);
  cpSync(`target/npm-native/rlmesh.${platform}.node`, `${dir}/rlmesh.node`);
  const pkg = {
    name: `@rlmesh/${platform}`,
    ...common,
    description: `The rlmesh package's native addon for ${platform}. Install rlmesh, not this package.`,
    os: [os],
    cpu: [cpu],
    ...(libc ? { libc: [libc] } : {}),
    main: "rlmesh.node",
    files: ["rlmesh.node", ...licenses],
  };
  writeFileSync(`${dir}/package.json`, `${JSON.stringify(pkg, null, 2)}\n`);
  pack(dir);
}

if (!existsSync("javascript/rlmesh/dist/browser.js")) {
  console.error("javascript/rlmesh/dist is not built: run `mise run build:js`");
  process.exit(1);
}
const dir = `${out}/rlmesh`;
mkdirSync(dir);
cpSync("javascript/rlmesh/dist", `${dir}/dist`, {
  recursive: true,
  filter: (f) => !f.endsWith(".node"),
});
cpSync("javascript/rlmesh/README.md", `${dir}/README.md`);
const { private: _, scripts: __, devDependencies: ___, ...rest } = source;
const pkg = {
  ...rest,
  ...common,
  files: [...rest.files, ...licenses],
  optionalDependencies: Object.fromEntries(
    Object.keys(PLATFORMS).map((p) => [`@rlmesh/${p}`, version]),
  ),
};
writeFileSync(`${dir}/package.json`, `${JSON.stringify(pkg, null, 2)}\n`);
pack(dir);

for (const f of readdirSync(out).sort()) console.log(`${out}/${f}`);
if (missing.length)
  console.log(`(no addon for ${missing.join(", ")}: those platforms fall back to connect())`);
