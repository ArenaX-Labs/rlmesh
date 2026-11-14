# rlmesh

RLMesh real-time sessions (protocol v2) for Node and the browser.

> Experimental: the API may change in any 0.x release.

```bash
npm install rlmesh
```

- **Node 22+ and Electron** run a session host, robots and policies in-process
  over a native addon. npm installs the addon for your platform as an optional
  dependency (`@rlmesh/linux-x64`, `@rlmesh/darwin-arm64`, …).
- **Browsers** join a session through `rlmesh-gateway` with `connect()`, over
  the session core compiled to wasm.

Docs: [the Node binding](https://github.com/ArenaX-Labs/rlmesh/blob/main/docs/node.md)
and [the browser binding](https://github.com/ArenaX-Labs/rlmesh/blob/main/docs/browser.md).
