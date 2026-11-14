# Remote clients

RLMesh 0.2 remote envs and models are realtime session participants. The
reviewed member and parameter mappings live in [manifest.toml](manifest.toml).
The following replaces the 0.1 transport design, including its authentication,
endpoint helpers, shutdown RPC and wire deadlines.

## Addresses and options

`RemoteEnv`, `RemoteVectorEnv` and `RemoteModel` take a session address such as
`:0/env`, `host:port/env`, or the address returned by a server. TCP, Unix and
HTTPS v1 endpoint addresses fail with ValueError and the migration hint:
`v1 endpoints need RLMesh 0.1; serve with rlmesh.serve / pass the session address`.
There are no host, port, path, transport, api_key or tls_ca_file constructor
helpers. The platform HTTP client retains its own credentials.

Env options contain address, connect timeout, request timeout and workflow
edition. Model options contain address and the two timeouts. Timeouts are local
bounds: no request deadline header is sent. Zero and None mean no bound;
negative or non-finite seconds fail with ValueError. C uses explicit presence
flags, C++ optional durations, and Rust Option<Duration>.

## Client lifecycle

Env construction connects and caches the env contract, spaces, metadata and
identity. Model construction validates its address and options; the model is
reached when a run or binding opens a session. Scalar and vector clients check
the endpoint's lane count. The session controls its driver lease.

`close` releases a client locally. It does not shut down the serving process.
Remote shutdown is absent. A server owner uses EnvServer.shutdown, cancellation
or its local lifecycle controls. Session call cancellation and request timeout
abandon the caller's wait while the session owns completion of an in-flight op.
The env close waits up to request_timeout, or a five-second grace when unset.

Native stubs, the Python API snapshot, header extraction, generated C# interop,
and behavioural parity fixtures check these mappings. C, C++ and C# remain
fenced; the C ABI will be released later as ABI 7.

## Decisions

| ID     | 0.2 behavior                                                                                |
| ------ | ------------------------------------------------------------------------------------------- |
| RMT-1  | Env construction connects.                                                                  |
| RMT-2  | Address is an options field in native APIs.                                                 |
| RMT-3  | Scalar and vector envs share RemoteEnvOptions.                                              |
| RMT-4  | Connect and request timeouts are local bounds.                                              |
| RMT-5  | Retired: endpoint tokens are removed.                                                       |
| RMT-6  | Cancellation abandons a call's wait.                                                        |
| RMT-7  | Async twins use the library runtime; remote shutdown twins are removed.                     |
| RMT-8  | Implicit teardown performs bounded client close.                                            |
| RMT-9  | Cached contract and space accessors survive client close.                                   |
| RMT-10 | Native autoreset metadata stays a string; Python restores its Gymnasium enum.               |
| RMT-11 | Retired: port helpers are removed.                                                          |
| RMT-12 | Invalid options and arity fail with ValueError; failed dials with ConnectionError.          |
| RMT-13 | I/O on a closed client is RuntimeError; close remains idempotent.                           |
| RMT-14 | Rust-only remote plumbing belongs to binding.                                               |
| RMT-15 | Vector rewards and flags use native arrays with explicit C capacities.                      |
| RMT-16 | Render returns a decoded uint8 Value or None; malformed frames fail with ProtocolException. |
| RMT-17 | C text accessors return owned RLMeshBytes and explicit optional presence flags.             |
| RMT-18 | Env info wins over runtime episode keys, with one log message per client.                   |
| RMT-19 | Retired: contradictory transport helpers are removed.                                       |
| RMT-20 | Workflow edition selection belongs to the session.                                          |
| RMT-21 | Retired: no request deadlines travel on a v1 wire.                                          |
| RMT-22 | Retired: gRPC API keys and TLS CA options are removed.                                      |
| RMT-23 | A served model can drive a local env.                                                       |
| RMT-24 | C# construction follows RMT-1; async construction uses a factory method.                    |
| RMT-25 | Cached accessor values follow native ownership rules.                                       |
| RMT-26 | Retired: only the platform HTTP client uses RLMESH_API_KEY.                                 |
