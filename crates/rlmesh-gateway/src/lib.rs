//! The RLMesh browser gateway: browsers join a real-time session (protocol
//! v2) over WebSockets, one per traffic class, because a page cannot open
//! the session's own transport (no raw TCP or UDP).
//!
//! The gateway is a session participant, not a second authority: every
//! browser control connection attaches a participant of its own
//! ([`bridge`]); leases, releases and the session verbs go to the host
//! through it, the browser's own heartbeats and chunks are forwarded as
//! sent, and samples and authority updates reach the browser as their
//! publisher sent them ([`frame`]). The browser side ([`server`]) is the
//! `rlmesh` npm package's three class sockets, behind [`security`]: a
//! loopback bind by default, a bearer token in the subprotocol list, an
//! Origin allowlist and per-connection limits, and optional TLS ([`tls`]).
//!
//! `docs/browser.md` describes the binding and the security model.

#![warn(missing_docs)]

pub mod bridge;
pub mod config;
pub mod frame;
pub mod security;
pub mod server;
pub mod tls;

pub use bridge::{BrowserLink, Gateway, Outbound};
pub use config::{Args, Config};
pub use server::{AppState, router};
