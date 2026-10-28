//! Who may open a gateway WebSocket, and what one connection may do.
//!
//! - **Bind:** loopback (127.0.0.1) by default.
//! - **Token:** every socket needs one (design §5: no tokenless loopback), carried in the WebSocket
//!   subprotocol list, as `rlmesh.token.<token>`, next to `rlmesh.v2`. Browsers cannot set headers
//!   on a WebSocket, so the choice was the subprotocol or a `?token=` query. The subprotocol wins:
//!   URLs end up in access logs, proxy logs, tracing spans and devtools/resource-timing entries,
//!   while the `Sec-WebSocket-Protocol` header normally is not logged. The gateway answers with
//!   `rlmesh.v2` only, never echoes the token, and keeps only hashes of tokens
//!   ([`pairing::token_hash`]). A token is either
//!   - this machine's page token (`RLMESH_GATEWAY_TOKEN` or `--token-file`, which the page gets in
//!     its URL fragment, `/#t=…`, never sent to a server): its browser attaches with the gateway's
//!     own credential, at `--max-priority` at most; or
//!   - a browser token paired with the host (`rlmesh pair HOST --browser`, design §5): a device of
//!     its own, whose key the gateway attaches its browser with, so the host applies that device's
//!     cap, parts and verbs, and revoking it detaches exactly that browser.
//! - **Origin:** an allowlist. The default is the origin(s) this gateway serves its own page from,
//!   computed from the bind address and port (`http(s)://127.0.0.1:P`, `localhost:P`, `[::1]:P`),
//!   never from the request's `Host` header, so DNS rebinding (evil.example resolving to 127.0.0.1)
//!   does not pass. A request without `Origin` is not from a browser page and is allowed; the token
//!   still applies to it.
//! - **Limits per connection:** a maximum inbound message (and frame) size, and a cap on stream
//!   subscriptions. Over-cap and duplicate subscriptions are ignored and logged.

use axum::extract::ws::{CloseFrame, Message, WebSocketUpgrade};
use axum::extract::{FromRequestParts, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::warn;

use crate::auth::Credential;
use crate::crypto::Key;
use crate::pairing::{self, BrowserToken};

/// The subprotocol the gateway selects: the session protocol generation.
pub const SUBPROTOCOL: &str = "rlmesh.v2";
/// The prefix of the token subprotocol.
pub const TOKEN_PREFIX: &str = "rlmesh.token.";
/// The default cap on one inbound message (and frame).
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 64 * 1024;
/// The default cap on subscriptions per data connection.
pub const DEFAULT_MAX_SUBSCRIPTIONS: usize = 32;

/// Which page origins may open a socket.
#[derive(Debug, Clone, PartialEq)]
pub enum Origins {
    /// `--allow-origin '*'`: any origin (still subject to the token).
    Any,
    /// Exactly these (normalised) origins.
    List(Vec<String>),
}

/// The checks every upgrade and connection gets.
#[derive(Debug, Clone)]
pub struct Policy {
    /// The hash of this machine's page token ([`pairing::token_hash`]), if
    /// it has one.
    pub page: Option<[u8; 32]>,
    /// The browser tokens paired on this machine ([`BrowserToken`]), read
    /// at every upgrade.
    pub browsers: Option<PathBuf>,
    /// The Origin allowlist.
    pub origins: Origins,
    /// The largest inbound message (and frame).
    pub max_message_bytes: usize,
    /// The most subscriptions one data connection may hold.
    pub max_subscriptions: usize,
}

/// Whose token an upgrade presented: what its browser attaches with.
#[derive(Debug, Clone)]
pub enum Bearer {
    /// This machine's page token: the gateway's own credential.
    Page,
    /// A paired browser token: its device's.
    Browser {
        /// The device id.
        device: String,
        /// Its key and the host's.
        credential: Credential,
    },
}

/// Why an upgrade was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum Reject {
    /// No `rlmesh.token.<token>` subprotocol was offered.
    MissingToken,
    /// The token does not match.
    BadToken,
    /// The page origin is not allowed.
    BadOrigin(String),
}

impl Reject {
    /// The HTTP status: 401 for the token, 403 for the origin.
    pub fn status(&self) -> StatusCode {
        match self {
            Reject::MissingToken | Reject::BadToken => StatusCode::UNAUTHORIZED,
            Reject::BadOrigin(_) => StatusCode::FORBIDDEN,
        }
    }

    /// The WebSocket close code: 4401 for the token, 4403 for the origin
    /// (a browser cannot read the status of a refused upgrade).
    pub fn close_code(&self) -> u16 {
        4000 + self.status().as_u16()
    }
}

impl std::fmt::Display for Reject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reject::MissingToken => write!(
                f,
                "missing token (offer subprotocols {SUBPROTOCOL}, {TOKEN_PREFIX}<token>)"
            ),
            Reject::BadToken => write!(f, "bad token"),
            Reject::BadOrigin(o) => write!(f, "origin not allowed: {o}"),
        }
    }
}

/// `scheme://host[:port]`, lowercased, no trailing slash: the form browsers send in `Origin`.
pub fn normalize_origin(o: &str) -> String {
    o.trim().trim_end_matches('/').to_ascii_lowercase()
}

/// The origins the gateway's own page is served from, for a given bind address.
pub fn default_origins(bind: IpAddr, port: u16, tls: bool) -> Vec<String> {
    let scheme = if tls { "https" } else { "http" };
    let mut hosts: Vec<String> = vec![];
    if bind.is_loopback() || bind.is_unspecified() {
        hosts.extend([
            "127.0.0.1".to_string(),
            "localhost".to_string(),
            "[::1]".to_string(),
        ]);
    }
    if !bind.is_loopback() && !bind.is_unspecified() {
        hosts.push(match bind {
            IpAddr::V4(a) => a.to_string(),
            IpAddr::V6(a) => format!("[{a}]"),
        });
    }
    let default_port = if tls { 443 } else { 80 };
    hosts
        .into_iter()
        .map(|h| {
            if port == default_port {
                format!("{scheme}://{h}")
            } else {
                format!("{scheme}://{h}:{port}")
            }
        })
        .collect()
}

/// Token rules: RFC 7230 token characters only (it rides in a header list), at least 16 long.
pub fn validate_token(t: &str) -> Result<(), String> {
    if t.len() < 16 {
        return Err("token must be at least 16 characters".into());
    }
    if !t
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b"._~-".contains(&b))
    {
        return Err("token may only contain A-Z a-z 0-9 . _ ~ -".into());
    }
    Ok(())
}

/// The offered subprotocols, across repeated headers and comma lists.
fn offered_protocols(h: &HeaderMap) -> impl Iterator<Item = &str> {
    h.get_all(header::SEC_WEBSOCKET_PROTOCOL)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(str::trim)
}

impl Policy {
    /// Checks the Origin and the token of an upgrade request: whose token it
    /// is.
    pub fn check(&self, h: &HeaderMap) -> Result<Bearer, Reject> {
        if let Some(o) = h.get(header::ORIGIN) {
            let o = normalize_origin(o.to_str().unwrap_or(""));
            let ok = match &self.origins {
                Origins::Any => true,
                Origins::List(l) => l.contains(&o),
            };
            if !ok {
                return Err(Reject::BadOrigin(o));
            }
        }
        let got = offered_protocols(h)
            .find_map(|p| p.strip_prefix(TOKEN_PREFIX))
            .ok_or(Reject::MissingToken)?;
        if let Some(page) = self.page
            && Key::from_bytes(page).ct_eq(&Key::from_bytes(pairing::token_hash(got)))
        {
            return Ok(Bearer::Page);
        }
        let paired = match &self.browsers {
            Some(path) => BrowserToken::find(path, got).unwrap_or_else(|err| {
                warn!(%err, "browser tokens unreadable");
                None
            }),
            None => None,
        };
        paired
            .map(|token| Bearer::Browser {
                credential: token.credential(),
                device: token.device,
            })
            .ok_or(Reject::BadToken)
    }
}

/// Middleware for the WebSocket routes: rejects before the upgrade, with 401 or 403, and hands
/// the route the [`Bearer`].
pub async fn guard(State(policy): State<Arc<Policy>>, mut req: Request, next: Next) -> Response {
    let r = match policy.check(req.headers()) {
        Ok(bearer) => {
            req.extensions_mut().insert(bearer);
            return next.run(req).await;
        }
        Err(r) => r,
    };
    let peer = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_string();
    warn!(path = %req.uri().path(), origin = %peer, "rejected: {r}");
    // A browser cannot read the status of a refused upgrade, so a WebSocket
    // is accepted here, never reaching the route, and closed with the
    // reason (4401, 4403) for the page to report at once.
    let (mut parts, _) = req.into_parts();
    match WebSocketUpgrade::from_request_parts(&mut parts, &()).await {
        Ok(ws) => {
            let (code, reason) = (r.close_code(), r.to_string());
            ws.protocols([SUBPROTOCOL])
                .on_upgrade(move |mut socket| async move {
                    let _ = socket
                        .send(Message::Close(Some(CloseFrame {
                            code,
                            reason: reason.into(),
                        })))
                        .await;
                })
                .into_response()
        }
        Err(_) => (r.status(), r.to_string()).into_response(),
    }
}

/// Apply the per-connection limits and select `rlmesh.v2` on an upgrade.
pub fn limit<F>(
    ws: axum::extract::ws::WebSocketUpgrade<F>,
    p: &Policy,
) -> axum::extract::ws::WebSocketUpgrade<F> {
    ws.max_message_size(p.max_message_bytes)
        .max_frame_size(p.max_message_bytes)
        .protocols([SUBPROTOCOL])
}

/// The paths one data connection subscribed to, capped.
#[derive(Debug)]
pub struct SubscriptionSet {
    paths: Vec<String>,
    cap: usize,
}

/// What [`SubscriptionSet::add`] did.
#[derive(Debug, PartialEq)]
pub enum SubAdd {
    /// A new path.
    New,
    /// Already subscribed: ignored.
    Duplicate,
    /// Over the cap: ignored.
    OverCap,
}

impl SubscriptionSet {
    /// An empty set of at most `cap` paths.
    pub fn new(cap: usize) -> Self {
        Self { paths: vec![], cap }
    }

    /// Adds `path` unless it is a duplicate or over the cap.
    pub fn add(&mut self, path: &str) -> SubAdd {
        if self.paths.iter().any(|p| p == path) {
            SubAdd::Duplicate
        } else if self.paths.len() >= self.cap {
            SubAdd::OverCap
        } else {
            self.paths.push(path.into());
            SubAdd::New
        }
    }
    /// Removes `path`.
    pub fn remove(&mut self, path: &str) {
        self.paths.retain(|p| p != path);
    }

    /// Paths held.
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Whether none is held.
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn policy() -> Policy {
        Policy {
            page: Some(pairing::token_hash(TOK)),
            browsers: None,
            origins: Origins::List(default_origins("127.0.0.1".parse().unwrap(), 18080, false)),
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
            max_subscriptions: 2,
        }
    }

    fn headers(origin: Option<&str>, protos: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(o) = origin {
            h.insert(header::ORIGIN, HeaderValue::from_str(o).unwrap());
        }
        if let Some(p) = protos {
            h.insert(
                header::SEC_WEBSOCKET_PROTOCOL,
                HeaderValue::from_str(p).unwrap(),
            );
        }
        h
    }

    const TOK: &str = "0123456789abcdef-test";

    fn check(p: &Policy, origin: Option<&str>, protos: Option<&str>) -> Result<(), Reject> {
        p.check(&headers(origin, protos)).map(|_| ())
    }

    #[test]
    fn default_origins_are_the_served_origin_not_host_derived() {
        let o = default_origins("127.0.0.1".parse().unwrap(), 18080, false);
        assert_eq!(
            o,
            [
                "http://127.0.0.1:18080",
                "http://localhost:18080",
                "http://[::1]:18080"
            ]
        );
        assert_eq!(
            default_origins("10.1.2.3".parse().unwrap(), 443, true),
            ["https://10.1.2.3"]
        );
        assert_eq!(
            default_origins("0.0.0.0".parse().unwrap(), 8080, false).len(),
            3
        );
    }

    #[test]
    fn every_socket_needs_a_token() {
        let p = policy();
        let ok = format!("{SUBPROTOCOL}, {TOKEN_PREFIX}{TOK}");
        assert_eq!(
            check(&p, Some("http://127.0.0.1:18080"), None),
            Err(Reject::MissingToken)
        );
        assert_eq!(
            check(&p, None, Some(SUBPROTOCOL)),
            Err(Reject::MissingToken)
        );
        assert_eq!(
            check(
                &p,
                None,
                Some(&format!("{SUBPROTOCOL}, {TOKEN_PREFIX}nope"))
            ),
            Err(Reject::BadToken)
        );
        assert!(matches!(
            p.check(&headers(Some("http://localhost:18080/"), Some(&ok))),
            Ok(Bearer::Page)
        ));
        assert_eq!(Reject::MissingToken.close_code(), 4401);
    }

    #[test]
    fn origin_allowlist() {
        let p = policy();
        let ok = format!("{SUBPROTOCOL}, {TOKEN_PREFIX}{TOK}");
        let ok = Some(ok.as_str());
        assert_eq!(check(&p, Some("http://127.0.0.1:18080"), ok), Ok(()));
        assert_eq!(check(&p, Some("HTTP://LOCALHOST:18080"), ok), Ok(()));
        // DNS rebinding: the page origin is evil even though it resolves to loopback.
        let r = check(&p, Some("http://evil.example:18080"), ok);
        assert_eq!(
            r,
            Err(Reject::BadOrigin("http://evil.example:18080".into()))
        );
        assert_eq!(r.unwrap_err().status(), StatusCode::FORBIDDEN);
        assert!(check(&p, Some("http://127.0.0.1:9999"), ok).is_err());
        // No Origin: not a browser page; allowed (the token still applies).
        assert_eq!(check(&p, None, ok), Ok(()));
        let any = Policy {
            origins: Origins::Any,
            ..policy()
        };
        assert_eq!(check(&any, Some("http://evil.example"), ok), Ok(()));
    }

    #[test]
    fn token_rules() {
        assert!(validate_token(TOK).is_ok());
        assert!(validate_token("short").is_err());
        assert!(validate_token("has space in it 0123456").is_err());
        assert!(validate_token("comma,breaks,the,list,0123").is_err());
    }

    #[test]
    fn subscription_cap_and_dedupe() {
        let mut s = SubscriptionSet::new(2);
        assert_eq!(s.add("a"), SubAdd::New);
        assert_eq!(s.add("a"), SubAdd::Duplicate);
        assert_eq!(s.add("b"), SubAdd::New);
        assert_eq!(s.add("c"), SubAdd::OverCap);
        assert_eq!(s.len(), 2);
    }
}
