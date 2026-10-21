//! Who may open a gateway WebSocket, and what one connection may do.
//!
//! - **Bind:** loopback (127.0.0.1) by default. A non-loopback bind refuses to start without a token.
//! - **Token:** carried in the WebSocket subprotocol list, as `rlmesh.token.<token>`, next to
//!   `rlmesh.v2`. Browsers cannot set headers on a WebSocket, so the choice was the subprotocol or a
//!   `?token=` query. The subprotocol wins: URLs end up in access logs, proxy logs, tracing spans and
//!   devtools/resource-timing entries, while the `Sec-WebSocket-Protocol` header normally is not
//!   logged. The gateway answers with `rlmesh.v2` only and never echoes the token back. Tokens are
//!   restricted to RFC 7230 token characters (`[A-Za-z0-9._~-]`) and at least 16 characters long.
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
use std::sync::Arc;
use tracing::warn;

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
    /// The bearer token, when one is required.
    pub token: Option<String>,
    /// The Origin allowlist.
    pub origins: Origins,
    /// The largest inbound message (and frame).
    pub max_message_bytes: usize,
    /// The most subscriptions one data connection may hold.
    pub max_subscriptions: usize,
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

/// A non-loopback bind needs a token.
pub fn validate_bind(bind: IpAddr, token: Option<&str>) -> Result<(), String> {
    if !bind.is_loopback() && token.is_none() {
        return Err(format!(
            "refusing to bind {bind} without a token: set RLMESH_GATEWAY_TOKEN or --token-file (or bind 127.0.0.1)"
        ));
    }
    Ok(())
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
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
    /// Checks the Origin and the token of an upgrade request.
    pub fn check(&self, h: &HeaderMap) -> Result<(), Reject> {
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
        if let Some(want) = &self.token {
            let got = offered_protocols(h)
                .find_map(|p| p.strip_prefix(TOKEN_PREFIX))
                .ok_or(Reject::MissingToken)?;
            if !ct_eq(got.as_bytes(), want.as_bytes()) {
                return Err(Reject::BadToken);
            }
        }
        Ok(())
    }
}

/// Middleware for the WebSocket routes: rejects before the upgrade, with 401 or 403.
pub async fn guard(State(policy): State<Arc<Policy>>, req: Request, next: Next) -> Response {
    let Err(r) = policy.check(req.headers()) else {
        return next.run(req).await;
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

    fn policy(token: Option<&str>) -> Policy {
        Policy {
            token: token.map(Into::into),
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
    fn token_required_when_configured() {
        let p = policy(Some(TOK));
        let ok = format!("{SUBPROTOCOL}, {TOKEN_PREFIX}{TOK}");
        assert_eq!(
            p.check(&headers(Some("http://127.0.0.1:18080"), None)),
            Err(Reject::MissingToken)
        );
        assert_eq!(
            p.check(&headers(None, Some(SUBPROTOCOL))),
            Err(Reject::MissingToken)
        );
        assert_eq!(
            p.check(&headers(
                None,
                Some(&format!("{SUBPROTOCOL}, {TOKEN_PREFIX}nope"))
            )),
            Err(Reject::BadToken)
        );
        assert_eq!(
            p.check(&headers(Some("http://localhost:18080/"), Some(&ok))),
            Ok(())
        );
        assert_eq!(Reject::MissingToken.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn origin_allowlist() {
        let p = policy(None);
        assert_eq!(
            p.check(&headers(Some("http://127.0.0.1:18080"), None)),
            Ok(())
        );
        assert_eq!(
            p.check(&headers(Some("HTTP://LOCALHOST:18080"), None)),
            Ok(())
        );
        // DNS rebinding: the page origin is evil even though it resolves to loopback.
        let r = p.check(&headers(Some("http://evil.example:18080"), None));
        assert_eq!(
            r,
            Err(Reject::BadOrigin("http://evil.example:18080".into()))
        );
        assert_eq!(r.unwrap_err().status(), StatusCode::FORBIDDEN);
        assert!(
            p.check(&headers(Some("http://127.0.0.1:9999"), None))
                .is_err()
        );
        // No Origin: not a browser page; allowed (the token, when configured, still applies).
        assert_eq!(p.check(&headers(None, None)), Ok(()));
        let any = Policy {
            origins: Origins::Any,
            ..policy(None)
        };
        assert_eq!(
            any.check(&headers(Some("http://evil.example"), None)),
            Ok(())
        );
    }

    #[test]
    fn bind_and_token_rules() {
        assert!(validate_bind("127.0.0.1".parse().unwrap(), None).is_ok());
        assert!(validate_bind("0.0.0.0".parse().unwrap(), None).is_err());
        assert!(validate_bind("0.0.0.0".parse().unwrap(), Some(TOK)).is_ok());
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
