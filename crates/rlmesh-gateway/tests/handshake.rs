//! Real WebSocket handshakes against the gateway's security layer (no
//! session needed): the same `security::guard` middleware and
//! `security::limit` upgrade settings the gateway uses, on an echo route.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocketUpgrade};
use axum::middleware;
use axum::response::IntoResponse;
use axum::routing::get;
use futures::{SinkExt, StreamExt};
use rlmesh_gateway::security::{self, Origins, Policy, SUBPROTOCOL, TOKEN_PREFIX};
use rlmesh_gateway::tls;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{self as tg, http};

const TOKEN: &str = "test-token-0123456789abcdef";

async fn echo(ws: WebSocketUpgrade, State(p): State<Arc<Policy>>) -> impl IntoResponse {
    security::limit(ws, &p).on_upgrade(|mut s| async move {
        while let Some(Ok(m)) = s.recv().await {
            if let Message::Binary(_) | Message::Text(_) = m
                && s.send(m).await.is_err()
            {
                break;
            }
        }
    })
}

fn policy(port: u16, token: Option<&str>, tls: bool) -> Arc<Policy> {
    Arc::new(Policy {
        token: token.map(Into::into),
        origins: Origins::List(security::default_origins(
            "127.0.0.1".parse().unwrap(),
            port,
            tls,
        )),
        max_message_bytes: 1024,
        max_subscriptions: 4,
    })
}

fn app(p: Arc<Policy>) -> Router {
    Router::new()
        .route("/control", get(echo))
        .route_layer(middleware::from_fn_with_state(p.clone(), security::guard))
        .route("/", get(|| async { "page" }))
        .with_state(p)
}

/// Bind first so the default origin can use the real port.
async fn serve(token: Option<&str>) -> (SocketAddr, Arc<Policy>) {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    let p = policy(addr.port(), token, false);
    let a = app(p.clone());
    tokio::spawn(async move { axum::serve(l, a).await.unwrap() });
    (addr, p)
}

fn request(url: &str, origin: Option<&str>, protocols: Option<&str>) -> http::Request<()> {
    let mut r = url.into_client_request().unwrap();
    if let Some(o) = origin {
        r.headers_mut().insert("origin", o.parse().unwrap());
    }
    if let Some(p) = protocols {
        r.headers_mut()
            .insert("sec-websocket-protocol", p.parse().unwrap());
    }
    r
}

/// The close code a refused upgrade is closed with: the handshake succeeds
/// (a browser could not read a refused one's status), then the gateway
/// closes before any route sees the connection.
async fn refused_with(req: http::Request<()>) -> u16 {
    let (mut ws, _) = tokio_tungstenite::connect_async(req)
        .await
        .expect("the upgrade completes, to carry the reason");
    match ws.next().await {
        Some(Ok(tg::Message::Close(Some(frame)))) => u16::from(frame.code),
        other => panic!("expected a close frame, got {other:?}"),
    }
}

fn with_token() -> String {
    format!("{SUBPROTOCOL}, {TOKEN_PREFIX}{TOKEN}")
}

#[tokio::test]
async fn rejected_without_token() {
    let (addr, _) = serve(Some(TOKEN)).await;
    let url = format!("ws://{addr}/control");
    let good_origin = format!("http://127.0.0.1:{}", addr.port());
    assert_eq!(
        refused_with(request(&url, Some(&good_origin), Some(SUBPROTOCOL))).await,
        4401
    );
    let wrong = format!("{SUBPROTOCOL}, {TOKEN_PREFIX}not-the-right-token-000");
    assert_eq!(
        refused_with(request(&url, Some(&good_origin), Some(&wrong))).await,
        4401
    );
    // The page itself is not behind the token.
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut c = tokio::net::TcpStream::connect(addr).await.unwrap();
    c.write_all(b"GET / HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut body = String::new();
    c.read_to_string(&mut body).await.unwrap();
    assert!(
        body.starts_with("HTTP/1.1 200") && body.ends_with("page"),
        "{body}"
    );
}

#[tokio::test]
async fn rejected_from_bad_origin() {
    let (addr, _) = serve(Some(TOKEN)).await;
    let url = format!("ws://{addr}/control");
    // DNS-rebinding shape: right port, hostile name.
    assert_eq!(
        refused_with(request(
            &url,
            Some(&format!("http://evil.example:{}", addr.port())),
            Some(&with_token()),
        ))
        .await,
        4403
    );
    assert_eq!(
        refused_with(request(
            &url,
            Some("https://127.0.0.1"),
            Some(&with_token()),
        ))
        .await,
        4403
    );
}

#[tokio::test]
async fn accepted_from_good_origin_with_token_and_limits_apply() {
    let (addr, _) = serve(Some(TOKEN)).await;
    let url = format!("ws://{addr}/control");
    for origin in [
        format!("http://127.0.0.1:{}", addr.port()),
        format!("http://localhost:{}", addr.port()),
    ] {
        let (mut ws, resp) =
            tokio_tungstenite::connect_async(request(&url, Some(&origin), Some(&with_token())))
                .await
                .unwrap();
        // Only rlmesh.v2 is selected; the token is never echoed.
        assert_eq!(
            resp.headers().get("sec-websocket-protocol").unwrap(),
            SUBPROTOCOL
        );
        ws.send(tg::Message::binary(vec![1u8; 512])).await.unwrap();
        assert_eq!(ws.next().await.unwrap().unwrap().into_data().len(), 512);
        // Over the 1 KiB limit: the server closes the connection.
        ws.send(tg::Message::binary(vec![2u8; 4096])).await.unwrap();
        let next = ws.next().await;
        assert!(
            matches!(next, None | Some(Err(_)) | Some(Ok(tg::Message::Close(_)))),
            "oversize message should close the socket, got {next:?}"
        );
    }
}

#[tokio::test]
async fn loopback_without_token_accepts_non_browser_clients() {
    let (addr, _) = serve(None).await;
    let (_ws, _) = tokio_tungstenite::connect_async(request(
        &format!("ws://{addr}/control"),
        None,
        Some(SUBPROTOCOL),
    ))
    .await
    .unwrap();
    // Without a token, Origin is still enforced for browser pages.
    assert_eq!(
        refused_with(request(
            &format!("ws://{addr}/control"),
            Some("http://evil.example"),
            None,
        ))
        .await,
        4403
    );
}

#[tokio::test]
async fn wss_with_rustls() {
    // Self-signed cert for 127.0.0.1, written under the target dir (not /tmp).
    let ck =
        rcgen::generate_simple_self_signed(vec!["127.0.0.1".into(), "localhost".into()]).unwrap();
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("gw-tls-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (cert, key) = (dir.join("cert.pem"), dir.join("key.pem"));
    std::fs::write(&cert, ck.cert.pem()).unwrap();
    std::fs::write(&key, ck.signing_key.serialize_pem()).unwrap();
    let cfg = tls::load_config(&cert, &key).unwrap();

    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    let p = policy(addr.port(), Some(TOKEN), true);
    let a = app(p);
    tokio::spawn(async move {
        axum::serve(tls::TlsListener::new(l, cfg).unwrap(), a)
            .await
            .unwrap()
    });

    let mut roots = rustls::RootCertStore::empty();
    roots.add(ck.cert.der().clone()).unwrap();
    let client = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_root_certificates(roots)
    .with_no_client_auth();
    let connector = tokio_rustls::TlsConnector::from(Arc::new(client));
    let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
    let tls_stream = connector
        .connect("127.0.0.1".try_into().unwrap(), tcp)
        .await
        .unwrap();
    let origin = format!("https://127.0.0.1:{}", addr.port());
    let req = request(
        &format!("wss://127.0.0.1:{}/control", addr.port()),
        Some(&origin),
        Some(&with_token()),
    );
    let (mut ws, _) = tokio_tungstenite::client_async(req, tls_stream)
        .await
        .unwrap();
    ws.send(tg::Message::binary(vec![7u8; 16])).await.unwrap();
    assert_eq!(
        ws.next().await.unwrap().unwrap().into_data().as_ref(),
        &[7u8; 16]
    );
    std::fs::remove_dir_all(&dir).ok();
}
