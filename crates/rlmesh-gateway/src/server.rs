//! The browser side: one WebSocket per traffic class, behind the security
//! layer, and the static page with COOP/COEP.
//!
//! - `/control`: the first message is a JSON hello (`{"type": "hello",
//!   "priority": "model"}`); the gateway attaches a participant for the
//!   connection and answers with its id, then the authority snapshot. From
//!   then on binary frames carry `ctl` requests and replies, heartbeats and
//!   echoes, chunks and authority updates ([`crate::frame`]).
//! - `/state` and `/bulk`: JSON `{"type": "subscribe", "id": 0, "path":
//!   "arm7/arm"}` (and `unsubscribe`); samples come back as `SAMPLE` frames.
//!   State connections queue 64 frames, bulk ones 4; a full queue drops the
//!   newest frame (the stream's own seq shows the loss).

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderName, HeaderValue};
use axum::middleware;
use axum::response::IntoResponse;
use axum::routing::get;
use bytes::Bytes;
use futures::{FutureExt, SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;
use tracing::{debug, info, warn};

use crate::bridge::{Gateway, Outbound, report};
use crate::config::parse_priority;
use crate::security::{self, Policy, SubAdd, SubscriptionSet};

/// How long a control connection may take to say hello.
const HELLO_TIMEOUT: Duration = Duration::from_secs(10);

/// What every route shares.
#[derive(Clone)]
pub struct AppState {
    /// The session end.
    pub gateway: Arc<Gateway>,
    /// The security policy.
    pub policy: Arc<Policy>,
}

/// The router: the three class sockets behind the guard, and `static_dir`
/// (when given) at `/`, every response cross-origin isolated.
pub fn router(state: AppState, static_dir: Option<&std::path::Path>) -> Router {
    let header = |k: &'static str, v: &'static str| {
        SetResponseHeaderLayer::overriding(HeaderName::from_static(k), HeaderValue::from_static(v))
    };
    let mut app = Router::new()
        .route("/control", get(control))
        .route(
            "/state",
            get(|ws: WebSocketUpgrade, st: State<AppState>| data(ws, st, "state")),
        )
        .route(
            "/bulk",
            get(|ws: WebSocketUpgrade, st: State<AppState>| data(ws, st, "bulk")),
        )
        // Token and Origin checks on the sockets only; the page is public.
        .route_layer(middleware::from_fn_with_state(
            state.policy.clone(),
            security::guard,
        ));
    if let Some(dir) = static_dir {
        app = app.fallback_service(ServeDir::new(dir));
    }
    app.layer(header("cross-origin-opener-policy", "same-origin"))
        .layer(header("cross-origin-embedder-policy", "require-corp"))
        .layer(header("cross-origin-resource-policy", "same-origin"))
        .layer(header("cache-control", "no-store"))
        .with_state(state)
}

/// The close code of a session that ended (its host is gone).
pub const SESSION_ENDED: u16 = 4410;

fn to_message(out: Outbound) -> Message {
    match out {
        Outbound::Binary(b) => Message::Binary(b),
        Outbound::Text(t) => Message::Text(t.into()),
        Outbound::Close(code, reason) => Message::Close(Some(CloseFrame {
            code,
            reason: reason.into(),
        })),
    }
}

/// A new connection to a session that already ended is closed at once.
fn refusal(st: &AppState) -> Option<(u16, String)> {
    (*st.gateway.ended().borrow()).then(|| (SESSION_ENDED, "the session ended".to_string()))
}

async fn close_with(mut socket: WebSocket, code: u16, reason: String) {
    let _ = socket.send(to_message(Outbound::Close(code, reason))).await;
}

/// Resolves once the session ended.
async fn session_ended(mut ended: tokio::sync::watch::Receiver<bool>) {
    if ended.wait_for(|e| *e).await.is_err() {
        std::future::pending::<()>().await;
    }
}

async fn control(ws: WebSocketUpgrade, State(st): State<AppState>) -> impl IntoResponse {
    let ws = security::limit(ws, &st.policy);
    match refusal(&st) {
        Some((code, reason)) => ws.on_upgrade(move |socket| close_with(socket, code, reason)),
        None => ws.on_upgrade(move |socket| control_conn(socket, st)),
    }
}

async fn control_conn(socket: WebSocket, st: AppState) {
    let (mut tx, mut rx) = socket.split();
    let (out, mut out_rx) = mpsc::channel::<Outbound>(256);
    let writer = tokio::spawn(async move {
        while let Some(m) = out_rx.recv().await {
            if tx.send(to_message(m)).await.is_err() {
                break;
            }
        }
        let _ = tx.close().await;
    });
    // The hello names the class this connection's participant attaches at.
    let hello = tokio::time::timeout(HELLO_TIMEOUT, async {
        while let Some(Ok(m)) = rx.next().await {
            if let Message::Text(t) = m {
                return serde_json::from_str::<Value>(&t).ok();
            }
        }
        None
    })
    .await
    .ok()
    .flatten();
    let priority = match hello
        .as_ref()
        .filter(|h| h["type"] == "hello")
        .map(|h| parse_priority(h["priority"].as_str().unwrap_or("model")))
    {
        Some(Ok(priority)) => priority,
        Some(Err(err)) => {
            let _ = out.send(report("error", err)).await;
            drop(out);
            let _ = writer.await;
            return;
        }
        None => {
            let _ = out
                .send(report("error", "expected {\"type\": \"hello\"} first"))
                .await;
            drop(out);
            let _ = writer.await;
            return;
        }
    };
    let mut link = match st.gateway.attach(priority, out.clone()).await {
        Ok(link) => link,
        Err(err) => {
            warn!(%err, "browser attach");
            let _ = out.send(report("error", format!("attach: {err}"))).await;
            drop(out);
            let _ = writer.await;
            return;
        }
    };
    info!(participant = %link.id(), ?priority, "browser attached");
    let _ = out.send(Outbound::Text(link.hello(priority))).await;
    if let Some(state) = link.state_frame() {
        let _ = out.send(Outbound::Binary(state)).await;
    }
    if let Err(err) = link.start(out.clone()).await {
        warn!(participant = %link.id(), %err, "binding");
        let _ = out.send(report("error", format!("binding: {err}"))).await;
    }
    let ended = session_ended(st.gateway.ended());
    tokio::pin!(ended);
    loop {
        let m = tokio::select! {
            m = rx.next() => m,
            () = &mut ended => {
                let _ = out.send(Outbound::Close(SESSION_ENDED, "the session ended".into())).await;
                break;
            }
        };
        let Some(Ok(m)) = m else { break };
        match m {
            Message::Binary(b) => {
                if let Some(reply) = link.on_frame(b).await
                    && out.send(reply).await.is_err()
                {
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    drop(link);
    drop(out);
    // Let a queued close frame go out.
    let _ = tokio::time::timeout(Duration::from_millis(500), writer).await;
}

async fn data(
    ws: WebSocketUpgrade,
    st: State<AppState>,
    socket: &'static str,
) -> impl IntoResponse {
    let State(st) = st;
    let ws = security::limit(ws, &st.policy);
    match refusal(&st) {
        Some((code, reason)) => ws.on_upgrade(move |s| close_with(s, code, reason)),
        None => ws.on_upgrade(move |s| data_conn(s, st, socket)),
    }
}

async fn data_conn(socket: WebSocket, st: AppState, class: &'static str) {
    let (mut tx, mut rx) = socket.split();
    let (frames, mut frames_rx) = mpsc::channel::<Bytes>(if class == "state" { 64 } else { 4 });
    let (replies, mut replies_rx) = mpsc::channel::<String>(32);
    let mut ended = st.gateway.ended();
    let writer = tokio::spawn(async move {
        loop {
            let m = tokio::select! {
                Some(b) = frames_rx.recv() => Message::Binary(b),
                Some(t) = replies_rx.recv() => Message::Text(t.into()),
                Ok(()) = ended.wait_for(|e| *e).map(|r| r.map(|_| ())) => {
                    let _ = tx.send(to_message(Outbound::Close(SESSION_ENDED, "the session ended".into()))).await;
                    break;
                }
                else => break,
            };
            if tx.send(m).await.is_err() {
                break;
            }
        }
    });
    let mut paths = SubscriptionSet::new(st.policy.max_subscriptions);
    let mut subs: Vec<(u32, String, zenoh::pubsub::Subscriber<()>)> = Vec::new();
    while let Some(Ok(m)) = rx.next().await {
        let Message::Text(t) = m else {
            if matches!(m, Message::Close(_)) {
                break;
            }
            continue;
        };
        let Ok(msg) = serde_json::from_str::<Value>(&t) else {
            let _ = replies.try_send(json!({"type": "error", "message": "not JSON"}).to_string());
            continue;
        };
        let id = msg["id"].as_u64().and_then(|i| u32::try_from(i).ok());
        let path = msg["path"].as_str().unwrap_or_default().to_string();
        match (msg["type"].as_str(), id) {
            (Some("subscribe"), Some(id)) => {
                match paths.add(&path) {
                    SubAdd::New => {}
                    SubAdd::Duplicate => continue,
                    SubAdd::OverCap => {
                        debug!(class, %path, cap = st.policy.max_subscriptions, "subscription over the cap ignored");
                        let _ = replies.try_send(
                            json!({"type": "error", "id": id, "message": format!("over the cap of {} subscriptions", st.policy.max_subscriptions)})
                                .to_string(),
                        );
                        continue;
                    }
                }
                match st.gateway.subscribe(&path, class, id, frames.clone()).await {
                    Ok((sub, channel)) => {
                        debug!(class, %path, id, "subscribed");
                        subs.push((id, path.clone(), sub));
                        let _ = replies.try_send(
                            json!({"type": "subscribed", "id": id, "path": path, "channel": channel.0}).to_string(),
                        );
                    }
                    Err(err) => {
                        paths.remove(&path);
                        let _ = replies.try_send(
                            json!({"type": "error", "id": id, "path": path, "message": err})
                                .to_string(),
                        );
                    }
                }
            }
            (Some("unsubscribe"), Some(id)) => {
                if let Some(i) = subs.iter().position(|(sid, _, _)| *sid == id) {
                    let (_, path, _) = subs.remove(i);
                    paths.remove(&path);
                }
            }
            _ => {
                let _ = replies.try_send(json!({"type": "error", "message": "expected subscribe or unsubscribe with an id"}).to_string());
            }
        }
    }
    drop(subs);
    writer.abort();
    let c = &st.gateway.data[usize::from(class == "bulk")];
    debug!(
        class,
        forwarded = c.forwarded.load(Ordering::Relaxed),
        dropped = c.dropped.load(Ordering::Relaxed),
        "data connection closed"
    );
}
