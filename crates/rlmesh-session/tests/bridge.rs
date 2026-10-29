#![cfg(feature = "gateway")]
//! The gateway against a real session host and robot: a browser connection
//! is a participant of its own, its heartbeats and chunks are its own, its
//! leases are the host's, and samples arrive byte for byte as published. A
//! paired browser token is a device of its own: the host applies its entry,
//! revoking it detaches exactly its browser, and no token reaches a log.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use rlmesh_session::auth::Device;
use rlmesh_session::authority::DeviceAccess;
use rlmesh_session::crypto::StaticKeypair;
use rlmesh_session::crypto::sealed::unverified_body;
use rlmesh_session::edge::{EdgeConfig, Robot};
use rlmesh_session::gateway::security::{Origins, Policy, SUBPROTOCOL, TOKEN_PREFIX};
use rlmesh_session::gateway::{AppState, Gateway, frame, router};
use rlmesh_session::host::OsRng;
use rlmesh_session::pairing::{BrowserToken, Paired, new_token, token_hash};
use rlmesh_session::transport::{open_local_listeners, payload_bytes};
use rlmesh_session::wire::codec;
use rlmesh_session::wire::{
    AuthoritySnapshot, Command, CtlStatus, Epoch, Health, Heartbeat, Mode, PairInfo, ParticipantId,
    PriorityClass, Request, Role, SessionId, SessionKeys, TenantId,
};
use rlmesh_session::{
    HostConfig, Participant, ParticipantConfig, SessionHost, SystemMonotonicClock, Transport,
    TransportConfig,
};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{self as tg};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

const WAIT: Duration = Duration::from_secs(10);
/// This machine's page token.
const PAGE: &str = "page-token-0123456789abcdef";
const ROBOT: &str = r#"{"arm": {"kind": "joints", "count": 2, "fields": ["pos"], "hz": 200, "command": "position",
                       "hold_max_ms": 100, "max_action_age_ms": 1000}}"#;

struct World {
    base: std::net::SocketAddr,
    tenant: TenantId,
    session: SessionId,
    url: String,
    stop: Arc<AtomicBool>,
    host: SessionHost,
    browsers: tempfile::TempDir,
    _robot: std::thread::JoinHandle<()>,
}

async fn open(base: std::net::SocketAddr) -> Transport {
    // Peers listen too, so the others they learn of can dial them.
    let any = "127.0.0.1:0".parse().unwrap();
    Transport::open(&TransportConfig::per_class_tcp(Some(any), Some(base)).unwrap())
        .await
        .unwrap()
}

impl World {
    async fn start() -> Self {
        let (transport, base) = open_local_listeners("127.0.0.1".parse().unwrap())
            .await
            .unwrap();
        let tenant = TenantId::new("gateway").unwrap();
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.clock = SystemMonotonicClock.shared();
        config.health_log = None;
        let host = SessionHost::start(transport, config).await.unwrap();
        let mut pc = ParticipantConfig::new(
            tenant.clone(),
            session,
            ParticipantId::new("bot").unwrap(),
            vec![Role::Embodiment],
            PriorityClass::Scripted,
        );
        pc.clock = SystemMonotonicClock.shared();
        pc.credential = Some(host.credential());
        let embodiment = Participant::attach(open(base).await, pc).await.unwrap();
        let robot = Arc::new(
            Robot::declare(embodiment, ROBOT, EdgeConfig::default())
                .await
                .unwrap(),
        );
        robot.participant().arm().await.unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let spin = std::thread::spawn({
            let (robot, stop) = (robot.clone(), stop.clone());
            move || {
                let clock = robot.participant().clock().clone();
                let mut q = 0.0_f32;
                while !stop.load(Ordering::Relaxed) {
                    let _ = robot.due("arm");
                    q += 0.001;
                    let value: Vec<u8> = [q, -q].iter().flat_map(|v| v.to_le_bytes()).collect();
                    robot
                        .publish("arm", clock.now_ns(), &[&value])
                        .expect("publish");
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        });
        let gateway = Gateway::open(
            open(base).await,
            tenant.clone(),
            session,
            host.credential(),
            PriorityClass::Operator,
        )
        .await
        .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let browsers = tempfile::tempdir().unwrap();
        let policy = Arc::new(Policy {
            page: Some(token_hash(PAGE)),
            browsers: Some(browsers.path().join("browsers.toml")),
            origins: Origins::List(Vec::new()),
            max_message_bytes: 1 << 16,
            max_subscriptions: 4,
        });
        let app = router(AppState { gateway, policy }, None);
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            base,
            tenant,
            session,
            url: format!("ws://{addr}"),
            stop,
            host,
            browsers,
            _robot: spin,
        }
    }

    async fn socket(&self, class: &str) -> Ws {
        self.socket_with(class, PAGE).await
    }

    async fn socket_with(&self, class: &str, token: &str) -> Ws {
        let mut request = format!("{}/{class}", self.url)
            .into_client_request()
            .unwrap();
        request.headers_mut().insert(
            "sec-websocket-protocol",
            format!("{SUBPROTOCOL}, {TOKEN_PREFIX}{token}")
                .parse()
                .unwrap(),
        );
        tokio_tungstenite::connect_async(request).await.unwrap().0
    }

    /// A browser token paired as `device` with `access`, as `rlmesh pair
    /// --browser` and `pair approve` leave it: the host admits the device,
    /// the gateway's file keeps the token's hash and key.
    async fn browser(&self, device: &str, access: DeviceAccess) -> String {
        let key = StaticKeypair::generate(&mut OsRng).unwrap();
        self.host
            .admit(Device {
                id: device.into(),
                public: *key.public(),
                access,
            })
            .await
            .unwrap();
        let paired = Paired {
            info: PairInfo {
                device: device.into(),
                host_key: *self.host.credential().host_static(),
                host_id: String::new(),
                addresses: Vec::new(),
            },
            device: key,
        };
        let token = new_token().unwrap();
        BrowserToken::new(&paired, &token)
            .add(&self.browsers.path().join("browsers.toml"))
            .unwrap();
        token
    }

    /// A control socket with `token`, its hello at `priority` answered:
    /// the socket and its participant.
    async fn control_with(&self, token: &str, priority: &str) -> (Ws, ParticipantId) {
        let mut control = self.socket_with("control", token).await;
        control
            .send(tg::Message::text(format!(
                r#"{{"type": "hello", "priority": "{priority}"}}"#
            )))
            .await
            .unwrap();
        let hello = text(&mut control).await;
        assert_eq!(hello["type"], "hello", "{hello}");
        let me = ParticipantId::new(hello["participant"].as_str().unwrap()).unwrap();
        (control, me)
    }
}

impl Drop for World {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

async fn next(ws: &mut Ws) -> tg::Message {
    tokio::time::timeout(WAIT, ws.next())
        .await
        .expect("a message in time")
        .expect("open")
        .expect("ok")
}

async fn text(ws: &mut Ws) -> Value {
    loop {
        if let tg::Message::Text(t) = next(ws).await {
            return serde_json::from_str(&t).unwrap();
        }
    }
}

/// The next binary frame with `tag`, skipping others.
async fn frame_with(ws: &mut Ws, tag: u8) -> Bytes {
    loop {
        if let tg::Message::Binary(b) = next(ws).await
            && b.first() == Some(&tag)
        {
            return b;
        }
    }
}

/// Snapshots until `predicate` holds.
async fn state_until(
    ws: &mut Ws,
    mut predicate: impl FnMut(&AuthoritySnapshot) -> bool,
) -> AuthoritySnapshot {
    loop {
        let b = frame_with(ws, frame::STATE).await;
        let s = codec::decode_state(&b[1..]).unwrap();
        if predicate(&s) {
            return s;
        }
    }
}

fn heartbeat(me: &ParticipantId, seq: u64, epoch: Epoch, t1_ns: i64) -> Vec<u8> {
    frame::tagged(
        frame::HEARTBEAT,
        &codec::encode_heartbeat(&Heartbeat {
            participant: me.clone(),
            epoch,
            mode: Mode::Armed,
            seq,
            t1_ns,
            previous: None,
            last_seq_seen: Vec::new(),
            last_chunk_accepted: 0,
            last_chunk_applied: 0,
            queue_depth: 0,
            health: Health::Ok,
        }),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn a_browser_is_a_participant_with_the_hosts_leases_and_its_own_heartbeats() {
    let world = World::start().await;
    let mut control = world.socket("control").await;
    control
        .send(tg::Message::text(
            r#"{"type": "hello", "priority": "operator"}"#,
        ))
        .await
        .unwrap();
    let hello = text(&mut control).await;
    assert_eq!(hello["type"], "hello", "{hello}");
    let me = ParticipantId::new(hello["participant"].as_str().unwrap()).unwrap();
    assert!(me.as_str().starts_with("web-"));
    assert_eq!(hello["session"], world.session.to_string());

    // The host's view: the browser is in the roster at its class, bound to
    // the robot's commandable part.
    let state = state_until(&mut control, |s| s.binding(&me).is_some()).await;
    let info = state.participant(&me).expect("in the roster");
    assert_eq!(info.priority, PriorityClass::Operator);
    let arm = state
        .part_at(&ParticipantId::new("bot").unwrap(), "arm")
        .unwrap()
        .channel;
    let binding = state.binding(&me).unwrap();
    assert_eq!(binding.channels, vec![arm]);

    // Heartbeats are the browser's: the echo carries its own t1.
    control
        .send(tg::Message::binary(heartbeat(
            &me,
            100,
            state.epoch,
            123_456,
        )))
        .await
        .unwrap();
    let echo = frame_with(&mut control, frame::ECHO).await;
    let echo = codec::decode_echo(&echo[1..]).unwrap();
    assert_eq!((echo.seq, echo.t1_ns), (100, 123_456));
    // Someone else's heartbeat is refused.
    let stranger = ParticipantId::new("someone").unwrap();
    control
        .send(tg::Message::binary(heartbeat(&stranger, 1, state.epoch, 0)))
        .await
        .unwrap();
    assert_eq!(text(&mut control).await["type"], "error");

    // The lease is the host's: granted through the browser's participant.
    let request = Request {
        from: me.clone(),
        epoch: Some(state.epoch),
        request_id: 7,
        command: Command::GrantLease {
            channels: vec![arm],
            release: Vec::new(),
            binding: binding.fingerprint,
            ttl_ns: Some(2_000_000_000),
        },
    };
    control
        .send(tg::Message::binary(frame::with_id(
            frame::REQUEST,
            7,
            &codec::encode_request(&request),
        )))
        .await
        .unwrap();
    let reply = frame_with(&mut control, frame::REPLY).await;
    assert_eq!(frame::read_id(&reply), Some(7));
    let reply = codec::decode_reply(&reply[5..]).unwrap();
    assert!(reply.result.is_ok(), "{:?}", reply.result);
    let lease = reply.leases.first().expect("granted").clone();
    assert_eq!((lease.holder.clone(), lease.channel), (me.clone(), arm));
    assert_eq!(reply.state.mode, Mode::Active);

    // A verb sent as somebody else is refused by the gateway.
    let forged = Request {
        from: stranger,
        epoch: None,
        request_id: 8,
        command: Command::Disarm,
    };
    control
        .send(tg::Message::binary(frame::with_id(
            frame::REQUEST,
            8,
            &codec::encode_request(&forged),
        )))
        .await
        .unwrap();
    let refused = frame_with(&mut control, frame::REPLY).await;
    assert!(codec::decode_reply(&refused[5..]).unwrap().result.is_err());

    // Without heartbeats the host lapses the lease (2 s ttl) and goes SAFE.
    let lapsed = state_until(&mut control, |s| s.lease(arm).is_none()).await;
    assert_eq!(lapsed.mode, Mode::Safe);

    // Dropping the connection drops the participant.
    drop(control);
    let observer = world.observer().await;
    let deadline = Instant::now() + WAIT;
    loop {
        if observer
            .authority()
            .is_some_and(|s| s.participant(&me).is_none())
        {
            break;
        }
        assert!(Instant::now() < deadline, "{me} still in the roster");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

impl World {
    async fn observer(&self) -> Participant {
        let mut config = ParticipantConfig::new(
            self.tenant.clone(),
            self.session,
            ParticipantId::new("test-observer").unwrap(),
            vec![Role::Observer],
            PriorityClass::Scripted,
        );
        config.credential = Some(self.host.credential());
        Participant::attach(open(self.base).await, config)
            .await
            .unwrap()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn samples_arrive_as_published_and_subscriptions_are_checked() {
    let world = World::start().await;
    // A raw subscriber on the same key: what the publisher put on the wire
    // (signed: its body is in clear).
    let transport = open(world.base).await;
    let keys = SessionKeys::new(&world.tenant, world.session);
    let (raw_tx, mut raw_rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
    let _raw = transport
        .session(rlmesh_session::TrafficClass::Realtime)
        .declare_subscriber(keys.data(&ParticipantId::new("bot").unwrap(), "arm"))
        .callback(move |s| {
            let _ = raw_tx.send(payload_bytes(s.payload()));
        })
        .await
        .unwrap();
    let mut state = world.socket("state").await;
    state
        .send(tg::Message::text(
            r#"{"type": "subscribe", "id": 3, "path": "bot/arm"}"#,
        ))
        .await
        .unwrap();
    let ok = text(&mut state).await;
    assert_eq!(ok["type"], "subscribed", "{ok}");
    let mut seen = std::collections::BTreeMap::new();
    let deadline = Instant::now() + WAIT;
    let matched = loop {
        assert!(Instant::now() < deadline, "no byte-identical sample");
        let frame_bytes = frame_with(&mut state, frame::SAMPLE).await;
        assert_eq!(frame::read_id(&frame_bytes), Some(3));
        assert_eq!(frame_bytes[5], 0, "a sample, not a batch");
        let body = frame_bytes.slice(6..);
        let sample = codec::decode_sample(body.clone()).unwrap();
        while let Ok(raw) = raw_rx.try_recv() {
            let raw = raw.slice(unverified_body(&raw).expect("signed"));
            let s = codec::decode_sample(raw.clone()).unwrap();
            seen.insert(s.seq, raw);
        }
        if let Some(raw) = seen.get(&sample.seq) {
            break raw == &body;
        }
    };
    assert!(matched, "the gateway re-encoded a sample");
    // The camera-class socket refuses a state stream, and garbage is an error.
    let mut bulk = world.socket("bulk").await;
    bulk.send(tg::Message::text(
        r#"{"type": "subscribe", "id": 0, "path": "bot/arm"}"#,
    ))
    .await
    .unwrap();
    let refused = text(&mut bulk).await;
    assert_eq!(refused["type"], "error");
    assert!(
        refused["message"]
            .as_str()
            .unwrap()
            .contains("state socket")
    );
    bulk.send(tg::Message::text("not json")).await.unwrap();
    assert_eq!(text(&mut bulk).await["type"], "error");
}

/// The close code `ws` is closed with, skipping what comes before it.
async fn closed_with(ws: &mut Ws) -> u16 {
    loop {
        match tokio::time::timeout(WAIT, ws.next())
            .await
            .expect("closed in time")
        {
            Some(Ok(tg::Message::Close(Some(frame)))) => return u16::from(frame.code),
            Some(Ok(_)) => {}
            other => panic!("expected a close frame, got {other:?}"),
        }
    }
}

/// Sends verb `command` as `me` and returns the host's verdict.
async fn verb(
    control: &mut Ws,
    me: &ParticipantId,
    id: u32,
    command: Command,
) -> Option<CtlStatus> {
    let request = Request {
        from: me.clone(),
        epoch: None,
        request_id: u64::from(id),
        command,
    };
    control
        .send(tg::Message::binary(frame::with_id(
            frame::REQUEST,
            id,
            &codec::encode_request(&request),
        )))
        .await
        .unwrap();
    loop {
        let reply = frame_with(control, frame::REPLY).await;
        if frame::read_id(&reply) == Some(id) {
            return codec::decode_reply(&reply[5..])
                .unwrap()
                .result
                .err()
                .map(|e| e.status);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_browser_token_attaches_as_its_device_with_its_cap_parts_and_verbs() {
    let world = World::start().await;
    let token = world
        .browser(
            "web-a",
            DeviceAccess {
                parts: vec!["other/**".into()],
                ..DeviceAccess::new(Some(PriorityClass::Model))
            },
        )
        .await;
    // Above its cap: the host refuses, and the socket closes with 4403.
    let mut over = world.socket_with("control", &token).await;
    over.send(tg::Message::text(
        r#"{"type": "hello", "priority": "operator"}"#,
    ))
    .await
    .unwrap();
    assert_eq!(closed_with(&mut over).await, 4403);

    let (mut control, me) = world.control_with(&token, "model").await;
    let state = state_until(&mut control, |s| s.participant(&me).is_some()).await;
    assert_eq!(state.participant(&me).unwrap().device, "web-a");
    // No arm verb, and bot/arm outside its parts.
    assert_eq!(
        verb(&mut control, &me, 1, Command::Arm).await,
        Some(CtlStatus::NotAuthorized)
    );
    let arm = state
        .part_at(&ParticipantId::new("bot").unwrap(), "arm")
        .unwrap()
        .channel;
    let grant = Command::GrantLease {
        channels: vec![arm],
        release: Vec::new(),
        binding: state
            .binding(&me)
            .map(|b| b.fingerprint)
            .unwrap_or_default(),
        ttl_ns: None,
    };
    assert_eq!(
        verb(&mut control, &me, 2, grant).await,
        Some(CtlStatus::NotAuthorized)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn revoking_a_browser_token_detaches_exactly_its_browser() {
    let world = World::start().await;
    let access = DeviceAccess::new(Some(PriorityClass::Model));
    let a = world.browser("web-a", access.clone()).await;
    let b = world.browser("web-b", access).await;
    let (mut control_a, _) = world.control_with(&a, "model").await;
    let mut state_a = world.socket_with("state", &a).await;
    let (mut control_b, me_b) = world.control_with(&b, "model").await;
    let state = state_until(&mut control_b, |s| s.participant(&me_b).is_some()).await;

    world.host.revoke("web-a").await.unwrap();
    assert_eq!(closed_with(&mut control_a).await, 4409);
    assert_eq!(closed_with(&mut state_a).await, 4409);
    // Its reconnect is refused: the host no longer knows its key.
    let mut again = world.socket_with("control", &a).await;
    again
        .send(tg::Message::text(
            r#"{"type": "hello", "priority": "model"}"#,
        ))
        .await
        .unwrap();
    assert_eq!(closed_with(&mut again).await, 4401);
    // The other browser is untouched: its heartbeats are echoed.
    control_b
        .send(tg::Message::binary(heartbeat(&me_b, 1, state.epoch, 42)))
        .await
        .unwrap();
    let echo = frame_with(&mut control_b, frame::ECHO).await;
    assert_eq!(codec::decode_echo(&echo[1..]).unwrap().t1_ns, 42);
}

/// A `tracing` writer into a shared buffer.
#[derive(Clone, Default)]
struct Captured(Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn tokens_never_reach_the_logs() {
    let logs = Captured::default();
    tracing_subscriber::fmt()
        // Everything but zenoh and this test's own client, which logs
        // the requests it sends.
        .with_env_filter("trace,zenoh=warn,tungstenite::handshake::client=off")
        .with_ansi(false)
        .with_writer({
            let logs = logs.clone();
            move || logs.clone()
        })
        .try_init()
        .expect("this test's process logs nowhere else");
    let world = World::start().await;
    let browser = world
        .browser("web-a", DeviceAccess::new(Some(PriorityClass::Model)))
        .await;
    for token in [PAGE, browser.as_str()] {
        let (mut control, me) = world.control_with(token, "model").await;
        state_until(&mut control, |s| s.participant(&me).is_some()).await;
        let mut state = world.socket_with("state", token).await;
        state
            .send(tg::Message::text(
                r#"{"type": "subscribe", "id": 1, "path": "bot/arm"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(text(&mut state).await["type"], "subscribed");
    }
    let wrong = "wrong-token-0123456789abcdef";
    assert_eq!(
        closed_with(&mut world.socket_with("control", wrong).await).await,
        4401
    );
    let logs = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    assert!(logs.contains("browser attached"), "the capture works");
    for token in [PAGE, browser.as_str(), wrong] {
        assert!(!logs.contains(token), "{token} logged");
    }
}
