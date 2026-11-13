//! Tests of the address grammar, the hosting rule and placement.

use super::*;
use crate::host::HostConfig;
use crate::host::SessionHost;
use crate::transport::{TransportConfig, TransportTuning};
use crate::wire::CtlStatus;

fn host(name: &str, port: Option<u16>) -> Option<HostName> {
    Some(HostName {
        name: name.into(),
        port,
    })
}

fn address(host: Option<HostName>, session: Option<&str>, name: Option<&str>) -> Address {
    Address {
        insecure: false,
        host,
        session: session.map(str::to_string),
        participant: name.map(str::to_string),
        key: None,
    }
}

#[test]
fn participant_addresses_parse_as_the_sdk_doc_writes_them() {
    let table = [
        ("cartpole", address(None, None, Some("cartpole"))),
        (
            "robot-lab/ur10",
            address(None, Some("robot-lab"), Some("ur10")),
        ),
        (
            "lab:7447/cartpole",
            address(host("lab", Some(7447)), None, Some("cartpole")),
        ),
        (
            "lab:7447/bench-3/cartpole",
            address(host("lab", Some(7447)), Some("bench-3"), Some("cartpole")),
        ),
        (
            "0.0.0.0:7447/cartpole",
            address(host("0.0.0.0", Some(7447)), None, Some("cartpole")),
        ),
        (
            ":7447/cartpole",
            address(host("", Some(7447)), None, Some("cartpole")),
        ),
        (
            "robot-lab.local/ur10",
            address(host("robot-lab.local", None), None, Some("ur10")),
        ),
        (
            "10.0.0.5/bench-3/ur10",
            address(host("10.0.0.5", None), Some("bench-3"), Some("ur10")),
        ),
        (
            "[fe80::1]:7447/ur10",
            address(host("fe80::1", Some(7447)), None, Some("ur10")),
        ),
        (
            "localhost/ur10",
            address(host("localhost", None), None, Some("ur10")),
        ),
        (
            "rlmesh://lab:7447/bench-3/cartpole",
            address(host("lab", Some(7447)), Some("bench-3"), Some("cartpole")),
        ),
        (":0/gym", address(host("", Some(0)), None, Some("gym"))),
    ];
    for (text, want) in table {
        let got = Address::participant(text).unwrap_or_else(|e| panic!("{text}: {e}"));
        assert_eq!(got, want, "{text}");
        assert_eq!(
            got.to_string(),
            text.trim_start_matches(SCHEME),
            "{text} prints back"
        );
    }
}

#[test]
fn session_addresses_parse_and_name_their_session() {
    let table = [
        ("", address(None, None, None)),
        ("robot-lab", address(None, Some("robot-lab"), None)),
        ("lab:7447", address(host("lab", Some(7447)), None, None)),
        (
            "lab:7447/bench-3",
            address(host("lab", Some(7447)), Some("bench-3"), None),
        ),
        ("rlmesh://:0", address(host("", Some(0)), None, None)),
    ];
    for (text, want) in table {
        assert_eq!(Address::session(text).expect(text), want, "{text}");
    }
    let named = Address::session("lab:7447/bench-3").expect("parse");
    assert_eq!(named.session_id(), SessionId::named("bench-3"));
    assert_eq!(
        Address::session("").expect("parse").session_id(),
        SessionId::named(DEFAULT_SESSION)
    );
    let id = SessionId::new();
    let by_id = Address::session(&format!("lab:7447/{id}")).expect("uuid");
    assert_eq!(by_id.session_id(), id);
}

#[test]
fn the_hosting_rule_places_each_form() {
    let place = |text: &str| Address::participant(text).expect(text).placement();
    let local = |port| Placement::Host(SocketAddr::from((Ipv4Addr::LOCALHOST, port)));
    assert_eq!(place("cartpole"), Placement::Local);
    assert_eq!(place("robot-lab/ur10"), Placement::Local);
    assert_eq!(place("localhost/ur10"), Placement::Local);
    assert_eq!(place(":7447/x"), local(7447));
    assert_eq!(place(":0/x"), local(0));
    assert_eq!(place("127.0.0.1:9000/x"), local(9000));
    assert_eq!(
        place("0.0.0.0:7447/x"),
        Placement::Host("0.0.0.0:7447".parse().expect("addr"))
    );
    assert_eq!(place("lab:7447/x"), Placement::Join("lab".into(), 7447));
    assert_eq!(
        place("robot-lab.local/x"),
        Placement::Join("robot-lab.local".into(), DEFAULT_PORT)
    );
    assert_eq!(place("10.0.0.5:9/x"), Placement::Join("10.0.0.5".into(), 9));
}

#[test]
fn old_forms_raise_an_upgrade_error_with_the_new_form() {
    let id = SessionId::new();
    for (old, new) in [
        (
            format!("lockstep://127.0.0.1:7447/local/{id}/gym"),
            format!("127.0.0.1:7447/{id}/gym"),
        ),
        (
            "lockstep://127.0.0.1:0/gym".to_string(),
            "127.0.0.1:0/gym".to_string(),
        ),
        (
            format!("127.0.0.1:7447/local/{id}/gym"),
            format!("127.0.0.1:7447/{id}/gym"),
        ),
    ] {
        let err = Address::participant(&old).expect_err(&old).to_string();
        assert!(err.contains(&format!("{new:?}")), "{old}: {err}");
    }
    let err = Address::session(&format!("127.0.0.1:7447/local/{id}"))
        .expect_err("old session")
        .to_string();
    assert!(err.contains(&format!("\"127.0.0.1:7447/{id}\"")), "{err}");
    for bad in [
        "",
        "lab:7447",
        "a//b",
        "fe80::1/x",
        "a/b/c",
        "bad name",
        "_lobby/x",
    ] {
        assert!(Address::participant(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn v2_addresses_stand_apart_from_v1_endpoints() {
    for v2 in [
        "cartpole",
        "robot-lab/ur10",
        ":0/gym",
        "rlmesh://x",
        "lockstep://a:1/b",
    ] {
        assert!(is_v2_address(v2), "{v2}");
    }
    for v1 in [
        "127.0.0.1:50051",
        "50051",
        "tcp://h:1",
        "unix:///s",
        "https://h",
        "lab:7447",
    ] {
        assert!(!is_v2_address(v1), "{v1}");
    }
}

#[test]
fn a_session_address_prints_the_default_session_short() {
    let base: SocketAddr = "127.0.0.1:7447".parse().expect("addr");
    let default = SessionAddress::new(base, DEFAULT_SESSION).expect("default");
    assert_eq!(default.to_string(), "127.0.0.1:7447");
    assert_eq!(
        default.participant_invite("cartpole"),
        "127.0.0.1:7447/cartpole"
    );
    let named = SessionAddress::new(base, "bench-3").expect("named");
    assert_eq!(named.to_string(), "127.0.0.1:7447/bench-3");
    let back: SessionAddress = named.to_string().parse().expect("parse");
    assert_eq!(back, named);
}

#[test]
fn a_keyed_address_carries_its_key_through_a_participant_name() {
    let config = HostConfig::new(TenantId::new(TENANT).expect("tenant"), SessionId::new());
    let key = AddressKey::of(&config.identity);
    let base: SocketAddr = "127.0.0.1:7447".parse().expect("addr");
    for name in [DEFAULT_SESSION, "bench-3"] {
        let session = SessionAddress {
            key: Some(key.clone()),
            ..SessionAddress::new(base, name).expect("session")
        };
        let text = session.invite();
        assert!(text.ends_with(&format!("#{key}")), "{text}");
        assert!(is_v2_address(&text), "{text}");
        assert_eq!(text.parse::<SessionAddress>().expect("parse"), session);
        // Scripts append the participant: the key stays with its session.
        let invite = session.participant_invite("arm1");
        let participant = Address::participant(&invite).expect("parse");
        assert_eq!(participant.participant.as_deref(), Some("arm1"));
        assert_eq!(participant.joining().expect("joining"), session);
        assert_eq!(participant.invite(), invite);
        // Display and Debug, for logs, titles and errors, leave it out.
        let secret = key.to_string();
        let secret = secret.split('.').next().expect("secret");
        for shown in [
            session.to_string(),
            format!("{session:?}"),
            participant.to_string(),
            format!("{participant:?}"),
        ] {
            assert!(!shown.contains(secret) && shown.contains("#…"), "{shown}");
        }
    }
    // A launcher's keys: the host it starts hosts under them.
    let (text, launched) = launch_identity().expect("keys");
    let mut parts = text.split('.').map(|p| unbase32(p).expect("base32"));
    let (mut a, mut b, mut c) = (
        parts.next().expect("static"),
        parts.next().expect("sign"),
        parts.next().expect("local"),
    );
    assert_eq!(
        AddressKey::of(&HostIdentity::from_secrets(&mut a, &mut b, &mut c)),
        launched
    );
    let err = Address::session("127.0.0.1:7447#nope").expect_err("bad key");
    assert!(err.to_string().contains("#"), "{err}");
    // Errors never show a key, well-formed or not.
    let secret = key.to_string();
    let secret = secret.split('.').next().expect("secret");
    let malformed = format!("127.0.0.1:7447/bench 3#{key}");
    let err = Address::session(&malformed).expect_err("a bad name");
    assert!(!err.to_string().contains(secret), "{err}");
    // Nor in the address an upgrade error suggests.
    let old = format!("lockstep://127.0.0.1:7447/local/bench#{key}/env");
    let err = Address::participant(&old).expect_err("an old address");
    assert!(err.to_string().contains("127.0.0.1:7447/bench#"), "{err}");
    assert!(!err.to_string().contains(secret), "{err}");
}

/// A port that accepts and never answers (a forwarder whose host never comes
/// up) is a timeout a retry may get past, not a refusal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_port_that_accepts_and_never_answers_times_out() {
    let base = crate::transport::free_local_base(Ipv4Addr::LOCALHOST.into()).expect("port");
    let listener = tokio::net::TcpListener::bind(base).await.expect("base");
    // Holds each connection open and says nothing.
    let accepting = tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((stream, _)) = listener.accept().await {
            held.push(stream);
        }
    });
    let timeout = Duration::from_secs(3);
    let Err(err) = hosting::connect(base, timeout).await else {
        panic!("linked to a host that never answers");
    };
    accepting.abort();
    assert!(
        matches!(err, SessionError::Timeout(waited, _) if waited == timeout),
        "not a timeout a retry may get past: {err}"
    );
}

/// A port that accepts and closes at once (a container runtime's proxy in
/// front of a host still starting) is a host not up yet, never a refused key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_forwarder_that_drops_every_link_is_no_refusal() {
    let base = crate::transport::free_local_base(Ipv4Addr::LOCALHOST.into()).expect("port");
    let listener = tokio::net::TcpListener::bind(base).await.expect("port");
    let forwarding = tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            drop(stream);
        }
    });
    let timeout = Duration::from_secs(3);
    let Err(err) = hosting::connect(base, timeout).await else {
        panic!("linked to a forwarder");
    };
    forwarding.abort();
    assert!(
        matches!(err, SessionError::Timeout(waited, _) if waited == timeout),
        "not a timeout a retry may get past: {err}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_embedded_host_stays_on_loopback_and_links_only_with_its_key() {
    let grace = Duration::from_millis(250);
    let wide: SocketAddr = "0.0.0.0:0".parse().expect("addr");
    let err = host_at(wide, "lan", grace, false)
        .await
        .expect_err("off loopback");
    assert!(err.to_string().contains("rlmesh host"), "{err}");

    let mut placed = host_at("127.0.0.1:0".parse().expect("addr"), "gated", grace, false)
        .await
        .expect("host");
    let client = |base, link| {
        let tuning = TransportTuning {
            client: true,
            connect_timeout: Some(Duration::from_millis(500)),
            link,
            ..TransportTuning::default()
        };
        TransportConfig::per_class_tcp(base).with_tuning(tuning)
    };
    // Another local user's process, without the key: no link at all.
    let wrong = Some(LinkAuth {
        password: zeroize::Zeroizing::new("guess".into()),
        dictionary: None,
    });
    for link in [None, wrong.clone()] {
        assert!(
            Transport::open(&client(placed.address.base, link))
                .await
                .is_err()
        );
    }
    // Joined through the hosting rule, the wrong key is a refusal, final and
    // soon, not a timeout like a host that does not listen yet.
    let started = std::time::Instant::now();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let refused = open_client(placed.address.base, wrong.clone(), deadline)
        .await
        .expect_err("refused");
    assert_eq!(
        refused.refusal().map(|e| e.status),
        Some(CtlStatus::Unauthenticated),
        "{refused}"
    );
    assert!(started.elapsed() < Duration::from_secs(10), "{refused}");
    let linked = Transport::open(&client(placed.address.base, placed.address.link_auth()))
        .await
        .expect("the key's password links");
    linked.close().await.expect("close");
    placed.close().await.expect("close");

    // Bound beyond loopback, it gates no link (a board's zenoh-pico has
    // no usrpwd), and a join that offers a password links without one;
    // the handshake still refuses a device it never enrolled.
    let mut lan = host_at(wide, "lan", grace, true).await.expect("insecure");
    let base = lan.address.base;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let offered = open_client(base, wrong, deadline).await.expect("links");
    offered.close().await.expect("close");
    let transport = Transport::open(&client(base, None))
        .await
        .expect("links without a password");
    let host_static = *lan.address.credential().expect("credential").host_static();
    let mut config = ParticipantConfig::new(
        lan.address.tenant.clone(),
        lan.address.session,
        ParticipantId::new("intruder").expect("id"),
        vec![Role::Observer],
        PriorityClass::Scripted,
    );
    config.credential = Some(Credential::local(&[7; 32], host_static));
    let err = Participant::attach(transport, config)
        .await
        .err()
        .expect("refused");
    assert_eq!(
        err.refusal().map(|e| e.status),
        Some(CtlStatus::Unauthenticated),
        "{err}"
    );
    lan.close().await.expect("close");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_explicit_local_port_hosts_and_a_second_placement_joins_it() {
    let grace = Duration::from_millis(250);
    let mut first = place(&Address::session(":0").expect("parse"), grace)
        .await
        .expect("host");
    assert!(first.hosts());
    let port = first.address.base.port();
    let second = place(
        &Address::session(&format!(":{port}")).expect("parse"),
        grace,
    )
    .await
    .expect("join");
    assert!(!second.hosts(), "the port is taken: join it");
    assert_eq!(second.address, first.address);
    first.close().await.expect("close");
}

#[test]
fn a_host_name_that_does_not_resolve_is_a_lookup_failure_not_a_bad_address() {
    // A container's peer may not resolve until it starts: that is I/O a
    // retry may get past, where a malformed address is configuration.
    let (_, key) = launch_identity().expect("identity");
    let err = Address::session(&format!("nowhere.invalid:7447#{key}"))
        .expect("parse")
        .joining()
        .expect_err("no such host");
    assert!(matches!(err, SessionError::Io(_)), "{err:?}");
    assert!(err.to_string().contains("nowhere.invalid"), "{err}");
    assert!(!err.to_string().contains(&key.to_string()), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_participant_of_a_host_in_this_process_rides_its_sessions() {
    // No loopback link between a process and itself: no lease a starved
    // process could let lapse, losing the host it runs.
    let mut host = place(
        &Address::session(":0").expect("parse"),
        Duration::from_secs(1),
    )
    .await
    .expect("host");
    let participant = host
        .address
        .join(
            "p",
            vec![Role::Observer],
            PriorityClass::Scripted,
            Duration::from_secs(5),
        )
        .await
        .expect("join");
    let hosting = &host.hosted.as_ref().expect("hosted here").1;
    assert_eq!(
        participant.transport().control().zid(),
        hosting.control().zid()
    );
    participant.close().await;
    host.close().await.expect("close");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_host_serves_many_named_sessions_stops_empty_ones_and_refuses_fast_where_it_cannot() {
    let grace = Duration::from_millis(250);
    let mut host = place(&Address::session(":0").expect("parse"), grace)
        .await
        .expect("host");
    // A host this process placed needs no key in the address.
    let join = |base: SocketAddr, name: &str, key: Option<AddressKey>| {
        let session = SessionAddress {
            key,
            ..SessionAddress::new(base, name).expect("session")
        };
        async move {
            session
                .join(
                    "p",
                    vec![Role::Observer],
                    PriorityClass::Scripted,
                    Duration::from_secs(5),
                )
                .await
        }
    };
    let base = host.address.base;
    let bench = join(base, "bench-3", None)
        .await
        .expect("a new session starts for its first participant");
    let lab = join(base, "robot-lab", None)
        .await
        .expect("and another beside it");
    let server = &host.hosted.as_ref().expect("hosted here").0;
    let running = server.sessions().await;
    assert!(running.contains(&session_id("bench-3")) && running.contains(&session_id("robot-lab")));
    // A session stops when its last participant leaves.
    bench.close().await;
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while server.sessions().await.contains(&session_id("bench-3")) {
        assert!(
            std::time::Instant::now() < deadline,
            "bench-3 never stopped"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(server.sessions().await.contains(&session_id("robot-lab")));
    drop(lab);
    host.close().await.expect("close");

    // A plain single-session host cannot start sessions: refused at once.
    let tenant = TenantId::new(TENANT).expect("tenant");
    let config = HostConfig::new(tenant, session_id("only"));
    let local = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
    let (transport, base) = open_listener(local, Some(&config.identity.link_password()))
        .await
        .expect("listen");
    let key = AddressKey::of(&config.identity);
    let single = SessionHost::start(transport.clone(), config)
        .await
        .expect("single host");
    let started = std::time::Instant::now();
    let Err(err) = join(base, "elsewhere", Some(key)).await else {
        panic!("not served");
    };
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "fails fast: {err}"
    );
    let text = err.to_string();
    assert!(
        text.contains(&base.to_string()) && text.contains("elsewhere"),
        "{text}"
    );
    single.shutdown().await;
    transport.close().await.expect("close");
}

#[test]
fn a_launch_identity_gives_back_its_key_and_keyed_address() {
    let (identity, key) = launch_identity().unwrap();
    assert_eq!(identity_key(&identity).unwrap(), key);
    assert_eq!(
        identity_key(&format!(" {}\n", identity.as_str())).unwrap(),
        key
    );

    let mut address = launched_address(&identity, "10.0.0.5:7447/bench").unwrap();
    address.participant = Some("env".into());
    let invite = address.invite();
    assert_eq!(invite, format!("10.0.0.5:7447/bench#{key}/env"));
    let parsed = Address::participant(&invite).unwrap();
    assert_eq!(parsed.key, Some(key.clone()));
    assert_eq!(parsed.session_name(), "bench");
    assert_eq!(
        launched_address(&identity, "10.0.0.5:7447")
            .unwrap()
            .invite(),
        format!("10.0.0.5:7447#{key}")
    );

    let error = identity_key("not.an.identity").unwrap_err().to_string();
    assert!(!error.contains("not.an.identity"), "{error}");
    let (_, other) = launch_identity().unwrap();
    assert_ne!(other, key);
    assert!(launched_address(&identity, "bench").is_err(), "no host");
    assert!(launched_address(&identity, &format!("10.0.0.5:7447#{other}")).is_err());
}

/// The scratch directory of [`a_launched_host_is_joined_from_another_machine`]'s
/// processes, set in them only: the identities and keys (0600 files, never
/// argv) and the ports the host listens on.
const ACROSS: &str = "RLMESH_TEST_ACROSS_DIR";

/// The host and the client of [`a_launched_host_is_joined_from_another_machine`],
/// each in a process of its own, so nothing in-process supplies the key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn across_role() {
    let Some(dir) = std::env::var_os(ACROSS).map(std::path::PathBuf::from) else {
        return;
    };
    let ip: std::net::IpAddr = std::env::var("RLMESH_TEST_HOST_IP")
        .expect("the host's address")
        .parse()
        .expect("an IP");
    if std::env::var("RLMESH_TEST_ROLE").as_deref() == Ok("host") {
        across_host(&dir, ip).await;
    } else {
        across_client(&dir, ip).await;
    }
}

async fn across_host(dir: &std::path::Path, ip: std::net::IpAddr) {
    let grace = Duration::from_secs(1);
    // Launched (its identity in HOST_IDENTITY_VAR): every link gated.
    let gated = host_at(SocketAddr::new(ip, 0), "across", grace, false)
        .await
        .expect("a launched host beyond loopback");
    // A host that gates no link, as `rlmesh host` or an insecure one.
    let identity = std::fs::read_to_string(dir.join("lan-identity")).expect("identity");
    let mut config = HostConfig::new(TenantId::new(TENANT).expect("tenant"), session_id("across"));
    config.identity = std::sync::Arc::new(host_identity(&identity).expect("identity"));
    config.clock = SystemMonotonicClock.shared();
    let (transport, lan) = open_listener(SocketAddr::new(ip, 0), None)
        .await
        .expect("listen");
    let _lan = SessionHost::start(transport, config).await.expect("host");
    let ports = format!("{} {}", gated.address.base.port(), lan.port());
    std::fs::write(dir.join("ports.partial"), ports).expect("write");
    std::fs::rename(dir.join("ports.partial"), dir.join("ports")).expect("rename");
    // Until the script stops this process.
    std::future::pending::<()>().await;
}

async fn across_client(dir: &std::path::Path, ip: std::net::IpAddr) {
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    let ports = loop {
        if let Ok(text) = std::fs::read_to_string(dir.join("ports")) {
            break text;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the host never listened"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    let (gated, lan) = ports.split_once(' ').expect("two ports");
    let key = |name: &str| std::fs::read_to_string(dir.join(name)).expect("key");
    let (_, wrong) = launch_identity().expect("identity");
    let wrong = wrong.to_string();
    // A slowed link (RLMESH_TEST_NETEM): every Zenoh open takes seconds.
    let slow = std::env::var_os("RLMESH_TEST_NETEM").is_some();
    let timeout = Duration::from_secs(if slow { 40 } else { 10 });
    let join = |port: &str, key: Option<&str>| {
        let text = match key {
            Some(key) => format!("{ip}:{port}/across#{key}"),
            None => format!("{ip}:{port}/across"),
        };
        async move {
            let address: SessionAddress = text.parse().expect("an address");
            let started = std::time::Instant::now();
            let joined = address
                .join(
                    "driver",
                    vec![Role::Observer],
                    PriorityClass::Scripted,
                    timeout,
                )
                .await;
            (joined, started.elapsed())
        }
    };

    if slow {
        // A host that gates nothing, slow to open a link: not a refusal.
        let (joined, took) = join(lan, Some(&key("lan-key"))).await;
        let joined = joined.expect("a slow host that gates nothing admits a keyed address");
        eprintln!("ACROSS slowed link, ungated host, its key: joined in {took:?}");
        joined.close().await;
        let (joined, took) = join(gated, Some(&key("key"))).await;
        joined
            .expect("a slow launched host admits its keyed address")
            .close()
            .await;
        eprintln!("ACROSS slowed link, launched host, its key: joined in {took:?}");
        let (refused, took) = join(gated, Some(&wrong)).await;
        let err = refused.err().expect("a wrong key is refused");
        eprintln!("ACROSS slowed link, launched host, a wrong key: refused in {took:?}");
        assert_eq!(
            err.refusal().map(|e| e.status),
            Some(CtlStatus::Unauthenticated),
            "{err}"
        );
        return;
    }

    let (joined, took) = join(gated, Some(&key("key"))).await;
    let joined = joined.expect("a launched host admits its keyed address from another machine");
    eprintln!("ACROSS launched host, its key: joined in {took:?}");
    joined.close().await;

    let (refused, took) = join(gated, Some(&wrong)).await;
    let err = refused.err().expect("a wrong key is refused");
    let text = err.to_string();
    eprintln!("ACROSS launched host, a wrong key: refused in {took:?}");
    assert!(!text.contains(&wrong[..20]), "the error repeats the key");
    assert_eq!(
        err.refusal().map(|e| e.status),
        Some(CtlStatus::Unauthenticated),
        "{text}"
    );
    assert!(text.contains("key"), "{text}");
    assert!(
        took < Duration::from_secs(3),
        "refused at once, not timed out: {text}"
    );

    let (refused, took) = join(gated, None).await;
    let err = refused.err().expect("no key is refused");
    eprintln!("ACROSS launched host, no key: refused in {took:?}: {err}");
    assert!(err.to_string().contains("key"), "{err}");

    let (joined, took) = join(lan, Some(&key("lan-key"))).await;
    let joined = joined.expect("a host that gates nothing admits a keyed address");
    eprintln!("ACROSS ungated host, its key: joined in {took:?}");
    joined.close().await;
}

#[test]
fn a_launched_host_is_joined_from_another_machine() {
    across(None);
}

#[test]
fn a_slow_link_to_a_host_is_no_refusal() {
    // A 0.3 s round trip: a TCP connect outlasts a short probe, and a
    // link's four Zenoh opens take seconds, refused or not.
    across(Some("delay 150ms"));
}

/// Runs [`across_role`]'s host and client on two networks, the link between
/// them slowed by `netem` (tc netem arguments).
fn across(netem: Option<&str>) {
    let dir = tempfile::tempdir().expect("scratch");
    let (identity, key) = launch_identity().expect("identity");
    let (lan_identity, lan_key) = launch_identity().expect("identity");
    for (name, secret) in [
        ("identity", identity.to_string()),
        ("key", key.to_string()),
        ("lan-identity", lan_identity.to_string()),
        ("lan-key", lan_key.to_string()),
    ] {
        write_private(&dir.path().join(name), secret.as_bytes()).expect("write");
    }
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/test-across-netns.sh");
    // The host alone gets the identity, in its environment.
    let dispatch = format!(
        "if [ \"$RLMESH_TEST_ROLE\" = host ]; then \
         {HOST_IDENTITY_VAR}=$(cat \"${ACROSS}/identity\"); export {HOST_IDENTITY_VAR}; fi; \
         exec \"$1\" --exact address::tests::across_role --nocapture"
    );
    let mut command = std::process::Command::new("sh");
    if let Some(netem) = netem {
        command.env("RLMESH_TEST_NETEM", netem);
    }
    let output = command
        .arg(script)
        .args(["sh", "-c", &dispatch, "sh"])
        .arg(std::env::current_exe().expect("test binary"))
        .env(ACROSS, dir.path())
        .env(STATE_DIR_VAR, dir.path().join("state"))
        .env(
            crate::address::hosting::RUNTIME_DIR_VAR,
            dir.path().join("run"),
        )
        .env_remove(HOST_IDENTITY_VAR)
        .output()
        .expect("run the script");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    for secret in [identity.to_string(), key.to_string(), lan_key.to_string()] {
        assert!(!text.contains(&secret[..20]), "a secret reached the output");
    }
    eprintln!("{text}");
    if output.status.code() == Some(77) {
        eprintln!("skipped: no network namespaces (or, unslowed, no non-loopback address)");
        return;
    }
    assert!(output.status.success(), "{text}");
}

#[test]
fn no_parse_error_repeats_the_key() {
    let (identity, key) = launch_identity().expect("keys");
    let key = key.to_string();
    let participant = |text: String| Address::participant(&text).map(drop);
    let session = |text: String| Address::session(&text).map(drop);
    let errors = [
        participant(format!("lockstep://10.0.0.5:7447/t/s/r#{key}")),
        participant(format!("lockstep://10.0.0.5:7447#{key}/r")),
        participant(format!("h:1#{key}x/p")),
        participant(format!("h:1#{key}//p")),
        participant(format!("h:x#{key}/p")),
        participant(format!("[fe80::1#{key}/p")),
        participant(format!("[::1]x#{key}/p")),
        participant(format!("fe80::1:7447#{key}/p")),
        participant(format!("h:1#{key}")),
        participant(format!("h:1#{key}/t/s/p")),
        participant(format!("h:1#{key}/a/b/c/d")),
        participant(format!("h:1#{key}/bad!name/p")),
        participant(format!("h:1#{key}/_lobby/p")),
        participant(format!("h:1#{key}/s/bad!name")),
        participant(format!("h:1/s#{key}/bad!name")),
        session(format!("h:1#{key}/t/s")),
        session(format!("h:1#{key}/a/b/c")),
        session(format!("h:1/_s#{key}")),
        launched_address(&identity, &format!("s#{key}")).map(drop),
        launched_address(&identity, &format!("h:1#{key}")).map(drop),
    ];
    for (i, error) in errors.into_iter().enumerate() {
        let error = error.expect_err("refused").to_string();
        assert!(
            !error.contains(&key),
            "error {i} repeats the key: {}",
            error.replace(&key, "<KEY>")
        );
    }
    // What parses prints without it too.
    let parsed = Address::participant(&format!("h:1#{key}/s/p")).expect("parses");
    let joined: SessionAddress = format!("127.0.0.1:1/s#{key}").parse().expect("parses");
    for shown in [
        format!("{parsed} {parsed:?}"),
        format!("{joined} {joined:?}"),
    ] {
        assert!(!shown.contains(&key), "{}", shown.replace(&key, "<KEY>"));
    }
}
