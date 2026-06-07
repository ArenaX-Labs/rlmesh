use super::{
    BindTarget, normalize_endpoint, normalize_tcp_session_address, parse_bind_target,
    parse_env_connect_target,
};

#[test]
fn endpoint_normalization_accepts_tcp_runtime_forms() {
    assert_eq!(
        normalize_endpoint("localhost:50051").unwrap(),
        "http://localhost:50051"
    );
    assert_eq!(
        normalize_endpoint("tcp://localhost:50051").unwrap(),
        "http://localhost:50051"
    );
    assert_eq!(
        normalize_endpoint("http://localhost:50051").unwrap(),
        "http://localhost:50051"
    );
}

#[test]
fn endpoint_normalization_rejects_unix_and_https() {
    assert!(normalize_endpoint("unix:///tmp/rlmesh.sock").is_err());
    assert!(normalize_endpoint("https://control-plane.example").is_err());
}

#[test]
fn tcp_session_normalization_uses_tcp_scheme() {
    assert_eq!(
        normalize_tcp_session_address("localhost:50052").unwrap(),
        "tcp://localhost:50052"
    );
    assert_eq!(
        normalize_tcp_session_address("http://localhost:50052").unwrap(),
        "tcp://localhost:50052"
    );
}

#[test]
fn bind_target_accepts_tcp_shortcuts() {
    assert_eq!(
        parse_bind_target("7000").unwrap(),
        BindTarget::Tcp {
            host: "127.0.0.1".to_string(),
            port: 7000,
        }
    );
    assert_eq!(
        parse_bind_target("localhost:7001").unwrap(),
        BindTarget::Tcp {
            host: "127.0.0.1".to_string(),
            port: 7001,
        }
    );
    assert_eq!(
        parse_bind_target("tcp://0.0.0.0:7002").unwrap(),
        BindTarget::Tcp {
            host: "0.0.0.0".to_string(),
            port: 7002,
        }
    );
}

#[test]
fn bind_target_rejects_invalid_addresses() {
    assert!(parse_bind_target("").is_err());
    assert!(parse_bind_target("tcp://").is_err());
    assert!(parse_bind_target("http://localhost:50051").is_err());
    assert!(parse_bind_target("localhost:not-a-port").is_err());
}

#[test]
fn env_connect_target_accepts_tcp_and_http_forms() {
    let bare = parse_env_connect_target("localhost:50053").unwrap();
    assert_eq!(bare.endpoint(), "http://localhost:50053");
    assert_eq!(bare.display_address(), "tcp://localhost:50053");

    let tcp = parse_env_connect_target("tcp://localhost:50054").unwrap();
    assert_eq!(tcp.endpoint(), "http://localhost:50054");
    assert_eq!(tcp.display_address(), "tcp://localhost:50054");

    let http = parse_env_connect_target("http://localhost:50055").unwrap();
    assert_eq!(http.endpoint(), "http://localhost:50055");
    assert_eq!(http.display_address(), "http://localhost:50055");
}

#[test]
fn env_connect_target_rejects_invalid_session_urls() {
    assert!(parse_env_connect_target("https://control-plane.example").is_err());
    assert!(parse_env_connect_target("ftp://localhost:50051").is_err());
    assert!(parse_env_connect_target("localhost").is_err());
}

#[cfg(unix)]
#[test]
fn env_connect_target_accepts_unix_socket_paths() {
    let target = parse_env_connect_target("unix:///tmp/rlmesh.sock").unwrap();
    assert_eq!(target.endpoint(), "http://[::]:50051");
    assert_eq!(target.display_address(), "unix:///tmp/rlmesh.sock");
    assert_eq!(
        target.unix_path().unwrap(),
        &std::path::PathBuf::from("/tmp/rlmesh.sock")
    );
}

#[cfg(unix)]
#[test]
fn bind_target_accepts_unix_socket_paths() {
    assert_eq!(
        parse_bind_target("unix:///tmp/rlmesh-bind.sock").unwrap(),
        BindTarget::Unix {
            path: std::path::PathBuf::from("/tmp/rlmesh-bind.sock"),
        }
    );
}
