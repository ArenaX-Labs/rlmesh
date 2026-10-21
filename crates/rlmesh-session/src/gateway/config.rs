//! The gateway's command line and the configuration it resolves to.
//!
//! The token comes from `RLMESH_GATEWAY_TOKEN` or `--token-file`, never from
//! a command-line value (argv is world-readable). A non-loopback `--bind`
//! refuses to start without one, and without TLS (the token would cross the
//! network in the clear) unless `--insecure-http` says so. `--allow-origin
//! '*'` allows any origin and needs a token; without it, only the origins
//! this gateway serves its own page from. Browsers attach at `model` at
//! most unless `--max-priority` raises it.
//!
//! The session is the address its host prints (`--address
//! 127.0.0.1:7447/local/<id>`), or its parts (`--connect` with `--session`
//! or `--session-file`).

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use crate::SessionAddress;
use crate::wire::{PriorityClass, SessionId, TenantId};
use clap::Parser;

use crate::gateway::security::{self, Origins, Policy};

/// Bridges browsers to an RLMesh real-time session (protocol v2): one
/// WebSocket per traffic class (`/control`, `/state`, `/bulk`), each browser
/// connection a session participant of its own.
#[derive(Clone, Debug, Parser)]
#[command(name = "rlmesh-gateway", version, about)]
pub struct Args {
    /// The session's address as its host prints it:
    /// `{ip}:{port}/{tenant}/{session}`.
    #[arg(long, conflicts_with_all = ["connect", "session", "session_file", "tenant"])]
    pub address: Option<String>,
    /// The session host's per-class TCP base (class i on this port + i:
    /// control, realtime, data, bulk).
    #[arg(long, required_unless_present = "address")]
    pub connect: Option<SocketAddr>,
    /// Where the gateway's own session endpoints listen (port 0: one picked
    /// per class), so peers can dial it. Defaults to the host's IP when that
    /// is loopback, else every interface.
    #[arg(long)]
    pub zenoh_listen: Option<SocketAddr>,
    /// The tenant namespace.
    #[arg(long, default_value = "local")]
    pub tenant: String,
    /// The session id (a UUIDv7).
    #[arg(
        long,
        conflicts_with = "session_file",
        required_unless_present_any = ["session_file", "address"]
    )]
    pub session: Option<String>,
    /// Read the session id from this file (`rlmesh-session-host
    /// --session-file`).
    #[arg(long)]
    pub session_file: Option<PathBuf>,
    /// The address to serve on. Anything but loopback needs a token.
    #[arg(long, default_value = "127.0.0.1")]
    pub bind: IpAddr,
    /// The port to serve on.
    #[arg(long, default_value_t = 18080)]
    pub port: u16,
    /// Serve this directory at `/` (with COOP/COEP, so pages are
    /// cross-origin isolated).
    #[arg(long = "static")]
    pub static_dir: Option<PathBuf>,
    /// Read the bearer token from this file (or set RLMESH_GATEWAY_TOKEN).
    #[arg(long)]
    pub token_file: Option<PathBuf>,
    /// Allow WebSockets from this page origin too ('*': any).
    #[arg(long = "allow-origin")]
    pub allow_origin: Vec<String>,
    /// The largest inbound message (and frame), in bytes.
    #[arg(long, default_value_t = security::DEFAULT_MAX_MESSAGE_BYTES)]
    pub max_message_bytes: usize,
    /// The most stream subscriptions one data connection may hold.
    #[arg(long, default_value_t = security::DEFAULT_MAX_SUBSCRIPTIONS)]
    pub max_subscriptions: usize,
    /// The highest lease class a browser may attach at (scripted, model,
    /// operator, safety). `operator` lets a page take over from a model.
    #[arg(long, default_value = "model")]
    pub max_priority: String,
    /// Serve a non-loopback `--bind` over plain HTTP anyway (the token
    /// crosses the network in the clear).
    #[arg(long)]
    pub insecure_http: bool,
    /// A PEM certificate chain: serve https/wss.
    #[arg(long, requires = "tls_key")]
    pub tls_cert: Option<PathBuf>,
    /// The PEM private key for `--tls-cert`.
    #[arg(long, requires = "tls_cert")]
    pub tls_key: Option<PathBuf>,
}

/// A resolved configuration.
#[derive(Clone, Debug)]
pub struct Config {
    /// The host's per-class TCP base.
    pub connect: SocketAddr,
    /// The gateway's own session listen address.
    pub zenoh_listen: SocketAddr,
    /// The tenant.
    pub tenant: TenantId,
    /// The session.
    pub session: SessionId,
    /// Where to serve.
    pub bind: IpAddr,
    /// The port.
    pub port: u16,
    /// The static directory.
    pub static_dir: Option<PathBuf>,
    /// Who may connect and what a connection may do.
    pub policy: Policy,
    /// The highest class a browser may attach at.
    pub max_priority: PriorityClass,
    /// The certificate chain and key, for TLS.
    pub tls: Option<(PathBuf, PathBuf)>,
}

/// A lease class by its lowercase name.
pub fn parse_priority(name: &str) -> Result<PriorityClass, String> {
    match name {
        "scripted" => Ok(PriorityClass::Scripted),
        "model" => Ok(PriorityClass::Model),
        "operator" => Ok(PriorityClass::Operator),
        "safety" => Ok(PriorityClass::Safety),
        other => Err(format!(
            "unknown priority {other:?} (scripted | model | operator | safety)"
        )),
    }
}

impl Config {
    /// Resolves `args`, with the token from the environment (`env_token`)
    /// unless `--token-file` names one.
    pub fn from_args(args: Args, env_token: Option<String>) -> Result<Self, String> {
        let token = match (&args.token_file, env_token.filter(|t| !t.is_empty())) {
            (Some(file), _) => Some(
                std::fs::read_to_string(file)
                    .map_err(|e| format!("--token-file {}: {e}", file.display()))?
                    .trim()
                    .to_string(),
            ),
            (None, token) => token.map(|t| t.trim().to_string()),
        };
        if let Some(token) = &token {
            security::validate_token(token)?;
        }
        security::validate_bind(args.bind, token.as_deref())?;
        let tls = args.tls_cert.clone().zip(args.tls_key.clone());
        if !args.bind.is_loopback() && tls.is_none() && !args.insecure_http {
            return Err(format!(
                "--bind {} without TLS would send the token in the clear: pass --tls-cert and --tls-key (or --insecure-http)",
                args.bind
            ));
        }
        let origins = if args.allow_origin.iter().any(|o| o == "*") {
            if token.is_none() {
                return Err(
                    "--allow-origin '*' lets any page drive the session: it needs a token (RLMESH_GATEWAY_TOKEN or --token-file)"
                        .into(),
                );
            }
            Origins::Any
        } else {
            let mut list = security::default_origins(args.bind, args.port, tls.is_some());
            list.extend(
                args.allow_origin
                    .iter()
                    .map(|o| security::normalize_origin(o)),
            );
            Origins::List(list)
        };
        let address = match &args.address {
            Some(text) => text
                .parse::<SessionAddress>()
                .map_err(|e| format!("--address: {e}"))?,
            None => {
                let session = match (&args.session, &args.session_file) {
                    (Some(id), _) => id.clone(),
                    (None, Some(file)) => std::fs::read_to_string(file)
                        .map_err(|e| format!("--session-file {}: {e}", file.display()))?
                        .trim()
                        .to_string(),
                    (None, None) => {
                        return Err(
                            "pass --address, or --connect with --session or --session-file".into(),
                        );
                    }
                };
                SessionAddress {
                    base: args
                        .connect
                        .ok_or("pass --address, or --connect with --session or --session-file")?,
                    tenant: TenantId::new(args.tenant).map_err(|e| e.to_string())?,
                    session: session
                        .parse()
                        .map_err(|e: crate::wire::IdError| e.to_string())?,
                }
            }
        };
        let zenoh_listen = args.zenoh_listen.unwrap_or_else(|| {
            let ip = if address.base.ip().is_loopback() {
                address.base.ip()
            } else {
                std::net::Ipv4Addr::UNSPECIFIED.into()
            };
            SocketAddr::new(ip, 0)
        });
        Ok(Self {
            connect: address.base,
            zenoh_listen,
            tenant: address.tenant,
            session: address.session,
            bind: args.bind,
            port: args.port,
            static_dir: args.static_dir,
            policy: Policy {
                token,
                origins,
                max_message_bytes: args.max_message_bytes,
                max_subscriptions: args.max_subscriptions,
            },
            max_priority: parse_priority(&args.max_priority)?,
            tls,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SESSION: &str = "0192f3a4-5b6c-7d8e-9f00-112233445566";

    fn args(extra: &str) -> Args {
        let line = format!("rlmesh-gateway --connect 127.0.0.1:17600 --session {SESSION} {extra}");
        Args::try_parse_from(line.split_whitespace()).expect("args")
    }

    #[test]
    fn defaults_are_loopback_without_a_token() {
        let c = Config::from_args(args(""), None).expect("config");
        assert_eq!(c.bind.to_string(), "127.0.0.1");
        assert!(c.policy.token.is_none());
        assert!(c.tls.is_none());
        assert_eq!(c.policy.max_message_bytes, 65536);
        assert_eq!(c.max_priority, PriorityClass::Model, "operator is opt-in");
        assert_eq!(c.session.to_string(), SESSION);
    }

    #[test]
    fn a_non_loopback_bind_needs_a_token() {
        let err = Config::from_args(args("--bind 0.0.0.0"), None).expect_err("refused");
        assert!(err.contains("without a token"), "{err}");
        let err = Config::from_args(args("--bind 0.0.0.0"), Some("0123456789abcdefXYZ".into()))
            .expect_err("no TLS");
        assert!(err.contains("in the clear"), "{err}");
        let c = Config::from_args(
            args("--bind 0.0.0.0 --insecure-http --allow-origin https://ops.example/"),
            Some("0123456789abcdefXYZ".into()),
        )
        .expect("config");
        let Origins::List(list) = c.policy.origins else {
            panic!("a list")
        };
        assert!(list.contains(&"https://ops.example".to_string()));
        assert!(
            Args::try_parse_from(
                format!(
                    "rlmesh-gateway --connect 127.0.0.1:1 --session {SESSION} --tls-cert x.pem"
                )
                .split_whitespace()
            )
            .is_err()
        );
        let err = Config::from_args(args("--allow-origin *"), None).expect_err("no token");
        assert!(err.contains("needs a token"), "{err}");
        assert!(matches!(
            Config::from_args(args("--allow-origin *"), Some("0123456789abcdefXYZ".into()))
                .expect("config")
                .policy
                .origins,
            Origins::Any
        ));
        assert!(Config::from_args(args(""), Some("short".into())).is_err());
        assert!(Config::from_args(args("--max-priority root"), None).is_err());
    }

    #[test]
    fn the_printed_session_address_is_accepted() {
        let line = format!("rlmesh-gateway --address 10.0.0.5:7447/lab/{SESSION}");
        let c = Config::from_args(
            Args::try_parse_from(line.split_whitespace()).expect("args"),
            Some("0123456789abcdefXYZ".into()),
        )
        .expect("config");
        assert_eq!(c.connect.to_string(), "10.0.0.5:7447");
        assert_eq!(c.tenant.as_str(), "lab");
        assert_eq!(c.session.to_string(), SESSION);
        assert_eq!(c.zenoh_listen.ip().to_string(), "0.0.0.0");
        let both =
            format!("rlmesh-gateway --address 10.0.0.5:7447/lab/{SESSION} --connect 10.0.0.5:7447");
        assert!(Args::try_parse_from(both.split_whitespace()).is_err());
    }
}
