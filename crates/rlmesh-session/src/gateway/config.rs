//! The gateway's command line and the configuration it resolves to.
//!
//! Every socket needs a token ([`security`]). This machine's page token
//! comes from `RLMESH_GATEWAY_TOKEN` or `--token-file`, never from a
//! command-line value (argv is world-readable); paired browser tokens from
//! `--browsers` (`rlmesh pair HOST --browser` writes it). The gateway
//! refuses to start with neither. A non-loopback `--bind` refuses to start
//! without TLS (a token would cross the network in the clear) unless
//! `--insecure-http` says so. `--allow-origin '*'` allows any origin;
//! without it, only the origins this gateway serves its own page from.
//! Browsers attach at `model` at most unless `--max-priority` raises it.
//!
//! The session is the address its host prints (`--address
//! 127.0.0.1:7447/<session>#<key>`), or its parts: `--connect` with
//! `--session` or `--session-file`, either `<session>#<key>` or the bare
//! session of a host on this machine that left its key for this user (or
//! that this machine paired with). The gateway's own participant attaches
//! with that key; it joins the host as a client, as every participant
//! does.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use crate::SessionAddress;
use crate::address::AddressKey;
use crate::pairing;
use crate::wire::{PriorityClass, TenantId};
use clap::Parser;

use crate::gateway::security::{self, Origins, Policy};

/// Bridges browsers to an RLMesh real-time session (protocol v2): one
/// WebSocket per traffic class (`/control`, `/state`, `/bulk`), each browser
/// connection a session participant of its own.
#[derive(Clone, Debug, Parser)]
#[command(name = "rlmesh-gateway", version, about)]
pub struct Args {
    /// The session's address as its host prints it:
    /// `{ip}:{port}[/{session}]#{key}`.
    #[arg(long, conflicts_with_all = ["connect", "session", "session_file"])]
    pub address: Option<String>,
    /// The session host's per-class TCP base (class i on this port + i:
    /// control, realtime, data, bulk).
    #[arg(long, required_unless_present = "address")]
    pub connect: Option<SocketAddr>,
    /// The tenant namespace.
    #[arg(long, default_value = "local")]
    pub tenant: String,
    /// The session: its name or UUID, and `#<key>` (its host's address
    /// key) unless the host left it for this user.
    #[arg(
        long,
        conflicts_with = "session_file",
        required_unless_present_any = ["session_file", "address"]
    )]
    pub session: Option<String>,
    /// Read the session (as `--session` takes it) from this file, as a
    /// robot's `--session-file` writes it.
    #[arg(long)]
    pub session_file: Option<PathBuf>,
    /// The address to serve on. Anything but loopback needs TLS.
    #[arg(long, default_value = "127.0.0.1")]
    pub bind: IpAddr,
    /// The port to serve on.
    #[arg(long, default_value_t = 18080)]
    pub port: u16,
    /// Serve this directory at `/` (with COOP/COEP, so pages are
    /// cross-origin isolated).
    #[arg(long = "static")]
    pub static_dir: Option<PathBuf>,
    /// Read this machine's page token from this file (or set
    /// RLMESH_GATEWAY_TOKEN); its browser attaches as this gateway.
    #[arg(long)]
    pub token_file: Option<PathBuf>,
    /// The browser tokens paired on this machine (`rlmesh pair HOST
    /// --browser`), each a device of its own on the host (default:
    /// `rlmesh/browsers.toml` in the user config directory).
    #[arg(long, value_name = "FILE")]
    pub browsers: Option<PathBuf>,
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
    /// Serve a non-loopback `--bind` over plain HTTP anyway (tokens cross
    /// the network in the clear).
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
    /// The session, with its key when the address carries one.
    pub address: SessionAddress,
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
    /// Resolves `args`, with the page token from the environment
    /// (`env_token`) unless `--token-file` names one.
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
        let browsers = args.browsers.clone().or_else(pairing::browsers_path);
        if token.is_none() && !browsers.as_ref().is_some_and(|b| b.exists()) {
            return Err(format!(
                "every browser needs a token: set RLMESH_GATEWAY_TOKEN (or --token-file) for this machine's page, opened as http://127.0.0.1:{}/#t=<token>, or pair one with `rlmesh pair HOST --browser`",
                args.port
            ));
        }
        let tls = args.tls_cert.clone().zip(args.tls_key.clone());
        if !args.bind.is_loopback() && tls.is_none() && !args.insecure_http {
            return Err(format!(
                "--bind {} without TLS would send tokens in the clear: pass --tls-cert and --tls-key (or --insecure-http)",
                args.bind
            ));
        }
        let origins = if args.allow_origin.iter().any(|o| o == "*") {
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
            Some(text) => SessionAddress {
                // The platform's tenant is configuration, not address.
                tenant: TenantId::new(args.tenant.clone()).map_err(|e| e.to_string())?,
                ..text
                    .parse::<SessionAddress>()
                    .map_err(|e| format!("--address: {e}"))?
            },
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
                let base = args
                    .connect
                    .ok_or("pass --address, or --connect with --session or --session-file")?;
                let (session, key) = match session.split_once('#') {
                    Some((session, key)) => (
                        session.to_string(),
                        Some(key.parse::<AddressKey>().map_err(|e| e.to_string())?),
                    ),
                    None => (session, None),
                };
                SessionAddress {
                    // The platform's tenant is configuration, not address.
                    tenant: TenantId::new(args.tenant).map_err(|e| e.to_string())?,
                    key,
                    ..SessionAddress::new(base, &session).map_err(|e| e.to_string())?
                }
            }
        };
        Ok(Self {
            address,
            bind: args.bind,
            port: args.port,
            static_dir: args.static_dir,
            policy: Policy {
                page: token.as_deref().map(pairing::token_hash),
                browsers,
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
    const TOKEN: &str = "0123456789abcdefXYZ";

    fn args(extra: &str) -> Args {
        let line = format!(
            "rlmesh-gateway --connect 127.0.0.1:17600 --session {SESSION} --browsers /nonexistent {extra}"
        );
        Args::try_parse_from(line.split_whitespace()).expect("args")
    }

    #[test]
    fn defaults_are_loopback_with_a_page_token() {
        let err = Config::from_args(args(""), None).expect_err("no token at all");
        assert!(err.contains("needs a token"), "{err}");
        let c = Config::from_args(args(""), Some(TOKEN.into())).expect("config");
        assert_eq!(c.bind.to_string(), "127.0.0.1");
        assert_eq!(c.policy.page, Some(pairing::token_hash(TOKEN)));
        assert!(c.tls.is_none());
        assert_eq!(c.policy.max_message_bytes, 65536);
        assert_eq!(c.max_priority, PriorityClass::Model, "operator is opt-in");
        assert_eq!(c.address.session.to_string(), SESSION);
    }

    #[test]
    fn a_non_loopback_bind_needs_tls() {
        let err =
            Config::from_args(args("--bind 0.0.0.0"), Some(TOKEN.into())).expect_err("no TLS");
        assert!(err.contains("in the clear"), "{err}");
        let c = Config::from_args(
            args("--bind 0.0.0.0 --insecure-http --allow-origin https://ops.example/"),
            Some(TOKEN.into()),
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
        assert!(matches!(
            Config::from_args(args("--allow-origin *"), Some(TOKEN.into()))
                .expect("config")
                .policy
                .origins,
            Origins::Any
        ));
        assert!(Config::from_args(args(""), Some("short".into())).is_err());
        assert!(Config::from_args(args("--max-priority root"), Some(TOKEN.into())).is_err());
    }

    #[test]
    fn the_printed_session_address_is_accepted() {
        let line = format!(
            "rlmesh-gateway --address 10.0.0.5:7447/{SESSION} --tenant lab --browsers /nonexistent"
        );
        let c = Config::from_args(
            Args::try_parse_from(line.split_whitespace()).expect("args"),
            Some(TOKEN.into()),
        )
        .expect("config");
        assert_eq!(c.address.base.to_string(), "10.0.0.5:7447");
        assert_eq!(c.address.tenant.as_str(), "lab");
        assert_eq!(c.address.session.to_string(), SESSION);
        let both =
            format!("rlmesh-gateway --address 10.0.0.5:7447/{SESSION} --connect 10.0.0.5:7447");
        assert!(Args::try_parse_from(both.split_whitespace()).is_err());
    }
}
