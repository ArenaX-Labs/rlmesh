#[cfg(not(unix))]
use std::path::PathBuf;
#[cfg(unix)]
use std::path::{Path, PathBuf};

use crate::error::{Error, TransportError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvConnectTarget {
    endpoint: String,
    address: String,
    kind: EnvConnectTargetKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EnvConnectTargetKind {
    Tcp,
    #[cfg(unix)]
    Unix {
        path: PathBuf,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindTarget {
    Tcp { host: String, port: u16 },
    Unix { path: PathBuf },
}

impl EnvConnectTarget {
    pub fn display_address(&self) -> &str {
        &self.address
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn unix_path(&self) -> Option<&PathBuf> {
        match &self.kind {
            EnvConnectTargetKind::Tcp => None,
            #[cfg(unix)]
            EnvConnectTargetKind::Unix { path } => Some(path),
        }
    }
}

pub fn parse_env_connect_target(addr: &str) -> Result<EnvConnectTarget, Error> {
    let addr = addr.trim();
    if addr.is_empty() {
        return Err(TransportError::InvalidAddress("empty address".to_string()).into());
    }

    if let Some(path) = addr.strip_prefix("unix://") {
        #[cfg(unix)]
        {
            let socket_path = normalize_unix_path(path)?;
            return Ok(EnvConnectTarget {
                endpoint: "http://[::]:50051".to_string(),
                address: format!("unix://{}", socket_path.display()),
                kind: EnvConnectTargetKind::Unix { path: socket_path },
            });
        }

        #[cfg(not(unix))]
        {
            let _ = path;
            return Err(TransportError::InvalidAddress(
                "unix sockets are not supported on Windows; use tcp://host:port instead"
                    .to_string(),
            )
            .into());
        }
    }

    if let Some(target) = addr.strip_prefix("tcp://") {
        validate_tcp_authority(target)?;
        return Ok(EnvConnectTarget {
            endpoint: format!("http://{target}"),
            address: format!("tcp://{target}"),
            kind: EnvConnectTargetKind::Tcp,
        });
    }

    if let Some(target) = addr.strip_prefix("http://") {
        validate_tcp_authority(target)?;
        return Ok(EnvConnectTarget {
            endpoint: addr.to_string(),
            address: addr.to_string(),
            kind: EnvConnectTargetKind::Tcp,
        });
    }

    if addr.starts_with("https://") {
        return Err(TransportError::InvalidAddress(
            "https:// is reserved for control-plane URLs, not session links".to_string(),
        )
        .into());
    }

    if addr.contains("://") {
        let scheme = addr
            .split_once("://")
            .map(|(scheme, _)| scheme)
            .unwrap_or(addr);
        return Err(TransportError::InvalidAddress(format!(
            "unsupported address scheme '{scheme}'"
        ))
        .into());
    }

    validate_tcp_authority(addr)?;
    Ok(EnvConnectTarget {
        endpoint: format!("http://{addr}"),
        address: format!("tcp://{addr}"),
        kind: EnvConnectTargetKind::Tcp,
    })
}

pub fn parse_bind_target(addr: &str) -> Result<BindTarget, Error> {
    let addr = addr.trim();
    if addr.is_empty() {
        return Err(TransportError::InvalidAddress("empty address".to_string()).into());
    }

    if let Some(path) = addr.strip_prefix("unix://") {
        return Ok(BindTarget::Unix {
            path: normalize_unix_path(path)?,
        });
    }

    if addr.contains("://") && !addr.starts_with("tcp://") {
        let scheme = addr
            .split_once("://")
            .map(|(scheme, _)| scheme)
            .unwrap_or(addr);
        return Err(TransportError::InvalidAddress(format!(
            "unsupported address scheme '{scheme}'"
        ))
        .into());
    }

    let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
    if tcp_addr.is_empty() {
        return Err(TransportError::InvalidAddress("empty tcp address".to_string()).into());
    }

    if let Some((host, port)) = tcp_addr.rsplit_once(':') {
        return Ok(BindTarget::Tcp {
            host: normalize_bind_host(host).to_string(),
            port: parse_port(port, "tcp port")?,
        });
    }

    Ok(BindTarget::Tcp {
        host: "127.0.0.1".to_string(),
        port: parse_port(tcp_addr, "port")?,
    })
}

pub fn normalize_endpoint(addr: &str) -> Result<String, Error> {
    let addr = addr.trim();
    if addr.is_empty() {
        return Err(TransportError::InvalidAddress("empty address".to_string()).into());
    }

    if addr.starts_with("unix://") {
        return Err(TransportError::InvalidAddress(
            "unix sockets are not supported by rlmesh-grpc; use tcp://host:port instead"
                .to_string(),
        )
        .into());
    }

    if let Some(target) = addr.strip_prefix("tcp://") {
        validate_tcp_authority(target)?;
        return Ok(format!("http://{target}"));
    }

    if let Some(target) = addr.strip_prefix("http://") {
        validate_tcp_authority(target)?;
        return Ok(addr.to_string());
    }

    if addr.starts_with("https://") {
        return Err(TransportError::InvalidAddress(
            "https:// is reserved for control-plane URLs, not session links".to_string(),
        )
        .into());
    }

    if addr.contains("://") {
        let scheme = addr
            .split_once("://")
            .map(|(scheme, _)| scheme)
            .unwrap_or(addr);
        return Err(TransportError::InvalidAddress(format!(
            "unsupported address scheme '{scheme}'"
        ))
        .into());
    }

    validate_tcp_authority(addr)?;
    Ok(format!("http://{addr}"))
}

pub fn normalize_tcp_session_address(addr: &str) -> Result<String, Error> {
    let addr = addr.trim();
    if addr.is_empty() {
        return Err(TransportError::InvalidAddress("empty address".to_string()).into());
    }

    if addr.starts_with("unix://") {
        return Err(TransportError::InvalidAddress(
            "unix sockets are not supported by rlmesh-grpc; use tcp://host:port instead"
                .to_string(),
        )
        .into());
    }

    if let Some(target) = addr.strip_prefix("tcp://") {
        validate_tcp_authority(target)?;
        return Ok(format!("tcp://{target}"));
    }

    if let Some(target) = addr.strip_prefix("http://") {
        validate_tcp_authority(target)?;
        return Ok(format!("tcp://{target}"));
    }

    if addr.starts_with("https://") {
        return Err(TransportError::InvalidAddress(
            "https:// is reserved for control-plane URLs, not session links".to_string(),
        )
        .into());
    }

    if addr.contains("://") {
        let scheme = addr
            .split_once("://")
            .map(|(scheme, _)| scheme)
            .unwrap_or(addr);
        return Err(TransportError::InvalidAddress(format!(
            "unsupported address scheme '{scheme}'"
        ))
        .into());
    }

    validate_tcp_authority(addr)?;
    Ok(format!("tcp://{addr}"))
}

fn validate_tcp_authority(authority: &str) -> Result<(), Error> {
    let (host, port) = authority.rsplit_once(':').ok_or_else(|| {
        TransportError::InvalidAddress(format!(
            "missing port in tcp session endpoint '{authority}'"
        ))
    })?;
    if host.is_empty() {
        return Err(TransportError::InvalidAddress(format!(
            "missing host in tcp session endpoint '{authority}'"
        ))
        .into());
    }
    let _ = port.parse::<u16>().map_err(|error| {
        TransportError::InvalidAddress(format!("invalid tcp port '{port}': {error}"))
    })?;
    Ok(())
}

fn parse_port(value: &str, context: &str) -> Result<u16, Error> {
    value.parse::<u16>().map_err(|err| {
        TransportError::InvalidAddress(format!("invalid {context} '{value}': {err}")).into()
    })
}

fn normalize_bind_host(host: &str) -> &str {
    match host {
        "" | "localhost" => "127.0.0.1",
        _ => host,
    }
}

#[cfg(unix)]
fn normalize_unix_path(path: &str) -> Result<PathBuf, Error> {
    if path.is_empty() {
        return Err(
            TransportError::InvalidAddress("unix socket path cannot be empty".to_string()).into(),
        );
    }

    let socket_path = Path::new(path);
    if socket_path.is_absolute() {
        return Ok(socket_path.to_path_buf());
    }

    let cwd = std::env::current_dir()
        .map_err(|e| TransportError::InvalidAddress(format!("failed to read cwd: {e}")))?;
    Ok(cwd.join(socket_path))
}

#[cfg(not(unix))]
fn normalize_unix_path(path: &str) -> Result<PathBuf, Error> {
    let _ = path;
    Err(TransportError::InvalidAddress(
        "unix sockets are not supported on Windows; use tcp://host:port instead".to_string(),
    )
    .into())
}
