use crate::{Error, Result};
use std::{fmt, str::FromStr};

pub(crate) const V1_REFUSAL: &str =
    "v1 endpoints need RLMesh 0.1; serve with rlmesh.serve / pass the session address";

pub(crate) fn env_address(text: &str) -> String {
    rlmesh_session::address::defaulting_participant(text, rlmesh_session::address::DEFAULT_ENV)
}

pub(crate) fn bare_port(text: &str) -> Option<String> {
    let port = text.trim();
    (!port.is_empty() && port.bytes().all(|b| b.is_ascii_digit())).then(|| {
        format!(
            "{port:?} is a bare port, which names no host: write \":{port}\" for this machine's \
             port {port} (or \"host:{port}\")"
        )
    })
}

pub(crate) fn refusal(text: &str) -> Error {
    Error::invalid_argument(bare_port(text).unwrap_or_else(|| V1_REFUSAL.into()))
}

pub(crate) fn is_v2(text: &str) -> bool {
    let text = text.trim();
    if let Some((scheme, _)) = text.split_once("://")
        && !matches!(scheme, "rlmesh" | "lockstep")
    {
        return false;
    }
    rlmesh_session::address::is_v2_address(text)
}

pub(crate) fn redact(text: &str) -> String {
    rlmesh_session::address::redact(text)
}

/// A remote session participant address. `Display` and `Debug` leave its
/// key out; [`as_str`](Self::as_str) keeps it for dialing.
#[derive(Clone, PartialEq, Eq)]
pub enum ConnectAddress {
    /// A robot or model in a session (`:0/env`, `host:port/session/name`).
    Lockstep(String),
}
impl ConnectAddress {
    /// Parse a session participant address; v1 endpoints require RLMesh 0.1.
    pub fn parse(value: impl AsRef<str>) -> Result<Self> {
        let value = value.as_ref();
        if !is_v2(value) {
            return Err(refusal(value));
        }
        Ok(Self::Lockstep(value.to_string()))
    }

    /// Parse an env's participant address, where a lone `host:port` is
    /// short for `host:port/env`.
    pub fn env(value: impl AsRef<str>) -> Result<Self> {
        Self::parse(env_address(value.as_ref()))
    }
    /// The participant address as a string.
    pub fn as_str(&self) -> String {
        match self {
            Self::Lockstep(value) => value.clone(),
        }
    }
}
impl fmt::Display for ConnectAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&redact(&self.as_str()))
    }
}
impl fmt::Debug for ConnectAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lockstep(value) => f.debug_tuple("Lockstep").field(&redact(value)).finish(),
        }
    }
}
impl FromStr for ConnectAddress {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        Self::parse(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn v1_endpoints_require_a_01_runner() {
        for address in [
            "tcp://host:1",
            "unix:///tmp/env.sock",
            "https://host",
            "https://host:443#key/env",
            "tcp://host:1#key/env",
            "unix:///tmp/env.sock#key/env",
        ] {
            for parsed in [ConnectAddress::parse(address), ConnectAddress::env(address)] {
                assert_eq!(parsed.unwrap_err().to_string(), V1_REFUSAL, "{address}");
            }
        }
        for address in [":0/env", "host:7447/env", "env"] {
            assert!(ConnectAddress::parse(address).is_ok());
            assert!(ConnectAddress::env(address).is_ok());
        }
    }

    #[test]
    fn a_lone_host_and_port_is_an_envs_address() {
        for (short, full) in [
            ("127.0.0.1:5555", "127.0.0.1:5555/env"),
            (":0", ":0/env"),
            ("lab:7447", "lab:7447/env"),
        ] {
            assert_eq!(ConnectAddress::env(short).unwrap().as_str(), full);
            assert_eq!(
                ConnectAddress::parse(short).unwrap_err().to_string(),
                V1_REFUSAL
            );
        }
    }

    #[test]
    fn a_bare_port_names_the_host_it_needs() {
        for parsed in [ConnectAddress::parse("5555"), ConnectAddress::env("5555")] {
            let why = parsed.unwrap_err().to_string();
            assert!(
                why.contains("bare port") && why.contains("\":5555\""),
                "{why}"
            );
        }
    }
}
