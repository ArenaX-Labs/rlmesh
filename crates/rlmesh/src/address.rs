use crate::{Error, Result};
use std::{fmt, str::FromStr};

pub(crate) const V1_REFUSAL: &str =
    "v1 endpoints need RLMesh 0.1; serve with rlmesh.serve / pass the session address";

pub(crate) fn is_v2(text: &str) -> bool {
    let text = text.trim();
    if let Some((scheme, _)) = text.split_once("://")
        && !matches!(scheme, "rlmesh" | "lockstep")
    {
        return false;
    }
    rlmesh_session::address::is_v2_address(text)
}

/// `text` with its key (`#…`) left out: what errors, logs and `Debug` show.
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
            return Err(Error::invalid_argument(V1_REFUSAL));
        }
        Ok(Self::Lockstep(value.to_string()))
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
            "host:1",
            "50051",
        ] {
            assert_eq!(
                ConnectAddress::parse(address).unwrap_err().to_string(),
                V1_REFUSAL
            );
        }
        for address in [":0/env", "host:7447/env", "env"] {
            assert!(ConnectAddress::parse(address).is_ok());
        }
    }
}
