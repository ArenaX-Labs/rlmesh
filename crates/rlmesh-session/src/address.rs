//! A session's address, `{ip}:{base_port}/{tenant}/{session}`: what a host
//! embedded with per-class TCP listeners prints (`rlmesh.v2.Host.address`)
//! and what a participant in another process joins with.

use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;
use std::time::Duration;

use crate::wire::{ParticipantId, PriorityClass, Role, SessionId, TenantId};

use crate::clock::SystemMonotonicClock;
use crate::error::SessionError;
use crate::participant::{Participant, ParticipantConfig};
use crate::transport::{Transport, TransportConfig};

/// Where a session's host listens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionAddress {
    /// The host's control-class listener; the other classes follow it.
    pub base: SocketAddr,
    /// The tenant namespace.
    pub tenant: TenantId,
    /// The session.
    pub session: SessionId,
}

impl FromStr for SessionAddress {
    type Err = SessionError;

    fn from_str(text: &str) -> Result<Self, SessionError> {
        let invalid = |why: String| {
            SessionError::Config(format!(
                "session address {text:?}: {why} (expected {{ip}}:{{port}}/{{tenant}}/{{session}})"
            ))
        };
        let mut parts = text.trim().splitn(3, '/');
        let (Some(base), Some(tenant), Some(session)) = (parts.next(), parts.next(), parts.next())
        else {
            return Err(invalid("three parts".into()));
        };
        Ok(Self {
            base: base.parse().map_err(|e| invalid(format!("{e}")))?,
            tenant: TenantId::new(tenant).map_err(|e| invalid(e.to_string()))?,
            session: session.parse().map_err(|e| invalid(format!("{e}")))?,
        })
    }
}

impl fmt::Display for SessionAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}/{}", self.base, self.tenant, self.session)
    }
}

impl SessionAddress {
    /// Opens a transport to the host (listening on the host's interface,
    /// any port) and attaches as `id`, on this process's
    /// `CLOCK_MONOTONIC` (so capture times line up with other local
    /// processes').
    pub async fn join(
        &self,
        id: &str,
        roles: Vec<Role>,
        priority: PriorityClass,
        host_timeout: Duration,
    ) -> Result<Participant, SessionError> {
        let listen_ip = if self.base.ip().is_loopback() {
            self.base.ip()
        } else {
            IpAddr::from([0, 0, 0, 0])
        };
        let transport = Transport::open(&TransportConfig::per_class_tcp(
            Some(SocketAddr::new(listen_ip, 0)),
            Some(self.base),
        )?)
        .await?;
        let mut config = ParticipantConfig::new(
            self.tenant.clone(),
            self.session,
            ParticipantId::new(id)?,
            roles,
            priority,
        );
        config.clock = SystemMonotonicClock.shared();
        config.host_timeout = host_timeout;
        Participant::attach(transport, config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_parse_and_print_back() {
        let session = SessionId::new();
        let text = format!("127.0.0.1:7447/local/{session}");
        let address: SessionAddress = text.parse().expect("parse");
        assert_eq!(address.base, "127.0.0.1:7447".parse().expect("addr"));
        assert_eq!(address.session, session);
        assert_eq!(address.to_string(), text);
        for bad in [
            "",
            "127.0.0.1:7447",
            "nowhere/local/x",
            "127.0.0.1:7447/local/x",
        ] {
            assert!(bad.parse::<SessionAddress>().is_err(), "{bad:?}");
        }
    }
}
