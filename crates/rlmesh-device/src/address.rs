//! The address an embedded host prints, as a board takes it:
//! `{host}:{port}[/{session}]#{secret}.{host key}[/{name}]`. The key is
//! session-core's [`AddressKey`]; the tenant is always `local`.

use alloc::string::{String, ToString};

use rlmesh_session_core::auth::AddressKey;

/// The session a host address without one names.
pub const DEFAULT_SESSION: &str = "default";
/// The tenant every local session runs in.
pub const TENANT: &str = "local";

/// A parsed address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Address {
    /// The host's IP or name.
    pub host: String,
    /// Its first (control) port; class `i` listens on `port + i`.
    pub port: u16,
    /// The session's name or UUID.
    pub session: String,
    /// The session's key.
    pub key: AddressKey,
    /// A participant name, when the address carries one.
    pub name: Option<String>,
}

/// Parses `text`, or `None` when it isn't an address with its key.
pub fn parse(text: &str) -> Option<Address> {
    let (place, keyed) = text.trim().split_once('#')?;
    // The key's base32 has no '/': a name follows it after one.
    let (key, name) = match keyed.split_once('/') {
        Some((key, name)) if !name.is_empty() => (key, Some(name.to_string())),
        Some(_) => return None,
        None => (keyed, None),
    };
    let (hostport, session) = match place.split_once('/') {
        Some((hostport, session)) if !session.is_empty() => (hostport, session),
        Some(_) => return None,
        None => (place, DEFAULT_SESSION),
    };
    let (host, port) = hostport.rsplit_once(':')?;
    let port: u16 = port.parse().ok()?;
    // Four consecutive ports, one per traffic class.
    if host.is_empty() || port == 0 || port > u16::MAX - 3 {
        return None;
    }
    Some(Address {
        host: host.to_string(),
        port,
        session: session.to_string(),
        key: key.parse().ok()?,
        name,
    })
}

#[cfg(test)]
mod tests {
    use rlmesh_session_core::auth::address::base32;

    use super::*;

    fn key() -> String {
        alloc::format!("{}.{}", base32(&[1; 32]), base32(&[2; 32]))
    }

    #[test]
    fn a_host_address_names_the_default_session() {
        let a = parse(&alloc::format!("192.168.1.20:7447#{}", key())).unwrap();
        assert_eq!(
            (a.host.as_str(), a.port, a.session.as_str(), a.name),
            ("192.168.1.20", 7447, DEFAULT_SESSION, None)
        );
        assert_eq!(a.key, key().parse().unwrap());
    }

    #[test]
    fn a_session_and_a_name_are_read() {
        let a = parse(&alloc::format!("h.local:9000/bench-3#{}/eyes", key())).unwrap();
        assert_eq!(
            (
                a.host.as_str(),
                a.port,
                a.session.as_str(),
                a.name.as_deref()
            ),
            ("h.local", 9000, "bench-3", Some("eyes"))
        );
    }

    #[test]
    fn anything_else_is_refused() {
        for bad in [
            "192.168.1.20:7447".to_string(),
            alloc::format!("192.168.1.20#{}", key()),
            alloc::format!(":7447#{}", key()),
            alloc::format!("h:0#{}", key()),
            alloc::format!("h:65534#{}", key()),
            alloc::format!("h:7447/#{}", key()),
            alloc::format!("h:7447#{}/", key()),
            "h:7447#notakey".to_string(),
        ] {
            assert_eq!(parse(&bad), None, "{bad}");
        }
    }
}
