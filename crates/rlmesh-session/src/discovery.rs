//! Finding rlmeshd on the LAN (scene design §3.1). rlmeshd advertises
//! itself over mDNS as `_rlmesh._tcp.local` ([`advertise`]), with only its
//! host id (TXT `id`) and its base port: a machine or board that knows the
//! host's id and static key from pairing or provisioning finds its current
//! address there, and a wrong or forged answer only costs a failed IK
//! handshake.
//!
//! A paired machine reads its host file ([`crate::pairing::HostFile`]) the
//! way a board reads its NVS ([`locate`]): each stored address with a
//! [`CONNECT`] timeout, then an mDNS query keeping the instances whose `id`
//! is the host's. Each address that accepts a connection is tried with the
//! caller's handshake, and only one the host authenticated at replaces the
//! stored ones: a forged answer, or an old address now some other service,
//! costs one failed handshake and is never kept. Every address is a base
//! port: traffic class i listens at base + i.

use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::Duration;

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

use crate::error::SessionError;
use crate::pairing::HostFile;

/// The mDNS service type rlmeshd advertises.
pub const SERVICE: &str = "_rlmesh._tcp.local.";
/// How long one stored address gets to accept a connection.
pub const CONNECT: Duration = Duration::from_secs(2);
/// How long an mDNS query listens for answers.
pub const BROWSE: Duration = Duration::from_secs(3);
/// How long an attach at a found address waits for the host.
pub const ATTACH: Duration = Duration::from_secs(10);

fn mdns(err: mdns_sd::Error) -> SessionError {
    SessionError::Config(format!("mDNS: {err}"))
}

/// Whether `text` is a host id (`rlm-` and 6 hex digits).
pub fn is_host_id(text: &str) -> bool {
    text.strip_prefix("rlm-")
        .is_some_and(|hex| hex.len() == 6 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// rlmeshd's mDNS advertisement; dropping it withdraws it.
pub struct Advertisement {
    daemon: ServiceDaemon,
}

impl Drop for Advertisement {
    fn drop(&mut self) {
        let _ = self.daemon.shutdown();
    }
}

/// Advertises the host `host_id` listening on `base`: on every interface's
/// addresses when `base` is unspecified (`0.0.0.0`), else on its own.
///
/// # Errors
/// [`SessionError::Config`] when the mDNS responder can't start.
pub fn advertise(host_id: &str, base: SocketAddr) -> Result<Advertisement, SessionError> {
    let daemon = ServiceDaemon::new().map_err(mdns)?;
    let ip = if base.ip().is_unspecified() {
        String::new()
    } else {
        base.ip().to_string()
    };
    let info = ServiceInfo::new(
        SERVICE,
        host_id,
        &format!("{host_id}.local."),
        ip.as_str(),
        base.port(),
        &[("id", host_id)][..],
    )
    .map_err(mdns)?;
    let info = if ip.is_empty() {
        info.enable_addr_auto()
    } else {
        info
    };
    daemon.register(info).map_err(mdns)?;
    Ok(Advertisement { daemon })
}

/// The base addresses one mDNS answer gives for `host_id`: none unless its
/// TXT `id` is that id.
fn answer(host_id: &str, id: Option<&str>, ips: &[IpAddr], port: u16) -> Vec<String> {
    if id != Some(host_id) {
        return Vec::new();
    }
    let mut ips = ips.to_vec();
    // IPv4 first, then IPv6; the same order every time.
    ips.sort_by_key(|ip| (ip.is_ipv6(), *ip));
    ips.into_iter()
        .map(|ip| SocketAddr::new(ip, port).to_string())
        .collect()
}

/// Queries mDNS for `wait`: the base addresses of every instance of
/// [`SERVICE`] whose TXT `id` is `host_id`.
///
/// # Errors
/// [`SessionError::Config`] when the mDNS querier can't start.
pub async fn browse(host_id: &str, wait: Duration) -> Result<Vec<String>, SessionError> {
    let host_id = host_id.to_string();
    tokio::task::spawn_blocking(move || {
        let daemon = ServiceDaemon::new().map_err(mdns)?;
        let events = daemon.browse(SERVICE).map_err(mdns)?;
        let deadline = std::time::Instant::now() + wait;
        let mut found = Vec::new();
        while let Ok(event) = events.recv_deadline(deadline) {
            if let ServiceEvent::ServiceResolved(service) = event {
                let ips: Vec<IpAddr> = service.addresses.iter().map(|a| a.to_ip_addr()).collect();
                let id = service.get_property_val_str("id");
                for address in answer(&host_id, id, &ips, service.port) {
                    if !found.contains(&address) {
                        found.push(address);
                    }
                }
            }
        }
        let _ = daemon.shutdown();
        Ok(found)
    })
    .await
    .map_err(|err| SessionError::Config(format!("mDNS: {err}")))?
}

/// Every one of `addresses` (`host:port`) that accepts a connection within
/// [`CONNECT`], resolved, in order.
async fn reachable(addresses: &[String]) -> Vec<SocketAddr> {
    let addresses = addresses.to_vec();
    tokio::task::spawn_blocking(move || {
        addresses
            .iter()
            .filter_map(|address| {
                address
                    .to_socket_addrs()
                    .ok()?
                    .find(|base| TcpStream::connect_timeout(base, CONNECT).is_ok())
            })
            .collect()
    })
    .await
    .unwrap_or_default()
}

/// The host file of host `host_id` in `dir`.
pub fn host_file(dir: &Path, host_id: &str) -> PathBuf {
    dir.join(format!("{host_id}.toml"))
}

/// Finds the host of the host file at `path` and attaches there with
/// `attach` (an authenticated handshake): at each stored address that
/// accepts a connection, then at each one mDNS answers for its id. The
/// mDNS answers replace the stored addresses in the file only once the
/// host authenticated at one of them. The file, and what `attach` gave.
///
/// # Errors
/// What the last attach failed with, or [`SessionError::Timeout`] when no
/// address accepted a connection; what reading or writing the file, or
/// querying mDNS, failed with.
pub async fn locate<T, F, Fut>(path: &Path, mut attach: F) -> Result<(HostFile, T), SessionError>
where
    F: FnMut(SocketAddr) -> Fut,
    Fut: std::future::Future<Output = Result<T, SessionError>>,
{
    let mut file = HostFile::read(path)?;
    let mut failed = None;
    let stored = reachable(&file.addresses).await;
    for base in &stored {
        match attach(*base).await {
            Ok(attached) => return Ok((file, attached)),
            Err(err) => failed = Some(err),
        }
    }
    let found = browse(&file.host_id, BROWSE).await?;
    for base in reachable(&found).await {
        if stored.contains(&base) {
            continue;
        }
        match attach(base).await {
            Ok(attached) => {
                if found != file.addresses {
                    file.addresses = found;
                    if let Some(dir) = path.parent() {
                        file.write(dir)?;
                    }
                }
                return Ok((file, attached));
            }
            Err(err) => failed = Some(err),
        }
    }
    Err(failed.unwrap_or_else(|| {
        SessionError::Timeout(
            CONNECT,
            format!(
                "host {} at {:?} or through mDNS ({} answered)",
                file.host_id,
                file.addresses,
                found.len()
            ),
        )
    }))
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, TcpListener};

    use super::*;
    use crate::crypto::StaticKeypair;
    use crate::pairing::Paired;
    use crate::wire::PairInfo;

    #[test]
    fn an_answer_counts_only_for_its_host_id() {
        let ips = [IpAddr::from([10, 0, 0, 5]), "fe80::1".parse().expect("ip")];
        assert_eq!(
            answer("rlm-3f9a2c", Some("rlm-3f9a2c"), &ips, 7447),
            ["10.0.0.5:7447", "[fe80::1]:7447"]
        );
        assert!(answer("rlm-3f9a2c", Some("rlm-000000"), &ips, 7447).is_empty());
        assert!(answer("rlm-3f9a2c", None, &ips, 7447).is_empty());
        assert!(is_host_id("rlm-3f9a2c") && !is_host_id("rig-pi.local"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stale_stored_address_falls_back_to_the_advertisement_by_id() {
        // A host listening on loopback, advertised there under a fresh id.
        let host = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("listen");
        let base = host.local_addr().expect("addr");
        let host_id = format!("rlm-{:06x}", rand::random::<u32>() & 0xff_ffff);
        let _advertised = advertise(&host_id, base).expect("advertise");
        assert!(
            browse(&host_id, BROWSE)
                .await
                .expect("browse")
                .contains(&base.to_string()),
            "resolvable by its id"
        );

        // The file remembers an address nothing listens on any more, and
        // the advertised one is a service that fails the handshake: no
        // host, and the file is kept as it was.
        let stale = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("listen");
        let gone = stale.local_addr().expect("addr").to_string();
        drop(stale);
        let dir = tempfile::tempdir().expect("dir");
        let paired = Paired {
            info: PairInfo {
                device: "laptop-josh".into(),
                host_key: [7; 32],
                host_id: host_id.clone(),
                addresses: Vec::new(),
            },
            device: StaticKeypair::from_private(&mut [5; 32]),
        };
        let path = HostFile::new(&paired, &gone)
            .write(dir.path())
            .expect("write");
        let refused = |_: SocketAddr| async { Err::<(), _>(SessionError::Closed) };
        assert!(locate(&path, refused).await.is_err());
        assert_eq!(HostFile::read(&path).expect("read").addresses, [gone]);

        // A stored address some other service took (it accepts, then
        // fails the handshake) comes before the real host's advertisement.
        let other = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("listen");
        let taken = other.local_addr().expect("addr");
        let path = HostFile::new(&paired, &taken.to_string())
            .write(dir.path())
            .expect("write");
        let host = |at: SocketAddr| async move {
            if at == base {
                Ok(at)
            } else {
                Err(SessionError::Closed)
            }
        };
        let (file, found) = locate(&path, host).await.expect("located");
        assert_eq!(found, base);
        let kept = HostFile::read(&path).expect("read");
        assert_eq!(kept.addresses, file.addresses);
        assert!(
            kept.addresses.contains(&base.to_string())
                && !kept.addresses.contains(&taken.to_string()),
            "{:?}",
            kept.addresses
        );
    }
}
