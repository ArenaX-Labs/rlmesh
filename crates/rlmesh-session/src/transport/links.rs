//! The links a listening transport accepted, watched from below Zenoh.
//!
//! A peer that keeps its TCP connection up but stops reading (a frozen or
//! debugger-stopped process, a board whose task stopped reading while its
//! TCP stack still acknowledges) never lapses its Zenoh lease: its
//! keep-alives keep coming. What the host sends it piles up, and Zenoh
//! 1.10 then waits on that link without a deadline: closing its transport
//! (because a BLOCK message could not be queued, or because the host shuts
//! down) queues a `Close` message behind the full queue and blocks the
//! closing thread until a batch drains, which, for a link nothing reads, is
//! when the socket fails. Meanwhile every message routed to that peer waits
//! on the same queue, on the thread of whoever sent it.
//!
//! So the host fails the socket itself. A [`LinkGuard`] scans the process's
//! sockets accepted on the transport's listening ports every [`SCAN`], and
//! a link that had bytes waiting the whole of [`LINK_STALL`] while its peer
//! acknowledged less than [`STALL_PROGRESS`] of them is shut down at the
//! socket, with a zero linger so the kernel resets it rather than keep
//! offering the peer what it will not read. Zenoh's reads and writes then
//! fail, which is what ends every wait above. A link that is merely slow
//! (a Wi-Fi link with hundreds of milliseconds of delay, or a stall of a
//! second or two) acknowledges far more than that in the window.
//!
//! The scan needs the process's descriptors and the kernel's counters for
//! each: Linux (`/proc/self/fd`, `TCP_INFO`) and macOS (`/dev/fd`,
//! `TCP_CONNECTION_INFO`). Linux also gets a kernel fast path: listening
//! sockets carry a `TCP_USER_TIMEOUT` of [`LINK_SILENCE`], which accepted
//! sockets inherit, so the kernel resets a link that leaves data
//! unacknowledged that long and, on recent kernels, one whose window stays
//! closed that long however its probes are answered (it acts on its probe
//! timer, so often somewhat later). Elsewhere (Windows, other Unixes) the
//! guard watches and cuts nothing, and only [`Transport::close`]'s deadline
//! applies.
//!
//! [`Transport::close`]: super::Transport::close

use std::collections::HashMap;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use tracing::{debug, warn};

/// How often the guard looks at its links.
pub(crate) const SCAN: Duration = Duration::from_millis(250);
/// How long a link may have bytes waiting while its peer takes almost none
/// of them before the guard shuts it down: past a Wi-Fi link's stall of a
/// second or two.
pub(crate) const LINK_STALL: Duration = Duration::from_millis(2500);
/// What a peer must acknowledge within [`LINK_STALL`], while bytes wait for
/// it, to count as reading. A kernel that keeps a closed window's probes
/// answered may still let a few bytes trickle in.
pub(crate) const STALL_PROGRESS: u64 = 1024;
/// How long a host's link may leave what it sent unacknowledged, or keep
/// its window closed, before the kernel resets it.
#[cfg(target_os = "linux")]
pub(crate) const LINK_SILENCE: Duration = Duration::from_secs(2);

/// Watches the links accepted on a transport's listening ports, from a
/// thread of its own (never a runtime's, which a stuck Zenoh call may hold),
/// until dropped.
#[derive(Debug)]
pub(crate) struct LinkGuard {
    ports: Vec<u16>,
    /// Dropped to stop the thread.
    _stop: mpsc::Sender<()>,
}

/// Which links [`LinkGuard::cut`] shuts down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cut {
    /// Those with bytes waiting.
    Waiting,
    /// All of them.
    All,
}

impl LinkGuard {
    /// Guards the links accepted on `ports`, or nothing when there are
    /// none. Where this platform can't see them, it watches nothing and
    /// cuts nothing.
    pub(crate) fn start(mut ports: Vec<u16>) -> Option<Self> {
        ports.retain(|port| *port != 0);
        ports.sort_unstable();
        ports.dedup();
        if ports.is_empty() {
            return None;
        }
        let (stop, stopped) = mpsc::channel();
        if !sys::SUPPORTED {
            return Some(Self { ports, _stop: stop });
        }
        sys::bound_silent_links(&ports);
        let watched = ports.clone();
        let spawned = std::thread::Builder::new()
            .name("rlmesh-links".into())
            .spawn(move || watch(&watched, &stopped));
        match spawned {
            Ok(_) => debug!(?ports, "watching accepted links"),
            Err(err) => {
                warn!(%err, "no thread to watch links: a peer that stops reading is not cut")
            }
        }
        Some(Self { ports, _stop: stop })
    }

    /// Shuts down the links `which` names now: how many.
    pub(crate) fn cut(&self, which: Cut) -> usize {
        let mut cut = 0;
        for link in sys::accepted(&self.ports) {
            if (which == Cut::All || link.waiting > 0) && sys::cut(&link) {
                cut += 1;
            }
        }
        cut
    }
}

/// One accepted link, as the kernel sees it.
#[derive(Debug)]
#[cfg_attr(not(any(target_os = "linux", target_os = "macos")), allow(dead_code))]
struct Link {
    fd: i32,
    /// Tells this socket from a later one on the same descriptor.
    key: sys::Key,
    peer: Option<std::net::SocketAddr>,
    /// Bytes sent but unacknowledged, or not sent yet.
    waiting: u64,
    /// A count that grows as the peer takes bytes.
    taken: u64,
}

/// Since when a link has had bytes waiting, and what its peer had taken
/// then.
#[derive(Clone, Copy, Debug)]
struct Stall {
    since: Instant,
    taken: u64,
}

/// Whether a link stalled: `stall` updated for what the kernel says now.
fn stalled(stall: &mut Option<Stall>, waiting: u64, taken: u64, now: Instant) -> bool {
    if waiting == 0 {
        *stall = None;
        return false;
    }
    match stall {
        Some(s) if taken.saturating_sub(s.taken) < STALL_PROGRESS => {
            now.duration_since(s.since) >= LINK_STALL
        }
        _ => {
            *stall = Some(Stall { since: now, taken });
            false
        }
    }
}

fn watch(ports: &[u16], stop: &mpsc::Receiver<()>) {
    let mut stalls: HashMap<sys::Key, Option<Stall>> = HashMap::new();
    while let Err(mpsc::RecvTimeoutError::Timeout) = stop.recv_timeout(SCAN) {
        let now = Instant::now();
        let mut seen = HashMap::with_capacity(stalls.len());
        for link in sys::accepted(ports) {
            let mut stall = stalls.get(&link.key).copied().flatten();
            if stalled(&mut stall, link.waiting, link.taken, now) {
                if sys::cut(&link) {
                    warn!(
                        peer = ?link.peer,
                        waiting = link.waiting,
                        "a peer took almost nothing sent to it for {LINK_STALL:?}: its link is cut"
                    );
                }
                stall = None;
            }
            seen.insert(link.key, stall);
        }
        stalls = seen;
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[allow(unsafe_code, clippy::useless_conversion, clippy::cast_possible_wrap)]
mod sys {
    use std::mem::size_of;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
    use std::os::fd::RawFd;

    use super::Link;

    pub(super) const SUPPORTED: bool = true;

    /// The socket's inode, on Linux; elsewhere its local and peer address.
    #[cfg(target_os = "linux")]
    pub(super) type Key = u64;
    #[cfg(not(target_os = "linux"))]
    pub(super) type Key = (RawFd, Option<SocketAddr>, Option<SocketAddr>);

    #[cfg(target_os = "linux")]
    const FDS: &str = "/proc/self/fd";
    #[cfg(not(target_os = "linux"))]
    const FDS: &str = "/dev/fd";

    /// The process's open descriptors.
    fn descriptors() -> Vec<RawFd> {
        let Ok(dir) = std::fs::read_dir(FDS) else {
            return Vec::new();
        };
        dir.flatten()
            .filter_map(|e| e.file_name().to_str()?.parse().ok())
            .collect()
    }

    fn int_option(fd: RawFd, level: libc::c_int, name: libc::c_int) -> Option<libc::c_int> {
        let mut value: libc::c_int = 0;
        let mut len = size_of::<libc::c_int>() as libc::socklen_t;
        // SAFETY: both pointers are valid for the given length; a
        // descriptor that is no socket (or no longer open) only fails it.
        let ok =
            unsafe { libc::getsockopt(fd, level, name, (&raw mut value).cast(), &raw mut len) }
                == 0;
        ok.then_some(value)
    }

    fn address(
        fd: RawFd,
        get: unsafe extern "C" fn(
            libc::c_int,
            *mut libc::sockaddr,
            *mut libc::socklen_t,
        ) -> libc::c_int,
    ) -> Option<SocketAddr> {
        // SAFETY: an all-zero `sockaddr_storage` is valid.
        let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
        let mut len = size_of::<libc::sockaddr_storage>() as libc::socklen_t;
        // SAFETY: `storage` is valid for `len` bytes.
        if unsafe { get(fd, (&raw mut storage).cast(), &raw mut len) } != 0 {
            return None;
        }
        match libc::c_int::from(storage.ss_family) {
            libc::AF_INET => {
                // SAFETY: the kernel wrote a `sockaddr_in`.
                let v4 = unsafe { *(&raw const storage).cast::<libc::sockaddr_in>() };
                Some(SocketAddr::new(
                    IpAddr::V4(Ipv4Addr::from(u32::from_be(v4.sin_addr.s_addr))),
                    u16::from_be(v4.sin_port),
                ))
            }
            libc::AF_INET6 => {
                // SAFETY: the kernel wrote a `sockaddr_in6`.
                let v6 = unsafe { *(&raw const storage).cast::<libc::sockaddr_in6>() };
                Some(SocketAddr::new(
                    IpAddr::V6(Ipv6Addr::from(v6.sin6_addr.s6_addr)),
                    u16::from_be(v6.sin6_port),
                ))
            }
            _ => None,
        }
    }

    fn local(fd: RawFd) -> Option<SocketAddr> {
        address(fd, libc::getsockname)
    }

    fn peer(fd: RawFd) -> Option<SocketAddr> {
        address(fd, libc::getpeername)
    }

    /// A TCP socket on one of `ports`: whether it listens.
    fn on_ports(fd: RawFd, ports: &[u16]) -> Option<bool> {
        let kind = int_option(fd, libc::SOL_SOCKET, libc::SO_TYPE)?;
        if kind != libc::SOCK_STREAM {
            return None;
        }
        let at = local(fd)?;
        if !ports.contains(&at.port()) {
            return None;
        }
        Some(int_option(fd, libc::SOL_SOCKET, libc::SO_ACCEPTCONN)? != 0)
    }

    #[cfg(target_os = "linux")]
    fn key(fd: RawFd) -> Option<Key> {
        let target = std::fs::read_link(format!("/proc/self/fd/{fd}")).ok()?;
        target
            .to_str()?
            .strip_prefix("socket:[")?
            .strip_suffix(']')?
            .parse()
            .ok()
    }

    #[cfg(not(target_os = "linux"))]
    fn key(fd: RawFd) -> Option<Key> {
        Some((fd, local(fd), peer(fd)))
    }

    /// Bytes waiting, and bytes the peer acknowledged so far.
    #[cfg(target_os = "linux")]
    fn counters(fd: RawFd) -> Option<(u64, u64)> {
        // `struct tcp_info` up to `tcpi_bytes_acked` (Linux 4.2), at 120.
        const BYTES_ACKED: usize = 120;
        let mut info = [0_u8; 256];
        let mut len = info.len() as libc::socklen_t;
        // SAFETY: `info` is valid for `len` bytes.
        let ok = unsafe {
            libc::getsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_INFO,
                info.as_mut_ptr().cast(),
                &raw mut len,
            )
        } == 0;
        if !ok || (len as usize) < BYTES_ACKED + 8 {
            return None;
        }
        let acked = u64::from_ne_bytes(info[BYTES_ACKED..BYTES_ACKED + 8].try_into().ok()?);
        let mut waiting: libc::c_int = 0;
        // SAFETY: SIOCOUTQ writes one int.
        if unsafe { libc::ioctl(fd, libc::TIOCOUTQ, &raw mut waiting) } != 0 {
            return None;
        }
        Some((u64::try_from(waiting).unwrap_or(0), acked))
    }

    /// Bytes in the send buffer, and bytes first sent so far (what a
    /// closed window stops; its probes are a byte at a time).
    #[cfg(not(target_os = "linux"))]
    fn counters(fd: RawFd) -> Option<(u64, u64)> {
        // SAFETY: an all-zero `tcp_connection_info` is valid.
        let mut info: libc::tcp_connection_info = unsafe { std::mem::zeroed() };
        let mut len = size_of::<libc::tcp_connection_info>() as libc::socklen_t;
        // SAFETY: `info` is valid for `len` bytes.
        let ok = unsafe {
            libc::getsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_CONNECTION_INFO,
                (&raw mut info).cast(),
                &raw mut len,
            )
        } == 0;
        ok.then(|| {
            (
                u64::from(info.tcpi_snd_sbbytes),
                info.tcpi_txbytes
                    .saturating_sub(info.tcpi_txretransmitbytes),
            )
        })
    }

    /// The TCP links accepted on `ports`.
    pub(super) fn accepted(ports: &[u16]) -> Vec<Link> {
        descriptors()
            .into_iter()
            .filter_map(|fd| {
                if on_ports(fd, ports)? {
                    return None;
                }
                let key = key(fd)?;
                let (waiting, taken) = counters(fd)?;
                Some(Link {
                    fd,
                    key,
                    peer: peer(fd),
                    waiting,
                    taken,
                })
            })
            .collect()
    }

    /// Shuts `link` down and has its close reset it, if its descriptor
    /// still holds it. Works on a duplicate, so a descriptor closed and
    /// reused meanwhile is left alone.
    pub(super) fn cut(link: &Link) -> bool {
        // SAFETY: duplicating a descriptor has no effect on the original.
        let dup = unsafe { libc::fcntl(link.fd, libc::F_DUPFD_CLOEXEC, 0) };
        if dup < 0 {
            return false;
        }
        let same = key(dup).is_some_and(|key| key == same_key(link, dup));
        if same {
            let linger = libc::linger {
                l_onoff: 1,
                l_linger: 0,
            };
            // SAFETY: the value is valid for the given length; both calls
            // act on the socket `dup` refers to, which is `link`'s.
            unsafe {
                libc::setsockopt(
                    dup,
                    libc::SOL_SOCKET,
                    libc::SO_LINGER,
                    (&raw const linger).cast(),
                    size_of::<libc::linger>() as libc::socklen_t,
                );
                libc::shutdown(dup, libc::SHUT_RDWR);
            }
        }
        // SAFETY: `dup` is this function's own descriptor.
        unsafe { libc::close(dup) };
        same
    }

    /// `link`'s key as `dup` would report it if it is the same socket.
    #[cfg(target_os = "linux")]
    fn same_key(link: &Link, _dup: RawFd) -> Key {
        link.key
    }

    #[cfg(not(target_os = "linux"))]
    fn same_key(link: &Link, dup: RawFd) -> Key {
        (dup, link.key.1, link.key.2)
    }

    /// Sets [`super::LINK_SILENCE`] as the `TCP_USER_TIMEOUT` of the
    /// sockets listening on `ports`: the sockets they accept inherit it
    /// (Zenoh has no option for it).
    #[cfg(target_os = "linux")]
    pub(super) fn bound_silent_links(ports: &[u16]) {
        let timeout_ms = super::LINK_SILENCE.as_millis() as libc::c_int;
        let mut bounded = 0;
        for fd in descriptors() {
            if on_ports(fd, ports) != Some(true) {
                continue;
            }
            // SAFETY: the value is valid for the given length.
            let set = unsafe {
                libc::setsockopt(
                    fd,
                    libc::IPPROTO_TCP,
                    libc::TCP_USER_TIMEOUT,
                    (&raw const timeout_ms).cast(),
                    size_of::<libc::c_int>() as libc::socklen_t,
                )
            };
            if set == 0 {
                bounded += 1;
            }
        }
        tracing::debug!(?ports, bounded, "listeners reset silent links");
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn bound_silent_links(_ports: &[u16]) {}
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod sys {
    use super::Link;

    pub(super) const SUPPORTED: bool = false;
    pub(super) type Key = i32;

    pub(super) fn accepted(_ports: &[u16]) -> Vec<Link> {
        Vec::new()
    }

    pub(super) fn cut(_link: &Link) -> bool {
        false
    }

    pub(super) fn bound_silent_links(_ports: &[u16]) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_link_stalls_only_when_its_peer_takes_almost_nothing_for_the_whole_window() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let window = u64::try_from(LINK_STALL.as_millis()).expect("ms");
        let mut stall = None;
        // Nothing waiting: never stalled, however long.
        assert!(!stalled(&mut stall, 0, 0, at(0)));
        assert!(!stalled(&mut stall, 0, 0, at(window * 2)));
        // Bytes waiting from here, and the peer takes a trickle.
        let from = window * 2;
        assert!(!stalled(&mut stall, 10, 100, at(from)));
        assert!(!stalled(&mut stall, 10, 600, at(from + window - 1)));
        assert!(stalled(
            &mut stall,
            10,
            100 + STALL_PROGRESS - 1,
            at(from + window)
        ));
        // A slow but reading peer: the window starts over each time it
        // takes enough.
        let mut stall = None;
        let mut taken = 0;
        for ms in (0..window * 5).step_by(250) {
            taken += STALL_PROGRESS / 4;
            assert!(!stalled(&mut stall, 10, taken, at(ms)), "{ms} ms");
        }
        // Drained once: the window starts over.
        let mut stall = None;
        assert!(!stalled(&mut stall, 10, 0, at(0)));
        assert!(!stalled(&mut stall, 0, 0, at(window - 1)));
        assert!(!stalled(&mut stall, 10, 0, at(window)));
        assert!(!stalled(&mut stall, 10, 0, at(window * 2 - 1)));
        assert!(stalled(&mut stall, 10, 0, at(window * 2)));
    }
}
