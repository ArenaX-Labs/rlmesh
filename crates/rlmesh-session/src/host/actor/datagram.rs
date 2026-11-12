//! The session's robots on the host's datagram path: their routes, kept in
//! step with their parts and bindings, and the fallbacks either end makes.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rlmesh_session_core::datagram::{self, HEADER_LEN, MAX_DATAGRAM};
use tracing::{info, warn};

use crate::host::HostActor;
use crate::host::datagram::{DatagramRelay, Down, PathCounts, RobotPath, Up};
use crate::metrics::PathTime;
use crate::transport::{TrafficClass, declare_publisher, stream_qos};
use crate::wire::version::LINK_DATAGRAM;
use crate::wire::{
    AuthoritySnapshot, ChannelId, ChannelPriority, DataPath, DatagramRoute, DatagramRoutes,
    DatagramStats, PartSpec, ParticipantId, Role,
};

/// The control link counts as healthy while the robot's heartbeats arrive
/// at least this often.
const HEALTHY: Duration = Duration::from_millis(300);
/// The path is broken one way when, with the control link healthy, the
/// sending end sent at least [`BROKEN_SENT`] datagrams and for at least this
/// long the other end opened none of them. Long enough that a stall of the
/// whole link (a second or two on a weak Wi-Fi) is not mistaken for it.
const BROKEN_AFTER: Duration = Duration::from_secs(2);
/// See [`BROKEN_AFTER`]. Few: a robot whose chunks stop arriving goes safe
/// within its hold, and its policy sends no more after that.
const BROKEN_SENT: u64 = 3;
/// After moving a robot's parts back to the session's links, the host
/// tries the path again this long after (doubling each time it is broken
/// again, up to [`RETRY_MAX`]), once the robot's probes arrive.
const RETRY_FIRST: Duration = Duration::from_secs(5);
/// See [`RETRY_FIRST`].
const RETRY_MAX: Duration = Duration::from_secs(120);
/// A path that has carried the parts this long starts its retries over
/// from [`RETRY_FIRST`].
const STABLE: Duration = Duration::from_secs(60);
/// A robot's probes count as arriving while the newest is at most this old
/// (it sends one a second).
const PROBING: Duration = Duration::from_secs(3);

/// The datagram path of one session's robots.
pub(in crate::host) struct DatagramSession {
    pub(in crate::host) relay: Arc<DatagramRelay>,
    robots: BTreeMap<ParticipantId, Entry>,
}

/// One robot generation on the path.
struct Entry {
    path: Arc<RobotPath>,
    downs: Vec<Arc<Down>>,
    ups: Vec<Arc<Up>>,
    /// The realtime parts the routes were made for, by channel.
    parts: Vec<ChannelId>,
    /// The board's own fallbacks, as its heartbeats last said (a board
    /// built before the host judged the path gives it up itself).
    board_fallbacks: u32,
    /// Each way's judgment: the counts when the receiving end last opened
    /// one, and since when.
    down: Mark,
    up: Mark,
    /// Its parts' time on each path and the moves between them.
    time: Times,
    /// When the path is tried again, after the parts were moved back.
    retry_at: Option<Instant>,
    retry_after: Duration,
}

impl Entry {
    /// Both ways judged from now on.
    fn remark(&mut self, now: Instant, board: Option<DatagramStats>) {
        let host = self.path.counts();
        let board = board.unwrap_or_default();
        self.down = Mark::new(now, host.sent, board.received, 0);
        self.up = Mark::new(now, board.sent, host.samples, host.probes);
    }

    /// The way that is broken now, if one is.
    fn broken(
        &mut self,
        now: Instant,
        host: PathCounts,
        board: DatagramStats,
    ) -> Option<&'static str> {
        let down = self.down.broken(now, host.sent, board.received, 0);
        let up = self.up.broken(now, board.sent, host.samples, host.probes);
        if down {
            Some("its chunks sent over UDP reach the board no more")
        } else if up {
            Some("its samples sent over UDP reach the host no more")
        } else {
            None
        }
    }
}

/// One way of the path, since the receiving end last opened a datagram:
/// what the sending end had sent then, and what the receiving end had
/// opened.
#[derive(Clone, Copy)]
struct Mark {
    at: Instant,
    sent: u64,
    opened: u64,
    /// Probes that opened then (up: the board's sends count its probes).
    probes: u64,
}

impl Mark {
    fn new(at: Instant, sent: u64, opened: u64, probes: u64) -> Self {
        Self {
            at,
            sent,
            opened,
            probes,
        }
    }

    /// Whether this way is broken: nothing it sent opened at the other end,
    /// though it sent enough over long enough to tell. A new count opened
    /// moves the mark: ordinary loss never adds up to broken, and a way that
    /// sends nothing (a policy that stopped) never is.
    fn broken(&mut self, now: Instant, sent: u64, opened: u64, probes: u64) -> bool {
        if opened > self.opened || sent < self.sent {
            *self = Self::new(now, sent, opened, probes);
            return false;
        }
        // Up, the board's count holds its probes too: those that opened
        // are no sample.
        let unanswered = (sent - self.sent).saturating_sub(probes.saturating_sub(self.probes));
        unanswered >= BROKEN_SENT && now.saturating_duration_since(self.at) >= BROKEN_AFTER
    }
}

/// Where a robot's realtime parts rode, from when it attached asking for
/// the path: on the session's links until its first routes.
struct Times {
    on_datagram: bool,
    since: Instant,
    datagram: Duration,
    session: Duration,
    fallbacks: u32,
    returns: u32,
}

impl Times {
    fn new(now: Instant) -> Self {
        Self {
            on_datagram: false,
            since: now,
            datagram: Duration::ZERO,
            session: Duration::ZERO,
            fallbacks: 0,
            returns: 0,
        }
    }

    /// A move: counted as a fallback or a return when `counted`.
    fn switch(&mut self, now: Instant, on_datagram: bool, counted: bool) {
        if on_datagram == self.on_datagram {
            return;
        }
        let spent = now.saturating_duration_since(self.since);
        if self.on_datagram {
            self.datagram += spent;
        } else {
            self.session += spent;
        }
        self.since = now;
        self.on_datagram = on_datagram;
        if counted && on_datagram {
            self.returns += 1;
        } else if counted {
            self.fallbacks += 1;
        }
    }

    fn at(&self, now: Instant) -> PathTime {
        let spent = now.saturating_duration_since(self.since);
        let (datagram, session) = if self.on_datagram {
            (self.datagram + spent, self.session)
        } else {
            (self.datagram, self.session + spent)
        };
        PathTime {
            datagram,
            session,
            fallbacks: self.fallbacks,
            returns: self.returns,
        }
    }
}

impl DatagramSession {
    pub(in crate::host) fn new(relay: Arc<DatagramRelay>) -> Self {
        Self {
            relay,
            robots: BTreeMap::new(),
        }
    }

    fn remove(&mut self, robot: &ParticipantId) {
        if let Some(mut entry) = self.robots.remove(robot) {
            drop_routes(&self.relay, &mut entry);
            self.relay.remove_probe(&entry.path);
        }
    }
}

fn drop_routes(relay: &DatagramRelay, entry: &mut Entry) {
    for down in entry.downs.drain(..) {
        relay.remove_down(&down);
    }
    for up in entry.ups.drain(..) {
        relay.remove_up(&up);
    }
}

/// Why a part's chunks can't ride a datagram: the largest sealed chunk a
/// binding of `state` sends it, over the limit, and whose binding.
fn chunk_too_large(state: &AuthoritySnapshot, part: &PartSpec) -> Option<(usize, String, u32)> {
    let stride = part
        .command
        .as_ref()?
        .layout()
        .ok()
        .and_then(|l| l.stride())?;
    state
        .bindings
        .iter()
        .filter(|b| b.channels.contains(&part.channel))
        .filter_map(|b| {
            let grid = b.grids.iter().find(|g| g.channel == part.channel)?;
            (grid.chunk > 0).then(|| {
                let bound = datagram::chunk_bound(
                    part.channel,
                    &b.participant,
                    b.contracts.len(),
                    grid.chunk,
                    stride,
                );
                (bound, b.participant.to_string(), grid.chunk)
            })
        })
        .find(|(bound, _, _)| !datagram::fits(*bound))
}

/// The most a sealed envelope may take of one datagram.
const ROOM: usize = MAX_DATAGRAM - HEADER_LEN;

impl HostActor {
    /// Keeps every robot's routes in step with the session: a robot that
    /// attached asking for the path gets a probe route, its realtime parts
    /// routes when it declares them (a part whose chunks or samples could
    /// be larger than a datagram stays on the session), and a binding that
    /// would send larger chunks moves that part back. A robot that left or
    /// attached again loses its routes.
    pub(super) async fn sync_datagram(&mut self) {
        let Some(mut session) = self.datagram.take() else {
            return;
        };
        let state = self.snapshot();
        let eligible: Vec<_> = state
            .roster
            .iter()
            .filter(|p| {
                p.present
                    && p.has_role(Role::Embodiment)
                    && p.supports.iter().any(|f| f == LINK_DATAGRAM)
            })
            .map(|p| (p.id.clone(), p.generation))
            .collect();
        let gone: Vec<ParticipantId> = session
            .robots
            .iter()
            .filter(|(id, entry)| {
                !eligible
                    .iter()
                    .any(|(e, g)| e == *id && *g == entry.path.generation)
            })
            .map(|(id, _)| id.clone())
            .collect();
        for robot in gone {
            session.remove(&robot);
        }
        for (robot, generation) in eligible {
            if !session.robots.contains_key(&robot) {
                let peer = self
                    .peers
                    .read()
                    .ok()
                    .and_then(|p| p.current.get(&robot).cloned())
                    .filter(|p| p.identity.generation == generation);
                let Some(peer) = peer else {
                    continue;
                };
                let probe_key = self.keys.datagram_probe(&robot);
                let Some(path) =
                    session
                        .relay
                        .add_robot(robot.clone(), peer, probe_key, self.host_keys.clone())
                else {
                    warn!(%robot, "no free datagram route: its realtime parts ride TCP");
                    continue;
                };
                let now = Instant::now();
                session.robots.insert(
                    robot.clone(),
                    Entry {
                        path,
                        downs: Vec::new(),
                        ups: Vec::new(),
                        parts: Vec::new(),
                        board_fallbacks: 0,
                        down: Mark::new(now, 0, 0, 0),
                        up: Mark::new(now, 0, 0, 0),
                        time: Times::new(now),
                        retry_at: None,
                        retry_after: RETRY_FIRST,
                    },
                );
            }
            self.sync_routes(&mut session, &state, &robot).await;
        }
        self.datagram = Some(session);
    }

    /// One robot's routes: made when its realtime parts first appear (or
    /// change), and afterwards only taken away.
    async fn sync_routes(
        &mut self,
        session: &mut DatagramSession,
        state: &AuthoritySnapshot,
        robot: &ParticipantId,
    ) {
        let Some(entry) = session.robots.get_mut(robot) else {
            return;
        };
        if !lock_active(&entry.path) {
            return;
        }
        let parts: Vec<&PartSpec> = state
            .parts
            .iter()
            .filter(|p| &p.owner == robot && p.priority == ChannelPriority::Realtime)
            .collect();
        let channels: Vec<ChannelId> = parts.iter().map(|p| p.channel).collect();
        if channels == entry.parts {
            // A binding that would send a part larger chunks moves it back.
            let mut kept = Vec::with_capacity(entry.downs.len());
            for down in entry.downs.drain(..) {
                let too_large = parts
                    .iter()
                    .find(|p| p.channel == down.channel)
                    .and_then(|p| Some((p, chunk_too_large(state, p)?)));
                match too_large {
                    Some((part, (bound, by, chunk))) => {
                        session.relay.remove_down(&down);
                        self.warn_chunk_too_large(part, bound, &by, chunk);
                    }
                    None => kept.push(down),
                }
            }
            entry.downs = kept;
            return;
        }
        drop_routes(&session.relay, entry);
        entry.parts = channels;
        let path = entry.path.clone();
        let clock = state.clock_mode;
        let mut downs = Vec::new();
        let mut ups = Vec::new();
        for part in &parts {
            if part.command.is_some() {
                match chunk_too_large(state, part) {
                    Some((bound, by, chunk)) => self.warn_chunk_too_large(part, bound, &by, chunk),
                    None => {
                        let key = self.keys.commands(robot, part.path.as_str());
                        let realtime = self.transport.session(TrafficClass::Realtime).clone();
                        if let Some(down) = session
                            .relay
                            .add_down(&path, part.channel, &realtime, key)
                            .await
                        {
                            downs.push(down);
                        }
                    }
                }
            }
            let streams = [
                part.stream().ok(),
                part.applied_stream().and_then(Result::ok),
            ];
            for stream in streams.into_iter().flatten() {
                if stream.event {
                    continue;
                }
                let bound = datagram::sample_bound(&stream);
                if !datagram::fits(bound) {
                    if let Some(held_back) =
                        self.quiet
                            .admit("sample too large", &stream.qualified(), Instant::now())
                    {
                        warn!(
                            part = %stream.qualified(),
                            bytes = bound,
                            room = ROOM,
                            held_back,
                            "its samples stay on TCP: a sealed sample can be {bound} bytes, past \
                             the datagram path's {ROOM}; send smaller samples, or keep the part \
                             on TCP (as it is now)"
                        );
                    }
                    continue;
                }
                let key = self.keys.data(robot, &stream.path);
                let qos = stream_qos(&stream, clock);
                let publisher = match declare_publisher(
                    &self.transport,
                    TrafficClass::Realtime,
                    key.clone(),
                    qos,
                )
                .await
                {
                    Ok(publisher) => publisher,
                    Err(err) => {
                        warn!(part = %stream.qualified(), %err, "no relay publisher: its samples stay on TCP");
                        continue;
                    }
                };
                if let Some(up) =
                    session
                        .relay
                        .add_up(&path, stream.channel, stream.applied, key, publisher)
                {
                    ups.push(up);
                }
            }
        }
        let board = self.links.get(robot).and_then(|l| l.datagram);
        let Some(entry) = session.robots.get_mut(robot) else {
            return;
        };
        let now = Instant::now();
        entry.remark(now, board);
        if !downs.is_empty() || !ups.is_empty() {
            entry.time.switch(now, true, false);
        }
        info!(
            %robot,
            port = session.relay.port(),
            down = downs.len(),
            up = ups.len(),
            "realtime parts on the datagram path (UDP)"
        );
        entry.downs = downs;
        entry.ups = ups;
    }

    fn warn_chunk_too_large(&mut self, part: &PartSpec, bound: usize, by: &str, chunk: u32) {
        let name = format!("{}:{}", part.owner, part.path);
        if let Some(held_back) = self.quiet.admit("chunk too large", &name, Instant::now()) {
            warn!(
                part = %name,
                bytes = bound,
                room = ROOM,
                held_back,
                "its chunks stay on TCP: {by}'s binding sends up to {chunk} actions a chunk, up to \
                 {bound} bytes sealed, past the datagram path's {ROOM}; send fewer actions per \
                 chunk, or keep the part on TCP (as it is now)"
            );
        }
    }

    /// Every tick: a chunk that turned out too large moves that part back to
    /// TCP; a path shown broken one way (see [`Mark::broken`]) moves the
    /// robot's realtime parts back, unless the session is strict
    /// ([`crate::HostConfig::datagram_strict`]); and a robot moved back is
    /// tried on the path again, without a new attach, once its probes
    /// arrive and its retry time has come.
    pub(super) async fn check_datagram(&mut self) {
        let Some(mut session) = self.datagram.take() else {
            return;
        };
        let now = Instant::now();
        let now_ns = self.now();
        let strict = self.config.datagram_strict;
        let mut changed = false;
        let mut retried = false;
        let robots: Vec<ParticipantId> = session.robots.keys().cloned().collect();
        for robot in robots {
            let healthy = self
                .links
                .get(&robot)
                .and_then(|l| l.stats.last_arrival_ns())
                .is_some_and(|at| {
                    now_ns.saturating_sub(at) < i64::try_from(HEALTHY.as_nanos()).unwrap_or(0)
                });
            let board = self.links.get(&robot).and_then(|l| l.datagram);
            let board_fallbacks = board.map_or(0, |d| d.fallbacks);
            let Some(entry) = session.robots.get_mut(&robot) else {
                continue;
            };
            if !lock_active(&entry.path) {
                // Moved back: tried again once its retry time has come, its
                // control link is healthy and its probes arrive.
                let probing = entry.path.probe_age(now).is_some_and(|age| age <= PROBING);
                if healthy && probing && entry.retry_at.is_some_and(|at| now >= at) {
                    entry.retry_at = None;
                    entry.path.reactivate();
                    // Its routes anew, from the next sync.
                    entry.parts.clear();
                    entry.time.switch(now, true, true);
                    let time = entry.time.at(now);
                    info!(
                        %robot,
                        returns = time.returns,
                        on_udp_s = time.datagram.as_secs(),
                        on_tcp_s = time.session.as_secs(),
                        "its probes arrive again: its realtime parts go back on the datagram path"
                    );
                    retried = true;
                }
                continue;
            }
            if now.saturating_duration_since(entry.time.since) >= STABLE {
                entry.retry_after = RETRY_FIRST;
            }
            if board_fallbacks > entry.board_fallbacks {
                entry.board_fallbacks = board_fallbacks;
                Self::move_back(&session.relay, entry, now, &robot, "the board gave it up");
                changed = true;
                continue;
            }
            let oversize: Vec<(ChannelId, usize)> = entry
                .downs
                .iter()
                .filter_map(|d| Some((d.channel, d.oversize()?)))
                .collect();
            for (channel, bytes) in oversize {
                if let Some(at) = entry.downs.iter().position(|d| d.channel == channel) {
                    let down = entry.downs.remove(at);
                    session.relay.remove_down(&down);
                }
                let name = self.authority.part_name(channel);
                warn!(
                    part = %name,
                    bytes,
                    room = ROOM,
                    "its chunks go back to TCP: one sealed chunk was {bytes} bytes, past the \
                     datagram path's {ROOM}; send fewer actions per chunk, or keep the part on TCP"
                );
                changed = true;
            }
            // A link that is down altogether proves nothing about the path.
            let (Some(board), true) = (board, healthy) else {
                entry.remark(now, board);
                continue;
            };
            let Some(why) = entry.broken(now, entry.path.counts(), board) else {
                continue;
            };
            if strict {
                if let Some(held_back) = self.quiet.admit("datagram broken", robot.as_str(), now) {
                    warn!(
                        %robot,
                        held_back,
                        "{why}, but the session is strict (--board-data udp-only): its realtime \
                         parts stay on the datagram path"
                    );
                }
                entry.remark(now, Some(board));
                continue;
            }
            Self::move_back(&session.relay, entry, now, &robot, why);
            changed = true;
        }
        self.datagram = Some(session);
        if retried {
            self.sync_datagram().await;
            changed = true;
        }
        if changed {
            self.publish().await;
        }
    }

    /// Moves `robot`'s realtime parts back to the session's links, and sets
    /// when the path is tried again.
    fn move_back(
        relay: &DatagramRelay,
        entry: &mut Entry,
        now: Instant,
        robot: &ParticipantId,
        why: &str,
    ) {
        entry.path.fall_back();
        drop_routes(relay, entry);
        entry.time.switch(now, false, true);
        entry.retry_at = Some(now + entry.retry_after);
        let retry = entry.retry_after;
        entry.retry_after = (entry.retry_after * 2).min(RETRY_MAX);
        let time = entry.time.at(now);
        warn!(
            %robot,
            fallbacks = time.fallbacks,
            on_udp_s = time.datagram.as_secs(),
            on_tcp_s = time.session.as_secs(),
            retry_s = retry.as_secs(),
            "{why}: its realtime parts ride TCP, and go back on the datagram path when its \
             probes arrive, from {}s on (is UDP port {} open both ways?)",
            retry.as_secs(),
            relay.port()
        );
    }

    /// `robot`'s routes, while it is on the path.
    pub(in crate::host) fn routes_of(&self, robot: &ParticipantId) -> Option<DatagramRoutes> {
        let entry = self.datagram.as_ref()?.robots.get(robot)?;
        if !lock_active(&entry.path) {
            return None;
        }
        let downs = entry.downs.iter().map(|d| DatagramRoute {
            route: d.id,
            channel: d.channel,
            up: false,
        });
        let ups = entry.ups.iter().map(|u| DatagramRoute {
            route: u.id,
            channel: u.channel,
            up: true,
        });
        Some(DatagramRoutes {
            probe: entry.path.probe,
            routes: downs.chain(ups).collect(),
        })
    }

    /// The path `robot`'s realtime parts ride, and the host's counters for
    /// it on the datagram path.
    pub(in crate::host) fn data_path(
        &self,
        robot: &ParticipantId,
    ) -> (DataPath, Option<DatagramStats>) {
        let (path, stats, _) = self.data_path_time(robot);
        (path, stats)
    }

    /// [`Self::data_path`], with the time its realtime parts spent on each
    /// path since it attached, and the moves between them.
    pub(in crate::host) fn data_path_time(
        &self,
        robot: &ParticipantId,
    ) -> (DataPath, Option<DatagramStats>, Option<PathTime>) {
        let is_robot = self
            .authority
            .participant(robot)
            .is_some_and(|p| p.has_role(Role::Embodiment));
        if !is_robot {
            return (DataPath::Unknown, None, None);
        }
        let Some(entry) = self.datagram.as_ref().and_then(|d| d.robots.get(robot)) else {
            return (DataPath::Session, None, None);
        };
        let stats = entry.path.stats();
        let routed = !entry.downs.is_empty() || !entry.ups.is_empty();
        let path = if stats.active && routed {
            DataPath::Datagram
        } else {
            DataPath::Session
        };
        (path, Some(stats), Some(entry.time.at(Instant::now())))
    }

    /// Logs the path a robot that just attached takes.
    pub(super) fn log_data_path(&self, robot: &ParticipantId) {
        let Some(info) = self.authority.participant(robot) else {
            return;
        };
        if !info.has_role(Role::Embodiment) {
            return;
        }
        let asks = info.supports.iter().any(|f| f == LINK_DATAGRAM);
        let path = match (&self.datagram, asks) {
            (Some(_), true) => "udp",
            (Some(_), false) => "tcp (the robot does not take the datagram path)",
            (None, _) if self.config.datagram.is_some() => "tcp (a virtual-clock session)",
            (None, _) => "tcp",
        };
        info!(%robot, "realtime parts over {path}");
    }
}

fn lock_active(path: &RobotPath) -> bool {
    path.stats().active
}
