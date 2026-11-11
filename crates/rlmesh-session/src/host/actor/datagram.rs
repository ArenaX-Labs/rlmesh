//! The session's robots on the host's datagram path: their routes, kept in
//! step with their parts and bindings, and the fallbacks either end makes.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rlmesh_session_core::datagram::{self, HEADER_LEN, MAX_DATAGRAM};
use tracing::{info, warn};

use crate::host::HostActor;
use crate::host::datagram::{DatagramRelay, Down, RobotPath, Up};
use crate::transport::{TrafficClass, declare_publisher, stream_qos};
use crate::wire::version::LINK_DATAGRAM;
use crate::wire::{
    AuthoritySnapshot, ChannelId, ChannelPriority, DataPath, DatagramRoute, DatagramRoutes,
    DatagramStats, PartSpec, ParticipantId, Role,
};

/// The control link counts as healthy while the robot's heartbeats arrive
/// at least this often.
const HEALTHY: Duration = Duration::from_millis(300);
/// The least time the host waits for a robot's samples over the path,
/// while its control link is healthy, before it gives the path up.
const UP_SILENCE: Duration = Duration::from_secs(1);
/// What a board publishes a stream nobody demands at, at most
/// (`rlmesh_device::robot::UNDEMANDED_HZ`).
const UNDEMANDED_HZ: f64 = 10.0;

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
    /// The board's own fallbacks, as its heartbeats last said.
    board_fallbacks: u32,
    /// The up path's silence counts from here: reset while the control
    /// link is not healthy.
    up_clock: Instant,
    /// The longest a healthy board's samples may stop.
    up_window: Duration,
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
                session.robots.insert(
                    robot.clone(),
                    Entry {
                        path,
                        downs: Vec::new(),
                        ups: Vec::new(),
                        parts: Vec::new(),
                        board_fallbacks: 0,
                        up_clock: Instant::now(),
                        up_window: UP_SILENCE,
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
        let mut slowest = f64::INFINITY;
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
                    let hz = if part.hz > 0.0 {
                        part.hz.min(UNDEMANDED_HZ)
                    } else {
                        UNDEMANDED_HZ
                    };
                    slowest = slowest.min(hz);
                    ups.push(up);
                }
            }
        }
        let Some(entry) = session.robots.get_mut(robot) else {
            return;
        };
        entry.up_window = if slowest.is_finite() {
            UP_SILENCE.max(Duration::from_secs_f64(3.0 / slowest))
        } else {
            UP_SILENCE
        };
        entry.up_clock = Instant::now();
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

    /// Every tick: a chunk that turned out too large, a board that gave the
    /// path up, and a board whose samples stopped while its control link is
    /// healthy each move that robot back to TCP for this attach.
    pub(super) async fn check_datagram(&mut self) {
        let Some(mut session) = self.datagram.take() else {
            return;
        };
        let now = Instant::now();
        let now_ns = self.now();
        let mut changed = false;
        let robots: Vec<ParticipantId> = session.robots.keys().cloned().collect();
        for robot in robots {
            let healthy = self
                .links
                .get(&robot)
                .and_then(|l| l.stats.last_arrival_ns())
                .is_some_and(|at| {
                    now_ns.saturating_sub(at) < i64::try_from(HEALTHY.as_nanos()).unwrap_or(0)
                });
            let board_fallbacks = self
                .links
                .get(&robot)
                .and_then(|l| l.datagram)
                .map_or(0, |d| d.fallbacks);
            let Some(entry) = session.robots.get_mut(&robot) else {
                continue;
            };
            if !lock_active(&entry.path) {
                continue;
            }
            if board_fallbacks > entry.board_fallbacks {
                entry.board_fallbacks = board_fallbacks;
                entry.path.deactivate();
                drop_routes(&session.relay, entry);
                warn!(%robot, "the board gave the datagram path up: its realtime parts ride TCP until it attaches again");
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
            if entry.ups.is_empty() || !healthy {
                entry.up_clock = now;
                continue;
            }
            let silent = entry
                .path
                .up_silence(now)
                .min(now.saturating_duration_since(entry.up_clock));
            if silent > entry.up_window {
                entry.path.fall_back();
                drop_routes(&session.relay, entry);
                warn!(
                    %robot,
                    silent_ms = silent.as_millis(),
                    reachable = entry.path.reachable(),
                    "no samples over the datagram path while the robot's heartbeats arrive: its \
                     realtime parts ride TCP until it attaches again (is UDP port {} open?)",
                    session.relay.port()
                );
                changed = true;
            }
        }
        self.datagram = Some(session);
        if changed {
            self.publish().await;
        }
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
        let is_robot = self
            .authority
            .participant(robot)
            .is_some_and(|p| p.has_role(Role::Embodiment));
        if !is_robot {
            return (DataPath::Unknown, None);
        }
        let Some(entry) = self.datagram.as_ref().and_then(|d| d.robots.get(robot)) else {
            return (DataPath::Session, None);
        };
        let stats = entry.path.stats();
        let routed = !entry.downs.is_empty() || !entry.ups.is_empty();
        let path = if stats.active && routed {
            DataPath::Datagram
        } else {
            DataPath::Session
        };
        (path, Some(stats))
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
