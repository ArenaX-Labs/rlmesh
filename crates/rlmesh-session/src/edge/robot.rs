//! The embodiment runtime language bindings wrap: one robot map declared,
//! every part published, every commandable part served by an edge.
//!
//! [`Robot::declare`] sends the authored map (the host normalises it, §3.3),
//! then declares a publisher for every part (an image publisher, with its
//! bitrate budget and JPEG encoder, for `u8 [H, W, 3]` images) and one
//! [`Edge`] per command group (or per ungrouped commandable part).
//! Publishers exist from the start, so the host's publication plans are
//! installed and acknowledged before any controller asks for a lease
//! (§8.2, install before authority).
//!
//! The robot's control loop calls [`Robot::due`] once per control tick per
//! command group: that call *is* the edge tick (§9), so what it returns is
//! exactly what the `_applied` stream records. Nothing here calls back into
//! the robot's code from another thread: a group handover is returned with
//! the tick, the robot switches its control mode, then calls
//! [`Robot::handover_done`].

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::wire::{ChannelId, PartSpec};
use crate::{ImagePublisher, Offer, Participant, SamplePublisher, SessionError};
use bytes::Bytes;

use crate::edge::core::{EdgeConfig, EdgeStats, EdgeTick};
use crate::edge::error::EdgeError;
use crate::edge::queue::QueuedAction;
use crate::edge::runtime::Edge;
use crate::wire::EdgeMode;

/// How one part is published.
enum Outlet {
    Sample(SamplePublisher),
    Image(ImagePublisher),
}

/// A handover between two members of a command group (§9.7), by path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Handover {
    /// The group.
    pub group: String,
    /// The member that held the lease.
    pub from: String,
    /// The member that holds it now.
    pub to: String,
}

/// What the robot does this control tick.
#[derive(Clone, Debug)]
pub struct Due {
    /// The part the tick concerns (the group's active or last-driven
    /// member).
    pub path: String,
    /// The edge mode.
    pub mode: EdgeMode,
    /// The command to dispatch to `path` (packed at its command stride,
    /// validated and clamped), if any.
    pub action: Option<QueuedAction>,
    /// The declared safe behaviour to run, if it must run.
    pub safe: Option<Arc<str>>,
    /// Set on the tick a handover starts: switch the control mode, then
    /// call [`Robot::handover_done`]. The group dispatches nothing until
    /// then.
    pub handover: Option<Handover>,
}

/// One declared robot: see the module docs.
pub struct Robot {
    participant: Participant,
    parts: Vec<PartSpec>,
    paths: BTreeMap<ChannelId, String>,
    outlets: BTreeMap<String, Outlet>,
    edges: Vec<Edge>,
    edge_of: BTreeMap<String, usize>,
}

impl Robot {
    /// Declares `robot_json` (the authored robot map) through
    /// `participant`, then its publishers and edges.
    pub async fn declare(
        participant: Participant,
        robot_json: &str,
        config: EdgeConfig,
    ) -> Result<Self, EdgeError> {
        let view = participant.declare_parts(robot_json).await?;
        let me = participant.id().clone();
        let parts: Vec<PartSpec> = view
            .parts
            .iter()
            .filter(|p| p.owner == me)
            .cloned()
            .collect();
        let mut outlets = BTreeMap::new();
        for part in &parts {
            let path = part.path.to_string();
            let image = matches!(part.fields.as_slice(), [field] if field.rgb8_dims().is_some());
            let outlet = if image {
                Outlet::Image(participant.image_publisher(&path).await?)
            } else {
                Outlet::Sample(participant.publisher(&path).await?)
            };
            outlets.insert(path, outlet);
        }
        let mut edges: Vec<Edge> = Vec::new();
        let mut edge_of = BTreeMap::new();
        for part in parts.iter().filter(|p| p.command.is_some()) {
            let path = part.path.to_string();
            if edge_of.contains_key(&path) {
                continue;
            }
            let edge = Edge::attach(&participant, &path, config, None).await?;
            let index = edges.len();
            for channel in edge.channels() {
                if let Some(member) = parts.iter().find(|p| p.channel == *channel) {
                    edge_of.insert(member.path.to_string(), index);
                }
            }
            edges.push(edge);
        }
        Ok(Self {
            paths: parts
                .iter()
                .map(|p| (p.channel, p.path.to_string()))
                .collect(),
            participant,
            parts,
            outlets,
            edges,
            edge_of,
        })
    }

    /// The participant.
    pub fn participant(&self) -> &Participant {
        &self.participant
    }

    /// The parts as the host normalised them, in authored order.
    pub fn parts(&self) -> &[PartSpec] {
        &self.parts
    }

    /// The part at `path`.
    pub fn part(&self, path: &str) -> Option<&PartSpec> {
        self.parts.iter().find(|p| p.path.as_str() == path)
    }

    /// The path of one of this robot's streams.
    pub fn path_of(&self, channel: ChannelId) -> Option<&str> {
        self.paths.get(&channel).map(String::as_str)
    }

    /// Publishes one sample of the part at `path`, captured at `capture_ns`
    /// (this participant's clock), from each field's bytes in declaration
    /// order (one for a single-field part; UTF-8 for text). Copies the
    /// values, so the caller may reuse its buffers at once (§8.11). Returns
    /// the sample's seq, or `None` when the plan coalesced it or an image
    /// went to its encoder (which assigns the seq when it sends).
    pub fn publish(
        &self,
        path: &str,
        capture_ns: i64,
        fields: &[&[u8]],
    ) -> Result<Option<u64>, SessionError> {
        match self.outlet(path)? {
            Outlet::Sample(publisher) => {
                let spec = publisher.spec();
                let packed =
                    spec.layout
                        .pack(fields)
                        .map_err(|reason| SessionError::InvalidValue {
                            channel: spec.channel,
                            reason: format!("{path}: {reason}"),
                        })?;
                publisher.publish(capture_ns, Bytes::from(packed))
            }
            Outlet::Image(publisher) => match fields {
                [rgb] => publisher.offer(capture_ns, rgb).map(|offer| match offer {
                    Offer::Queued { .. } | Offer::OverBudget | Offer::Coalesced => None,
                }),
                _ => Err(SessionError::InvalidValue {
                    channel: publisher.spec().channel,
                    reason: format!("{path}: {} fields for an image", fields.len()),
                }),
            },
        }
    }

    /// Publishes one value of the part at `path` already packed (§10.6):
    /// for a single-field part whose size is a multiple of 8, the field's
    /// bytes. Sent without a further copy; an image goes to its encoder.
    pub fn publish_packed(
        &self,
        path: &str,
        capture_ns: i64,
        value: Bytes,
    ) -> Result<Option<u64>, SessionError> {
        match self.outlet(path)? {
            Outlet::Sample(publisher) => publisher.publish(capture_ns, value),
            Outlet::Image(_) => self.publish(path, capture_ns, &[&value]),
        }
    }

    fn outlet(&self, path: &str) -> Result<&Outlet, SessionError> {
        self.outlets
            .get(path)
            .ok_or_else(|| SessionError::UnknownPart(format!("{}/{path}", self.participant.id())))
    }

    fn edge_index(&self, path: &str) -> Result<usize, SessionError> {
        self.edge_of
            .get(path)
            .copied()
            .ok_or_else(|| match self.part(path) {
                Some(part) => SessionError::WrongChannel {
                    channel: part.channel,
                    reason: format!("{path} is not a commandable part"),
                },
                None => SessionError::UnknownPart(format!("{}/{path}", self.participant.id())),
            })
    }

    /// One control tick of the edge serving `path` (and every other member
    /// of its command group): call it once per tick per group, at the
    /// part's control rate. `None` when there is nothing to do: no command
    /// to dispatch, no safe behaviour to run and no handover (Idle, or a
    /// delta tick that dispatches nothing).
    pub fn due(&self, path: &str) -> Result<Option<Due>, SessionError> {
        let tick = self.edges[self.edge_index(path)?].tick();
        Ok(self.due_of(tick))
    }

    fn due_of(&self, tick: EdgeTick) -> Option<Due> {
        if tick.action.is_none() && tick.safe_behavior.is_none() && tick.handover.is_none() {
            return None;
        }
        let path_of = |c: ChannelId| self.paths.get(&c).cloned().unwrap_or_default();
        Some(Due {
            path: tick.channel.map(path_of).unwrap_or_default(),
            mode: tick.mode,
            action: tick.action,
            safe: tick.safe_behavior,
            handover: tick.handover.map(|h| Handover {
                group: h.group,
                from: path_of(h.from),
                to: path_of(h.to),
            }),
        })
    }

    /// The robot finished the handover of `path`'s group: it may dispatch
    /// again.
    pub fn handover_done(&self, path: &str) -> Result<(), SessionError> {
        self.edges[self.edge_index(path)?].handover_done();
        Ok(())
    }

    /// The counters of the edge serving `path`.
    pub fn edge_stats(&self, path: &str) -> Result<EdgeStats, SessionError> {
        Ok(self.edges[self.edge_index(path)?].stats())
    }

    /// Detaches (releasing nothing: an embodiment holds no leases) and
    /// closes, after the edges and publishers are undeclared.
    pub async fn detach(self) -> Result<(), SessionError> {
        let Self {
            participant,
            outlets,
            edges,
            ..
        } = self;
        drop(edges);
        drop(outlets);
        participant.detach().await
    }
}
