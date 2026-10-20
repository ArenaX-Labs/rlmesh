//! The Zenoh key layout of one session (§10.3).
//!
//! ```text
//! rlmesh/v2/{tenant}/s/{session}/host                          host liveliness
//! rlmesh/v2/{tenant}/s/{session}/host/state                    authority snapshot (queryable) + updates
//! rlmesh/v2/{tenant}/s/{session}/host/tick                     virtual-mode ticks (§11.3)
//! rlmesh/v2/{tenant}/s/{session}/ctl                           control verbs (query/reply)
//! rlmesh/v2/{tenant}/s/{session}/p/{participant}               participant liveliness
//! rlmesh/v2/{tenant}/s/{session}/p/{participant}/hb            heartbeats (participant -> host)
//! rlmesh/v2/{tenant}/s/{session}/p/{participant}/hb/echo       heartbeat echoes (host -> participant)
//! rlmesh/v2/{tenant}/s/{session}/d/{robot}/{path...}           samples
//! rlmesh/v2/{tenant}/s/{session}/d/{robot}/{path...}/_applied  applied samples
//! rlmesh/v2/{tenant}/s/{session}/c/{robot}/{path...}           action chunks
//! rlmesh/v2/{tenant}/s/{session}/h/{robot}/{path...}           history store (queryable, host)
//! rlmesh/v2/{tenant}/s/{session}/l/{robot}                     lockstep step (queryable, §12)
//! ```
//!
//! The `v2` chunk is the protocol generation
//! ([`crate::version::GENERATION`]), so peers of different generations never
//! route to each other. `{robot}` is the embodiment's participant id, and
//! each path segment is one key chunk, so `d/ur10e/wrist/**` subscribes to a
//! subtree.

use crate::ids::{ParticipantId, SessionId, TenantId};
use crate::version::GENERATION;

/// Every key of one session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionKeys {
    prefix: String,
}

impl SessionKeys {
    /// The keys of `session` in `tenant`'s namespace.
    pub fn new(tenant: &TenantId, session: SessionId) -> Self {
        Self {
            prefix: format!("rlmesh/v{GENERATION}/{tenant}/s/{session}"),
        }
    }

    /// The common prefix of every key.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The host's liveliness token.
    pub fn host_liveliness(&self) -> String {
        format!("{}/host", self.prefix)
    }

    /// The authority snapshot: queryable, and published on every change.
    pub fn host_state(&self) -> String {
        format!("{}/host/state", self.prefix)
    }

    /// Virtual-mode ticks, published by the host.
    pub fn host_tick(&self) -> String {
        format!("{}/host/tick", self.prefix)
    }

    /// Control verbs (query/reply).
    pub fn ctl(&self) -> String {
        format!("{}/ctl", self.prefix)
    }

    /// A participant's liveliness token.
    pub fn participant(&self, id: &ParticipantId) -> String {
        format!("{}/p/{id}", self.prefix)
    }

    /// Every participant's liveliness token.
    pub fn participants(&self) -> String {
        format!("{}/p/*", self.prefix)
    }

    /// A participant's heartbeats.
    pub fn heartbeat(&self, id: &ParticipantId) -> String {
        format!("{}/p/{id}/hb", self.prefix)
    }

    /// Every participant's heartbeats.
    pub fn heartbeats(&self) -> String {
        format!("{}/p/*/hb", self.prefix)
    }

    /// The host's echoes of a participant's heartbeats.
    pub fn heartbeat_echo(&self, id: &ParticipantId) -> String {
        format!("{}/p/{id}/hb/echo", self.prefix)
    }

    /// Samples of one stream: `path` is a part path, or a part path with
    /// `/_applied` for its applied stream.
    pub fn data(&self, robot: &ParticipantId, path: &str) -> String {
        format!("{}/d/{robot}/{path}", self.prefix)
    }

    /// Action chunks for a commandable part, consumed by its robot's edge.
    pub fn commands(&self, robot: &ParticipantId, path: &str) -> String {
        format!("{}/c/{robot}/{path}", self.prefix)
    }

    /// The host's history of one stream.
    pub fn history(&self, robot: &ParticipantId, path: &str) -> String {
        format!("{}/h/{robot}/{path}", self.prefix)
    }

    /// Every history key (the host's queryable).
    pub fn histories(&self) -> String {
        format!("{}/h/**", self.prefix)
    }

    /// A robot's lockstep step (queryable).
    pub fn lockstep(&self, robot: &ParticipantId) -> String {
        format!("{}/l/{robot}", self.prefix)
    }

    /// The participant a `p/{participant}[/...]` key names.
    pub fn participant_of(&self, key: &str) -> Option<ParticipantId> {
        let rest = key.strip_prefix(&self.prefix)?.strip_prefix("/p/")?;
        let id = rest.split('/').next()?;
        ParticipantId::new(id).ok()
    }

    /// The robot and stream path a `h/{robot}/{path...}` key names.
    pub fn history_of<'a>(&self, key: &'a str) -> Option<(ParticipantId, &'a str)> {
        let rest = key.strip_prefix(&self.prefix)?.strip_prefix("/h/")?;
        let (robot, path) = rest.split_once('/')?;
        Some((ParticipantId::new(robot).ok()?, path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_follow_the_layout_and_parse_back() {
        let tenant = TenantId::new("acme").expect("tenant");
        let session = SessionId::new();
        let keys = SessionKeys::new(&tenant, session);
        let me = ParticipantId::new("haply-emu").expect("id");
        let ur = ParticipantId::new("ur10e").expect("id");
        assert_eq!(
            keys.heartbeat(&me),
            format!("rlmesh/v2/acme/s/{session}/p/haply-emu/hb")
        );
        assert_eq!(
            keys.participant_of(&keys.heartbeat_echo(&me)),
            Some(me.clone())
        );
        assert_eq!(
            keys.participant_of(&keys.participant(&me)),
            Some(me.clone())
        );
        assert_eq!(keys.participant_of(&keys.ctl()), None);
        assert_eq!(keys.participant_of("elsewhere/p/x"), None);
        assert_eq!(
            keys.data(&ur, "wrist/cam"),
            format!("rlmesh/v2/acme/s/{session}/d/ur10e/wrist/cam")
        );
        assert_eq!(
            keys.data(&ur, "arm/_applied"),
            format!("rlmesh/v2/acme/s/{session}/d/ur10e/arm/_applied")
        );
        assert_eq!(
            keys.commands(&ur, "arm"),
            format!("rlmesh/v2/acme/s/{session}/c/ur10e/arm")
        );
        assert_eq!(
            keys.history_of(&keys.history(&ur, "left/wrist/cam")),
            Some((ur.clone(), "left/wrist/cam"))
        );
        assert_eq!(keys.history_of(&keys.data(&ur, "arm")), None);
        assert_eq!(
            keys.lockstep(&ur),
            format!("rlmesh/v2/acme/s/{session}/l/ur10e")
        );
        for key in [
            keys.data(&ur, "arm"),
            keys.commands(&ur, "arm"),
            keys.history(&ur, "arm"),
            keys.host_state(),
            keys.host_tick(),
            keys.ctl(),
        ] {
            // Concrete keys: no empty chunks, no wildcards.
            assert!(key.split('/').all(|c| !c.is_empty() && !c.contains('*')));
        }
    }
}
