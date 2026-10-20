//! The Zenoh key layout of one session.
//!
//! ```text
//! rlmesh/v2/{tenant}/s/{session}/host                     host liveliness
//! rlmesh/v2/{tenant}/s/{session}/host/state               authority snapshot (queryable) + updates
//! rlmesh/v2/{tenant}/s/{session}/ctl                      control verbs (query/reply)
//! rlmesh/v2/{tenant}/s/{session}/p/{participant}          participant liveliness
//! rlmesh/v2/{tenant}/s/{session}/p/{participant}/hb       heartbeats (participant -> host)
//! rlmesh/v2/{tenant}/s/{session}/p/{participant}/hb/echo  heartbeat echoes (host -> participant)
//! rlmesh/v2/{tenant}/s/{session}/e/{owner}/c/{channel}    samples
//! rlmesh/v2/{tenant}/s/{session}/a/{owner}/c/{channel}    action chunks
//! rlmesh/v2/{tenant}/s/{session}/history/{channel}        history windows (queryable, host)
//! ```
//!
//! The `v2` chunk is the protocol generation
//! ([`crate::version::GENERATION`]), so peers of different generations never
//! route to each other.

use crate::ids::{ChannelId, ParticipantId, SessionId, TenantId};
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

    /// Samples of `channel`, published by its owner.
    pub fn samples(&self, owner: &ParticipantId, channel: ChannelId) -> String {
        format!("{}/e/{owner}/c/{}", self.prefix, channel.0)
    }

    /// Action chunks for `channel`, consumed by its owner's edge.
    pub fn actions(&self, owner: &ParticipantId, channel: ChannelId) -> String {
        format!("{}/a/{owner}/c/{}", self.prefix, channel.0)
    }

    /// The host's history window of `channel`.
    pub fn history(&self, channel: ChannelId) -> String {
        format!("{}/history/{}", self.prefix, channel.0)
    }

    /// The participant a `p/{participant}[/...]` key names.
    pub fn participant_of(&self, key: &str) -> Option<ParticipantId> {
        let rest = key.strip_prefix(&self.prefix)?.strip_prefix("/p/")?;
        let id = rest.split('/').next()?;
        ParticipantId::new(id).ok()
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
            keys.samples(&me, ChannelId(3)),
            format!("rlmesh/v2/acme/s/{session}/e/haply-emu/c/3")
        );
        assert_eq!(
            keys.history(ChannelId(3)),
            format!("rlmesh/v2/acme/s/{session}/history/3")
        );
        for key in [
            keys.samples(&me, ChannelId(3)),
            keys.actions(&me, ChannelId(3)),
            keys.host_state(),
            keys.ctl(),
        ] {
            // Concrete keys: no empty chunks, no wildcards.
            assert!(key.split('/').all(|c| !c.is_empty() && !c.contains('*')));
        }
    }
}
