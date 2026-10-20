//! Identifiers: sessions, tenants, participants, channels, epochs and lease
//! tokens.

use std::fmt;
use std::str::FromStr;

use uuid::Uuid;

/// An id that cannot be used: it is malformed or would break the key layout.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid {kind} {value:?}: {reason}")]
pub struct IdError {
    /// What kind of id it was meant to be.
    pub kind: &'static str,
    /// The offending value.
    pub value: String,
    /// Why it was refused.
    pub reason: String,
}

impl IdError {
    fn new(kind: &'static str, value: &str, reason: impl Into<String>) -> Self {
        Self {
            kind,
            value: value.to_string(),
            reason: reason.into(),
        }
    }
}

/// A session id: a UUIDv7, so ids sort by creation time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionId(Uuid);

impl SessionId {
    /// A fresh UUIDv7.
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// The underlying UUID.
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.as_hyphenated().fmt(f)
    }
}

impl FromStr for SessionId {
    type Err = IdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let uuid =
            Uuid::parse_str(s).map_err(|err| IdError::new("session id", s, err.to_string()))?;
        if uuid.get_version_num() != 7 {
            return Err(IdError::new("session id", s, "not a UUIDv7"));
        }
        Ok(Self(uuid))
    }
}

/// The longest id that fits one key chunk.
pub const MAX_ID_LEN: usize = 128;

/// Checks that `value` can be one Zenoh key-expression chunk: 1 to
/// [`MAX_ID_LEN`] bytes of `[A-Za-z0-9._-]`. That keeps ids free of `/`,
/// wildcards and the `@`/`$` verbatim markers.
fn validate_key_chunk(kind: &'static str, value: &str) -> Result<(), IdError> {
    let ok = !value.is_empty()
        && value.len() <= MAX_ID_LEN
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if ok {
        Ok(())
    } else {
        Err(IdError::new(
            kind,
            value,
            format!("must be 1-{MAX_ID_LEN} characters of [A-Za-z0-9._-]"),
        ))
    }
}

/// A participant id, unique within a session and safe as a key chunk.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParticipantId(String);

impl ParticipantId {
    /// Validates `id` as a key chunk.
    pub fn new(id: impl Into<String>) -> Result<Self, IdError> {
        let id = id.into();
        validate_key_chunk("participant id", &id)?;
        Ok(Self(id))
    }

    /// The id as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ParticipantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for ParticipantId {
    type Err = IdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

/// A tenant namespace, safe as a key chunk.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TenantId(String);

impl TenantId {
    /// Validates `id` as a key chunk.
    pub fn new(id: impl Into<String>) -> Result<Self, IdError> {
        let id = id.into();
        validate_key_chunk("tenant id", &id)?;
        Ok(Self(id))
    }

    /// The id as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for TenantId {
    type Err = IdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

/// A channel, by its stable in-session id (chosen by the declarer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChannelId(pub u32);

impl fmt::Display for ChannelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ch{}", self.0)
    }
}

/// The host's epoch. It only increases; messages from an older one are
/// dropped. Zero means "unchecked" on the wire, so sessions start at 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch(pub u64);

impl Epoch {
    /// The epoch a fresh session starts in.
    pub const FIRST: Self = Self(1);

    /// The next epoch.
    #[must_use]
    pub fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

/// A lease token: unique and increasing within a session, so a preemption
/// always issues a token the old holder has never seen (a fencing token).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LeaseToken(pub u64);

impl fmt::Display for LeaseToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_ids_are_v7_and_round_trip() {
        let id = SessionId::new();
        assert_eq!(id.as_uuid().get_version_num(), 7);
        let parsed: SessionId = id.to_string().parse().expect("parses");
        assert_eq!(parsed, id);
        assert!(
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
                .parse::<SessionId>()
                .is_err()
        );
        assert!("not-a-uuid".parse::<SessionId>().is_err());
    }

    #[test]
    fn participant_and_tenant_ids_must_be_key_safe() {
        assert!(ParticipantId::new("ur10-sim.lane_0").is_ok());
        let long = "x".repeat(MAX_ID_LEN + 1);
        for bad in ["", "a/b", "*", "a b", "$x", "@x", "**", long.as_str()] {
            assert!(ParticipantId::new(bad).is_err(), "{bad:?} accepted");
            assert!(TenantId::new(bad).is_err(), "{bad:?} accepted");
        }
        let err = ParticipantId::new("a/b").expect_err("refused");
        assert_eq!(err.kind, "participant id");
    }

    #[test]
    fn epochs_saturate() {
        assert_eq!(Epoch(u64::MAX).next(), Epoch(u64::MAX));
        assert_eq!(Epoch::FIRST.next(), Epoch(2));
    }
}
