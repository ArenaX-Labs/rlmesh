//! Key deliveries and the issued keys they carry.

use super::bound;
use super::convert::{bad_enum, channels};
use super::pb;
use crate::prelude::*;

use bytes::Bytes;
use prost::Message as _;

use crate::wire::error::WireError;
use crate::wire::ids::{ChannelId, LeaseToken, ParticipantId};
use crate::wire::key::{IssuedKey, KeyBytes, KeyDelivery, KeyPurpose, Security};

/// Encodes a key delivery.
pub fn encode_key_delivery(delivery: &KeyDelivery) -> Vec<u8> {
    pb::KeyDelivery {
        keys: delivery.keys.iter().map(key_to_pb).collect(),
        retire_through: delivery.retire_through.clone(),
    }
    .encode_to_vec()
}

/// The most keys one delivery or reply carries: a generation's keys for
/// every lease, stream and member of a large session.
pub(super) const MAX_KEYS: usize = 1024;
/// The fewest bytes a key encodes to: its 32 key bytes, tagged.
pub(super) const MIN_KEY: usize = 34;
/// The most key ids one delivery retires.
const MAX_RETIRED: usize = 256;

/// Decodes a key delivery: key by key, each capped (at most 256 parts a
/// key), into vectors allocated once and fallibly.
///
/// # Errors
/// [`WireError`]: malformed, or over a cap.
pub fn decode_key_delivery(buf: &[u8]) -> Result<KeyDelivery, WireError> {
    const M: &str = "key delivery";
    Ok(KeyDelivery {
        keys: bound::each(buf, M, (1, MAX_KEYS, MIN_KEY), decode_key)?,
        retire_through: bound::each_u32(buf, M, (2, MAX_RETIRED))?,
    })
}

/// One key, capped.
pub(super) fn decode_key(buf: &[u8]) -> Result<IssuedKey, WireError> {
    bound::check(buf, &bound::KEY)?;
    key_from_pb(pb::Key::decode(buf)?)
}

pub(super) fn key_to_pb(k: &IssuedKey) -> pb::Key {
    let purpose = match k.purpose {
        KeyPurpose::Command => pb::KeyPurpose::Command,
        KeyPurpose::LockstepReply => pb::KeyPurpose::LockstepReply,
        KeyPurpose::Stream => pb::KeyPurpose::Stream,
        KeyPurpose::Estop => pb::KeyPurpose::Estop,
    };
    let security = match k.security {
        Security::Mac => pb::Security::Mac,
        Security::Seal => pb::Security::Seal,
    };
    pb::Key {
        key_id: k.key_id,
        purpose: purpose as i32,
        sender: k
            .sender
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default(),
        robot: k
            .robot
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default(),
        channels: channels(&k.channels),
        token: k.token.0,
        version: k.version,
        key: Bytes::copy_from_slice(&k.key.0),
        send: k.send,
        security: security as i32,
        generation: k.generation,
    }
}

pub(super) fn key_from_pb(k: pb::Key) -> Result<IssuedKey, WireError> {
    let purpose = match pb::KeyPurpose::try_from(k.purpose) {
        Ok(pb::KeyPurpose::Command) => KeyPurpose::Command,
        Ok(pb::KeyPurpose::LockstepReply) => KeyPurpose::LockstepReply,
        Ok(pb::KeyPurpose::Stream) => KeyPurpose::Stream,
        Ok(pb::KeyPurpose::Estop) => KeyPurpose::Estop,
        _ => return Err(bad_enum("key purpose", k.purpose)),
    };
    let security = match pb::Security::try_from(k.security) {
        Ok(pb::Security::Mac) => Security::Mac,
        Ok(pb::Security::Seal) => Security::Seal,
        _ => return Err(bad_enum("key security", k.security)),
    };
    let bytes: [u8; 32] =
        k.key.as_ref().try_into().map_err(|_| {
            WireError::malformed("key", format!("{} key bytes, not 32", k.key.len()))
        })?;
    let id = |s: String| (!s.is_empty()).then(|| ParticipantId::new(s)).transpose();
    Ok(IssuedKey {
        key_id: k.key_id,
        purpose,
        sender: id(k.sender)?,
        robot: id(k.robot)?,
        channels: k.channels.into_iter().map(ChannelId).collect(),
        token: LeaseToken(k.token),
        version: k.version,
        key: KeyBytes(bytes),
        send: k.send,
        security,
        generation: k.generation,
    })
}
