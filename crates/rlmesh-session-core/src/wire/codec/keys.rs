//! Key deliveries and the issued keys they carry.

use super::convert::{bad_enum, channels, collect};
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

/// Decodes a key delivery.
pub fn decode_key_delivery(buf: &[u8]) -> Result<KeyDelivery, WireError> {
    let d = pb::KeyDelivery::decode(buf)?;
    Ok(KeyDelivery {
        keys: collect(d.keys, key_from_pb)?,
        retire_through: d.retire_through,
    })
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
